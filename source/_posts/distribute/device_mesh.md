
---
title: pytorch devicemesh
date: 2025-06-14 10:00:00
categories:
  - 分布式基础
tags:
  - devicemesh
description: pytorch中devicemesh实现
---

<!-- more -->

---

## 一、为何使用 DeviceMesh？

在混合并行（DP/TP/PP/HSDP/…）中，需要管理多个子通信组（ProcessGroup），对应复杂的设备拓扑结构。`DeviceMesh` 提供了：

* 理论上无缝支持任意维度的多维拓扑；
* 自动拆分进程组(`new_group`/`split_group`)；
* 灵活切片子 Mesh；
* 经历设计周全的高效初始化方案 ([docs.pytorch.org][1], [pytorch.org][2])。

---

## 二、初始化流程

### `init_device_mesh(...)` 的作用

一个一行搞定的方法，它会：

1. 初始化全局 `init_process_group(...)`（若未初始化）；
2. 根据 `mesh_shape` 自动构造 CPU 上的 `torch.arange(...).view(...)`；
3. 创建 `DeviceMesh(...)`。内部完成子组拆分原理（见下一节）。

---

### `DeviceMesh.__init__()` + `_init_process_groups()`

* **存储**：`device_type`、`mesh`、`mesh_dim_names`；
* **通信组拆分**：遍历每个维度 `dim`：

  * 使用 `mesh.swapdims(-1, dim).reshape(-1, size(dim))` 列出该维所有子组 rank；
  * 若 NCCL 已绑定 GPU，即可用 `split_group` 一次拆出全部子组；
  * 否则使用 `new_group()` 分 group 拆；
  * 并将当前 rank 属于的那组信息放入 `self._dim_group_infos[dim]`；
* **结果**：每个维度对应一个包含当前 rank 的 `ProcessGroup` 信息列表。

```python
#pp
mesh = torch.tensor([
  [0, 1],  # pp=0
  [2, 3],  # pp=1
  [4, 5],  # pp=2
  [6, 7]   # pp=3
])

mesh.swapdims(-1, 0)

tensor([[0,2,4,6],
        [1,3,5,7]])

pg_ranks_by_dim = tmp.reshape(-1, mesh.size(0))

[
  [0,2,4,6],  # 对应 tp 行 0 各 pp 段
  [1,3,5,7]   # 对应 tp 行 1 各 pp 段
]

#tp

tmp = mesh.swapdims(-1, 1)  # 等于 transpose(1,1)，本身无变化
pg_ranks_by_dim = tmp.reshape(-1, mesh.size(1))

[
  [0,1],  # pp=0
  [2,3],
  [4,5],
  [6,7]
]
```

---

## 三、核心接口与内部实现解析

### 1. 属性与方法

```python
mesh.shape  # tuple(self.mesh.shape)
mesh.ndim   # int(self.mesh.ndim)
mesh.size(dim=None)  # 总元素数 or self.mesh.size(dim)
```

用于获取 mesh 元结构和规模，适用于判断维度数量、循环迭代、并行策略配置等场景。

---

### 2. Rank 与坐标

* `get_rank()`：等价于 `torch.distributed.get_rank()`，返回全局 rank；
* `get_local_rank(mesh_dim)`：内部调用 `get_rank(self.get_group(mesh_dim))` → 当前维度的小组内编号；
* `get_coordinate()`：返回 `self._coordinate_on_dim`，其在初始化中通过 `(self.mesh==global_rank).nonzero()` 获得。

示例：`mesh_shape=(4,2)`，rank=5 → local\_pp=2、local\_tp=1，coordinate `[2,1]`。

---

### 3. 通信组获取

* `get_group(mesh_dim)`：

  * 若 1D 且不传参，直接返回唯一子进程组；
  * 多维则根据 `mesh_dim`（索引或名字）检索 `self._dim_group_infos[dim]`，用 `_find_pg_by_ranks_and_tag()` 获取对应 `ProcessGroup`。
* `get_all_groups()`：返回所有维度的 group 列表；
* `__getitem__(dims)`：切片接口调用 `_mesh_resources._get_slice_mesh_dims(...)`，创建新的子 mesh，保留底层 communicator，但维度降。

  * 支持单维或多维切片，且返回的 submesh 顺序按传入顺序排列 ([discuss.ray.io][3], [gemfury.com][4], [pytorch.org][2])。

---

### 4. `from_group(...)` 方法

* 可接受单 group 或 group 列表；
* 创建新的 `DeviceMesh` 时不会调用 backend 初始化；
* 会复用现有 `ProcessGroup`，并填充 `_dim_group_infos`，因此 `get_group(...)` 将直接返回传入的实例，避免重复创建 group。

---

## 四、完整单机 8 卡 Demo：tp=2, pp=4

下面演示如何调用所有接口并输出结果。注意：需在 `torchrun --nproc_per_node=8` 下运行。

```python
import os, torch, torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh

def run_device_mesh_demo():
    dist.init_process_group("nccl")
    # ⬇️ 初始化 2-维 mesh：pp=4, tp=2
    mesh = init_device_mesh("cuda", mesh_shape=(4, 2), mesh_dim_names=("pp", "tp"))
    
    # ✅ rank 和坐标
    gr = mesh.get_rank()            # 全局 rank
    coord = mesh.get_coordinate()   # [pp_idx, tp_idx]
    local_pp = mesh.get_local_rank("pp")
    local_tp = mesh.get_local_rank("tp")
    
    # ⬇️ mesh 基本结构
    total = mesh.size()
    pp_size, tp_size = mesh.size("pp"), mesh.size("tp")
    ndim = mesh.ndim
    shape = mesh.shape
    
    # ⬇️ 获取通信组
    pp_group = mesh.get_group("pp")
    tp_group = mesh.get_group("tp")
    all_groups = mesh.get_all_groups()
    
    # ⬇️ 切片出子 mesh
    tp_mesh = mesh["tp"]
    pp_mesh = mesh["pp"]
    
    # ⬇️ 输出结果
    print(f"rank={gr}, coord={coord}, local_pp={local_pp}, local_tp={local_tp}")
    print(f"ndim={ndim}, shape={shape}, total={total}, pp={pp_size}, tp={tp_size}")
    print("pp_group ranks:", dist.get_process_group_ranks(pp_group))
    print("tp_group ranks:", dist.get_process_group_ranks(tp_group))
    print("all_groups sizes:", [len(dist.get_process_group_ranks(g)) for g in all_groups])
    print("tp_mesh ndim, shape:", tp_mesh.ndim, tp_mesh.shape)
    print("pp_mesh ndim, shape:", pp_mesh.ndim, pp_mesh.shape)

if __name__ == "__main__":
    run_device_mesh_demo()
```

### 💬 预期输出（例如 rank = 5）：

rank=5, coord=\[2,1], local\_pp=2, local\_tp=1
ndim=2, shape=(4,2), total=8, pp=4, tp=2
pp\_group ranks: \[4,5,6,7]
tp\_group ranks: \[5,7]
all\_groups sizes: \[4,2]
tp\_mesh ndim, shape: 1 (2,)
pp\_mesh ndim, shape: 1 (4,)


说明：
- rank=5 位于 pipeline 段 2，tp 内编号 1；
- `pp_group` 包含与其同 segment 的 4 张卡；
- `tp_group` 包含同 segment tp 维度的两张卡；
- 切片后 `tp_mesh`、`pp_mesh` 成为 1 维结构，用于后续 parallelization。

---

## 👏 总结

- `DeviceMesh` 构建自身通过 `init_device_mesh()` 完成初始化与子组拆分；
- 接口内部实现逻辑与 Group 管理机制清晰、高效；
- `__getitem__`为多维并行下子 Mesh 切片关键工具，对集成 parallel APIs 至关重要；
- 通过该机制，可以简单地组织复杂的 hybrid-parallel pipelines，同时充分复用 communicator 资源并简化开发流程。

[1]: https://docs.pytorch.org/tutorials/recipes/distributed_device_mesh.html?utm_source=chatgpt.com "Getting Started with DeviceMesh - PyTorch documentation"
[2]: https://pytorch.org/docs/stable/distributed.html?utm_source=chatgpt.com "Distributed communication package - torch.distributed"
[3]: https://discuss.ray.io/t/init-device-mesh-in-pytorch-distributed/22371?utm_source=chatgpt.com "Init device mesh in pytorch distributed - Ray Train"
[4]: https://gemfury.com/turingmotors/python%3Atorch/torch-2.1.2-cp311-cp311-linux_aarch64.whl/content/distributed/_tensor/device_mesh.py?utm_source=chatgpt.com "distributed/_tensor/device_mesh.py · turingmotors/torch - Gemfury"

