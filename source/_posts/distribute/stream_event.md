---
title: pytorch中的stream和event
mathjax: true
categories:
  - 分布式基础
tags:
  - attention
description: PyTorch 中的 Stream / Event 与跨流同步：原理、用法与可运行示例
---

<!-- more -->


---

> 一句话总览：**流（stream）是 GPU 上的“有序指令队列”，事件（event）是插在流时间线上的“栅栏/时间戳”。**把 `event.record()` 放在生产流上，再在消费流里 `wait_event()`，就能做到**设备侧**的无阻塞依赖编排。([docs.pytorch.org][1])

---

## 1. 基本概念

* **Stream（流）**：同一条流内按提交顺序（FIFO）执行；不同流彼此独立，可并行运行。PyTorch 的 `torch.cuda.Stream` 就是 CUDA 流的封装，并提供 `record_event / wait_event / wait_stream / synchronize` 等方法。([docs.pytorch.org][1])
* **Event（事件）**：同步标记。可用于**测时**与**跨流同步**：在生产流 `record()`，在消费流 `wait()`/`wait_event()`。事件也可 `elapsed_time()` 读取**GPU 端**的毫秒计时。([docs.pytorch.org][2])
* **默认流语义**：

  * **Legacy default stream** 会与其它（阻塞型）流**互相同步**；
  * **Per-thread default stream（PTDS）** 不与其他流同步，行为更像显式创建的流。
    两者可在编译/宏层面选择，行为不同会影响是否“自动同步”。([NVIDIA Docs][3])

---

## 2. 三种“等待”的作用域（越小越好）

* **设备级**：`torch.cuda.synchronize(device)` —— 等**该设备上所有流**到当前为止的工作完成。**最重**，一般少用。（语义等同 `cudaDeviceSynchronize`）([developer.download.nvidia.com][4])
* **单流级**：`stream.synchronize()` —— 只等**这一条流**已提交的工作，等同 `cudaStreamSynchronize`。([docs.pytorch.org][1])
* **事件级**：`event.synchronize()` —— 只等**该事件**所捕获的工作，等同 `cudaEventSynchronize`。**粒度最细**，推荐优先用事件来表达依赖。([docs.pytorch.org][2])

> 口诀：**device > stream > event**（等待范围从大到小）。选**最小必要范围**，保留并行度。([developer.download.nvidia.com][4])

---

## 3. 跨流同步的三种方式

1. **事件栅栏（推荐）**

   * 生产流：`event.record()`
   * 消费流：`consumer.wait_event(event)`（或 `event.wait(consumer)`）
     该调用**立即返回**，只是把“等待 e”这条依赖写进了消费流的队列；后续提交的工作都会在 e 完成后执行。([docs.pytorch.org][1])

2. **流-流等待**

   * `this.wait_stream(that)`：让 **this** 流后续工作，等待 **that** 流**当前已提交**的工作完成。([docs.pytorch.org][1])

3. **默认流语义（历史兼容）**

   * 若使用 legacy default stream，它会与其它阻塞流互相同步；PTDS 则不会。新代码不建议依赖这种“隐式同步”。([NVIDIA Docs][3])

---

## 4. 张量生命周期的\*\*安全（safe）\*\*用法

跨流共享同一块显存时，除了“写清楚依赖”（事件/流等待），还应在**使用该张量的流**上调用：

```python
tensor.record_stream(consumer_stream)
```

这会告诉 CUDA 缓存分配器：**该张量也在 consumer\_stream 上被用过**，从而避免在生产流释放后被过早复用，造成潜在读写竞态。否则需要在释放前把使用**同步回创建流**。([docs.pytorch.org][5])

---

## 5. CPU↔GPU 拷贝与 **non\_blocking** / **pinned memory**

* 只有当\*\*页锁定内存（pinned）\*\*参与时，很多拷贝才能真正异步化并与计算重叠；PyTorch 教程对 `pin_memory()` 与 `non_blocking=True` 的行为做了系统说明。([docs.pytorch.org][6])
* 读取 D2H 结果前，应**等待拷贝完成**（事件或同步），不要直接在 CPU 端消费异步结果。([docs.pytorch.org][6])

**推荐模式（D2H 拷贝不“卡住”整机，只在用到结果时小范围等待）**：

```python
import torch
x  = torch.randn(1_000_000, device="cuda")
dst = torch.empty_like(x, device="cpu", pin_memory=True)  # pinned CPU buffer
copy_stream = torch.cuda.Stream()
copy_done   = torch.cuda.Event()

with torch.cuda.stream(copy_stream):
    dst.copy_(x, non_blocking=True)  # 异步 D2H
    copy_done.record()               # 仅拷贝完成处打点

# ……CPU 可以先做别的活……
copy_done.synchronize()              # 只有在真正要用 dst 时才等这一次
print(dst[:5])
```

> 要点：**pinned + 专用拷贝流 + 事件**；避免用设备级 `torch.cuda.synchronize()` 粗暴“刹车”。([docs.pytorch.org][6], [developer.download.nvidia.com][4])

---

## 6. 可运行最小示例

### 6.1 计算流 → 通信/后处理流（事件栅栏）

```python
import torch
device = "cuda"

compute = torch.cuda.Stream()
comm    = torch.cuda.Stream()
done    = torch.cuda.Event()

x = torch.randn(1_000_000, device=device)

with torch.cuda.stream(compute):
    y = x.relu()
    done.record()           # 记录“y 已就绪”

comm.wait_event(done)       # 让 comm 流等到 y 就绪
with torch.cuda.stream(comm):
    z = y * 2               # 在 GPU 端自动等待，不阻塞 CPU

torch.cuda.synchronize()    # 示例收尾：真实工程里可继续提交后续工作
```

**机制说明**：`wait_event` 把“等待 e”插入到消费流队列，只有**事件触发**后，消费流后续 kernel 才会执行；这都是**设备侧**完成，CPU 不被阻塞。([docs.pytorch.org][1])

### 6.2 三流示例（S2 与 S3 都等 S1）

```python
s1, s2, s3 = torch.cuda.Stream(), torch.cuda.Stream(), torch.cuda.Stream()
e = torch.cuda.Event()

with torch.cuda.stream(s1):
    a = torch.randn(1024, 1024, device="cuda") @ torch.randn(1024, 1024, device="cuda")
    e.record()

s2.wait_event(e)
s3.wait_event(e)

with torch.cuda.stream(s2):
    b = a.relu_()
with torch.cuda.stream(s3):
    c = a.sum()
```

> 同一个事件可以被多条流等待，适合“一对多”的依赖。([docs.pytorch.org][1])

### 6.3 GPU 端**精准计时**（Event `elapsed_time`）

```python
import torch
s = torch.cuda.Stream()
start = torch.cuda.Event(enable_timing=True)
end   = torch.cuda.Event(enable_timing=True)

x = torch.randn(4096, 4096, device="cuda")
w = torch.randn(4096, 4096, device="cuda")

# 预热
for _ in range(2): (x @ w).sum().relu_()

with torch.cuda.stream(s):
    start.record()
    y = (x @ w).relu_()
    end.record()

end.synchronize()
print(f"elapsed = {start.elapsed_time(end):.3f} ms")
```

> `elapsed_time` 返回 **start.record 与 end.record 之间**的 GPU 毫秒数；`end.synchronize()` 确保测量闭区间已完成。([docs.pytorch.org][2])

---

## 7. 常见坑与速记

* **事件位置要对**：`record()` 只覆盖它之前已入队的工作；之后新提交的工作不包含在本事件内。使用时将 `record()` 放在**生产结束点**。([docs.pytorch.org][1])
* **`wait_event`/`wait_stream` 均为“写依赖、立即返回”**：它们不会阻塞 CPU，只影响**后续提交**到该流的工作。([docs.pytorch.org][1])
* **默认流陷阱**：Legacy 与 PTDS 语义不同。混用时，legacy 会与阻塞流互相等待；PTDS 不会。新工程建议**显式建流 + 显式同步**，避免踩隐式同步。([NVIDIA Docs][3])
* **流优先级**：低数字=高优先级；只是“倾向”，**不抢占**已在运行的 kernel。([NVIDIA Docs][7])

---

## 8. 术语一页纸

* **Stream**：设备上独立的有序执行队列。`record_event`、`wait_event`、`wait_stream`、`synchronize`。([docs.pytorch.org][1])
* **Event**：设备侧栅栏/时间戳；`record`、`wait`、`synchronize`、`elapsed_time`。([docs.pytorch.org][2])
* **安全跨流**：写依赖 + `tensor.record_stream(consumer)`（或手动确保释放前同步回创建流）。([docs.pytorch.org][5])
* **高效 D2H**：pinned + 专用拷贝流 + 事件；按需等待，避免全设备同步。([docs.pytorch.org][6])

---

### 参考资料（强烈建议细读原文）

* PyTorch：`torch.cuda.Stream` API（含 `wait_event / wait_stream / synchronize`）与文档注释。([docs.pytorch.org][1])
* PyTorch：`torch.cuda.Event` API（`record / wait / synchronize / elapsed_time`）。([docs.pytorch.org][2])
* PyTorch：`tensor.record_stream`（跨流内存生命周期管理）。([docs.pytorch.org][5])
* PyTorch 教程：`pin_memory()` 与 `non_blocking` 使用与注意事项。([docs.pytorch.org][6])
* NVIDIA CUDA 文档：默认流（Legacy vs PTDS）语义与流优先级说明。([NVIDIA Docs][3])
* NVIDIA 培训讲义：`cudaDeviceSynchronize / cudaStreamSynchronize / cudaEvent*` 的同步对比与示例。([developer.download.nvidia.com][4])

---

[1]: https://docs.pytorch.org/docs/2.8/generated/torch.cuda.Stream.html "Stream — PyTorch 2.8 documentation"
[2]: https://docs.pytorch.org/docs/stable/generated/torch.cuda.Event.html "Event — PyTorch 2.8 documentation"
[3]: https://docs.nvidia.com/cuda/cuda-driver-api/stream-sync-behavior.html "CUDA Driver API :: CUDA Toolkit Documentation"
[4]: https://developer.download.nvidia.com/CUDA/training/StreamsAndConcurrencyWebinar.pdf "Slide 1"
[5]: https://docs.pytorch.org/docs/stable/generated/torch.Tensor.record_stream.html "torch.Tensor.record_stream — PyTorch 2.8 documentation"
[6]: https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html "A guide on good usage of non_blocking and pin_memory() in PyTorch — PyTorch Tutorials 2.8.0+cu128 documentation"
[7]: https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html "CUDA Runtime API :: CUDA Toolkit Documentation"

