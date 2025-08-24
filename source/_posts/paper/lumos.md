---
title: "lumos:Efficient Performance Modeling and Estimation for Large-scale LLM Training" 
mathjax: true
categories:
  - 论文阅读
tags:
  - paper
description: lumos模拟器论文记录
---

<!-- more -->

---

> **一句话总结**：**Lumos** 是一个**基于运行时 trace 的建模/模拟工具**，从 PyTorch Kineto 等采集到的事件中**自动恢复精细的执行图**（含算-通重叠与跨流依赖），并支持在不重新跑模型的情况下，对 **DP/PP/模型结构** 做 “what-if” 修改与**快速估算**；在 **512×H100** 集群上回放**平均误差约 3.3%**。([arXiv][1])

---

## 1. 核心贡献与定位

* **精细执行图**：仅用框架内置的 profiler（如 PyTorch Kineto）即可从 CPU/GPU 事件恢复**四类关键依赖**（CPU→GPU、GPU→CPU、同流顺序、**跨流事件**），精准表达算-通重叠与同步关系。([arXiv][1])
* **图编辑 & 快速外推**：在**不改动模型/系统**的前提下，从原始 trace-graph 出发，对 **数据并行（DP）**、**流水并行（PP）** 与**模型层数/隐藏维度**做图级改写，再用模拟器**重放一整个迭代**估算性能。([arXiv][1])
* **高精度回放**：在生产集群 **最多 512×H100**、多种 GPT-3 变体、不同并行策略下，**迭代时间回放平均误差 3.3%**，并能再现实测的执行细分占比。([arXiv][1])

---

## 2. Lumos 如何从 trace 构建执行图

* **事件来源**：直接使用 PyTorch/TensorFlow 的**内置 profiler**（如 Kineto），无需对模型或框架做侵入式改造。([arXiv][1])
* **依赖建模（四类）**：

  1. **CPU→GPU（launch）**：用 **correlation ID** 绑定 CPU 端的 `cudaLaunchKernel`/`cudaMemsetAsync` 与对应的 GPU kernel。
  2. **GPU→CPU（同步）**：`cudaDeviceSync`/`cudaStreamSync` 等需要等到相关 GPU kernel 完成。
  3. **同流顺序**：同一 CUDA stream 内核严格顺序。
  4. **跨流事件**：`cudaEventRecord` 与 `cudaStreamWaitEvent` 形成“记录→等待”的跨流依赖，表达**不同流**间的有序性。([arXiv][1])

---

## 3. 图编辑：支持哪些 “what-if” 改动

* **数据并行（DP）**：只需**调整通信任务**（如梯度规约类）的执行时间；本地计算不变。([arXiv][1])
* **流水并行（PP）**：

  * 先按所选调度（如 **1F1B**）更新各微批的前后向顺序；
  * 将原图中任务**按层聚类**后重分配到新 stage；
  * 在 stage 边界插入/重连**激活与梯度**的 send/recv；
  * 保留原 trace 中的**依赖模式**以保证可重放正确性。([arXiv][1])
* **模型结构**：

  * **隐藏维度**变更：重写相关算子/内核的输入张量维度并**重估时长**；
  * **层数**变更：复制/删减层块并**重连依赖与通信**。([arXiv][1])
* **暂不支持**：**修改 Tensor Parallelism（TP）**（通常受限于单机且通信重，留作未来工作）。([arXiv][1])

---

## 4. 模拟器：事件驱动流程（论文算法 1 的要点）

* 维护两个集合：

  * **固定依赖**（初始化阶段一次性确定，如同线程/同流顺序、CPU→GPU 的 launch 边）；
  * **运行期依赖**（例如 `cudaStreamSync` 需要等待\*\*该流上“最后一个 kernel”\*\*完成，这个“最后”要在调度时才能确定）。
* 主循环：从就绪集合取任务 → 分配到其“处理器”（CPU 线程/CUDA stream）上运行 → 更新处理器可用时间与后继任务的最早可启动时间；若仍有运行期依赖未满足则**延后**。([arXiv][1])

---

## 5. 评测设置与关键数字

* **规模与环境**：**最多 512×H100（32 台主机）**，RoCE 数据中心网络（**每主机 8×400Gbps**），CUDA **12.4**，PyTorch **2.5**，Transformer Engine **0.12.0**，Lightning **1.9.4**。([arXiv][1])
* **对比基线**：与 **dPRO**（trace-driven 回放系统）相比，Lumos 在复杂并行配置下能更好捕捉**跨流依赖**与算-通重叠，显著降低回放误差。([arXiv][1])
* **结果**：回放**平均误差 \~3.3%**；并展示在 **DP/PP/结构**外推时的估算准确性与执行细分（暴露计算/暴露通信/重叠/其他）。([arXiv][1])

---

## 6. 工程实现与使用门槛

* **实现规模**：约 **5,200 行 Python**。([arXiv][1])
* **接入成本**：在训练代码里**插入 \~10 行 profiler hook** 采集 Kineto trace，随后走**自动化**流程：**建图 → 图编辑 → 模拟估算**。([arXiv][1])

---

## 7. 适用/不适用场景

* **适用**：

  * 需要在真实机群外**快速比较**并行/结构配置（DP/PP/层数/隐藏维）并估算收益；
  * 需要**高保真回放**来定位算-通重叠与跨流同步处的性能瓶颈。
* **当前不适用/注意**：

  * **修改 TP** 的外推（论文暂未支持）；
  * 追求 FLOPs、内存、带宽、能耗等**系统级指标**（论文称为**后续计划**）；
  * 估算假设**新配置可正常运行**（不考虑 OOM 等失效情形）。([arXiv][1])

---

## 8. 与既有工作的关系（示例：dPRO）

* **dPRO** 同样是 trace-driven 的性能诊断/回放系统，但在复杂 LLM 并行下，**跨流依赖**与重叠的精细建模更困难，容易导致过度乐观的并行预测；Lumos 在这些方面做了系统增强并显著降低误差。([arXiv][1])

---

## 9. 论文与会议信息（可引用）

* 论文（arXiv）：**“Lumos: Efficient Performance Modeling and Estimation for Large-scale LLM Training”**（2025-04-12 首次提交）。([arXiv][2])
* PDF（作者主页镜像/MLSys 论文）：可下载全文。([mingyu-liang.github.io][3])
* **MLSys 2025** 接收与日程页面（含报告/录播入口）。([mlsys.org][4])

---

## 10. 代码与开源状态（截至 2025-08-15）

* **未见官方代码库链接**（arXiv/MLSys 页面与作者 PDF 中均未提供），社区里存在**同名但无关**的 “Lumos” 项目（如 Agent/视频生成/视觉等），注意区分。([arXiv][2], [mlsys.org][4], [GitHub][5])

---

## 11. 快速上手（示意）

> 采集 **Kineto trace** → 构建执行图 → 在图上编辑（DP/PP/结构）→ 模拟回放/估算。
> 论文正文给出了典型的 **PyTorch profiler** 用法示意与全流程示意图。([arXiv][1])

---

## 12. 你可能关心的细节（精炼版）

* **为什么更准？**

  * 用 **correlation ID** 串起 CPU launch 与 GPU kernel；
  * 显式恢复 **跨流事件**（Record/Wait）与**同步**（Stream/Device Sync）；
  * 在模拟器中将依赖分为**固定**与**运行期**，确保像 “**等待该流最后一个 kernel**” 这类语义被正确表达。([arXiv][1])
* **改 DP/PP 怎么算？**

  * **DP**：只重赋通信任务时长；
  * **PP**：更新调度（如 1F1B）→ 任务按层分组并**重分 stage** → 在边界插入 send/recv → 保持依赖闭合。([arXiv][1])

---

### 参考文献 / 链接

* Lumos 论文（arXiv 页面与 PDF）：([arXiv][2])
* Lumos（MLSys 2025 会议页面/日程/录播）：([mlsys.org][4])
* dPRO（trace-driven 回放基线论文）：([arXiv][6])

---

> 注：本文档只摘取对工程落地最关键的事实与方法，更多图例（如 **PP×TP** 微批调度示例）与完整算法细节请参阅原论文正文与附图。([arXiv][1])

---

[1]: https://arxiv.org/pdf/2504.09307 "Lumos: Efficient Performance Modeling and Estimation for Large-scale LLM Training"
[2]: https://arxiv.org/abs/2504.09307?utm_source=chatgpt.com "Lumos: Efficient Performance Modeling and Estimation for Large-scale LLM Training"
[3]: https://mingyu-liang.github.io/files/mlsys25-lumos.pdf?utm_source=chatgpt.com "Lumos: Efficient Performance Modeling and Estimation for ..."
[4]: https://mlsys.org/virtual/2025/papers.html?utm_source=chatgpt.com "MLSys 2025 Papers"
[5]: https://github.com/allenai/lumos?utm_source=chatgpt.com "Code and data for \"Lumos: Learning Agents with Unified ..."
[6]: https://arxiv.org/pdf/2205.02473?utm_source=chatgpt.com "dPRO: A Generic Performance Diagnosis and Optimization ..."

