---
title: "Reducing Activation Recomputation in Large Transformer Models" 
mathjax: true
categories:
  - 训练
tags:
  - paper
description: 大规模 Transformer 激活重计算的系统级优化
---

<!-- more -->

---

## 一、论文速览

这篇论文关注的大问题是：在大规模 Transformer 模型训练中，激活（activations）占用的显存越来越夸张，为了省显存普遍使用“全层激活重计算（全 checkpoint）”，但这会带来 30%–40% 的额外算力开销。
作者从 Transformer 结构出发，建立了一套近似但非常实用的“激活内存模型”，系统分析了张量并行（TP）、序列并行（SP）、流水并行（PP）对激活内存的影响，并提出两大技术：**将序列并行与张量并行融合**，以及**对激活进行选择性重计算**。
综合起来，这些方法在不增加通信量的前提下，实现了激活显存约 **5×** 的压缩，相比“全层重算”只保留了 **~2%–7%** 的计算开销；在 22B–1T 规模模型上，迭代 throughput 提升大约 **30%**，GPU FLOPs 利用率能稳定在 50%+。

## 二、论文结构

1. **引言与相关工作**
   介绍大模型训练中的内存瓶颈、现有并行/内存优化技术（TP/PP、ZeRO、offload、已有 SP 等），并说明本文聚焦在“模型并行 + 激活内存”这个维度。适合快速了解问题背景和与其它方案关系时阅读。

2. **Transformer 结构与符号约定（Section 3）**
   统一定义 $s,b,h,a,L,t,p,v$ 等符号，并拆开分析 self-attention 与 MLP 内部的激活结构。适合在自己做推导、对接代码实现时重点看。

3. **激活内存建模与并行策略（Section 4）**
   先给出“单层激活内存近似公式”，再依次叠加：张量并行（TP）、TP+SP、再到流水并行（PP），推导出总激活内存的闭式表达，是全文最核心的理论部分。

4. **选择性激活重计算（Section 5）**
   对比“全层重算”和“只重算 attention 中一部分算子”的差异，给出在 GPT-3 / MT-NLG 规模下的内存与 FLOPs 量级，对工程上“该 checkpoint 哪些 op”给出明确指引。

5. **实验评估（Section 6）**
   通过单层 micro benchmark + 端到端训练（22B/175B/530B/1T）验证模型与实验的一致性，报告显存占用、每层时延、迭代时间、MFU/HFU 等指标，是判断“值不值的上工程实现”的关键。

6. **总结与未来工作（Section 7 + Appendix）**
   小结两大技术（TP+SP + selective recompute）的贡献，并讨论 pipeline 首段显存碎片、自动化搜索 checkpoint 策略等未来方向。

> 核心思想：针对大规模 Transformer，先用解析模型精确刻画激活内存，再通过“张量并行 + 序列并行”的组合将激活均匀分摊到各设备，并只对 FLOPs 便宜但内存巨大的子算子做选择性重计算，在几乎不增加通信、极小算力开销的前提下，实现约 5× 的激活显存压缩与 30% 左右的吞吐提升。

---

## 三、方法与系统设计

从工程视角看，本文要解决的是：

> **“如何在不崩掉训练吞吐的前提下，把激活显存压到能跑 trillion-scale 模型的水平？”**

整体思路是“两步走”：

1. **建模**：把 Transformer 每一层、每一块（attention / MLP / LayerNorm / Dropout）的激活内存用公式数清楚，顺带把 TP / SP / PP 的影响都代入进去。
2. **优化**：

   * 在结构层面：设计一种 **TP + SP 组合的并行方式**，通过 $g / \bar g$ 操作把非 TP 区域按序列切片，避免 LayerNorm/Dropout 这类激活在 TP 组内重复存储。
   * 在算子层面：只对 attention 中“大激活、低 FLOPs”那一部分做 **选择性重计算**，其它地方照常缓存，从而在“显存”和“重算开销”之间取得更优折中。

可以拆成几个具体子问题：

* *子问题 1：* 如何用一个简洁的公式刻画“单层 Transformer 的激活内存”，并能平滑代入 TP/SP/PP 等并行参数？
* *子问题 2：* 如何把序列并行和张量并行揉在一起，在不增加通信带宽的前提下，把之前 TP 里“没法切”的那一部分激活按序列分片？
* *子问题 3：* 在一层内部，哪些激活适合 checkpoint（重算），哪些应该直接存？怎样在 QKV/softmax/attention over V 这些子算子之间切分？
* *子问题 4：* 当再叠加流水并行时，第一 stage 需要存多少 micro-batch 的激活，以及如何在实践中控制 recompute 的开销不失控？

### 3.1 核心模块一览

按论文思路，把方法拆成几个“工程模块”会更清晰：

* **激活内存近似模型**：给出无并行时的“单层激活内存公式”，把 attention / MLP / LayerNorm / Dropout 各自的贡献拆开，并明确哪些可以忽略（小 buffer）。
* **张量并行（Tensor Parallel, TP）基线**：假设 TP 只切 attention / MLP 内部的大 GEMM，把激活在这些 op 内部均匀分摊到 $t$ 个设备，但 LayerNorm / Dropout 等非 TP 区域仍是每卡一份。
* **序列并行（Sequence Parallel, SP）+ 转换算子 $g/\bar g$**：在非 TP 区域沿序列维 $s$ 切分，设计 $g$（all-gather）和 $\bar g$（reduce-scatter）来在“序列切分域”和“张量切分域”之间无缝转换。
* **选择性激活重计算（Selective Activation Recomputation）**：只重算 attention 中在 $QK^\top$、softmax、softmax dropout、attention over V 区域的激活，它们内存巨大但每元素 FLOPs 不多；其余部分照常缓存。
* **与流水并行的结合（1F1B / interleaved 1F1B）**：分析在经典 1F1B 调度下，首个流水 stage 永远需要同时 hold $L$ 层激活；在此基础上给出总激活内存公式，并讨论 interleaved pipeline 时的修正因子。

### 3.2 数据流与控制流

用“从输入到 loss，再到反向”的视角，可以把数据流/控制流串成如下步骤（只关注单个 stack）：

1. **输入嵌入层**

   1. 词表 embedding：查表得到形状为 $(s, b, h)$ 的 token 表示。
   2. 加上可学习的位置编码（同形状），得到 $X^{(0)}$ 作为第 1 层输入。
   3. 在启用 SP 时，这一层的 Dropout mask 也可以按序列切分存储。

2. **第 $\ell$ 个 Transformer 层的前向（无并行视角）**

   1. LayerNorm：$Y = \text{LN}(X)$，输出仍为 $(s,b,h)$，需要缓存输入 $X$ 作为激活。
   2. Self-Attention：

      * QKV 投影：从 $Y$ 经过三次线性层得到 $Q,K,V$，尺寸 $ (s,b,h)$ 或 $(s,b,h/a)$。
      * $QK^\top$：计算注意力 logits，尺寸大约为 $(a,s,s,b)$。
      * softmax + dropout：得到注意力权重，再施加 dropout。
      * attention over V：用注意力权重加权 V，得到 $(s,b,h)$ 的输出。
      * 输出线性：再投影回 $(s,b,h)$。这些步骤产生大量中间激活。
   3. 残差 + LayerNorm：把 attention 输出加回输入，做第二次 LN。
   4. MLP：

      * 线性 $h \to 4h$，产生 $(s,b,4h)$。
      * GeLU 非线性，需要缓存输入。
      * 线性 $4h \to h$，再加 Dropout。
      * 残差加回。

3. **TP + SP 下的前向控制流（以 MLP 为例）**

   1. 在 LayerNorm 前，输入 $X$ 已按序列维切分：$[X^{s}_1, X^{s}_2, \dots, X^{s}_t]$。
   2. LayerNorm 在各 rank 本地做，输出 $[Y^{s}_1,\dots,Y^{s}_t]$，此时仍按序列切分。
   3. 为送入 MLP 中的 GEMM，需要完整序列：调用 $g$ 做 **all-gather** 把 $Y$ 在每个 TP rank 上拼成完整的 $(s,b,h)$。
   4. 线性 + GeLU + 线性内部沿隐藏维切分（标准 TP），每卡只处理 $h/t$ 或 $4h/t$ 的 slice。
   5. MLP 输出 $W_1, W_2, \dots, W_t$ 需要先求和再按序列切分给下游 Dropout，于是用 $\bar g$ 实现“求和 + 按序列 RS”的 **reduce-scatter**。
   6. Dropout、残差在序列切分域中本地完成。

4. **选择性重计算的控制流（以 attention 为主）**

   1. 正常前向时，只保留：

      * 输入 LN 前后的张量；
      * MLP 输入/输出；
      * 以及 attention 中“宽度尚未放大”的部分。
   2. 对于 $QK^\top$、softmax、softmax dropout、attention over V 等区域：

      * 不缓存中间激活，只在反向需要时重跑一次前向子图。
   3. 反向时，框架的 checkpoint 驱动：

      * 先重算被标记的子图，再基于重算激活做反向。
      * 其它未 checkpoint 的部分直接用缓存激活反向。

5. **流水并行下的时序关系**

   1. 采用 1F1B 调度，首个 stage 必须同时 hold 多个 micro-batch 的激活，以填满流水。
   2. 对首个 stage 来说，有效“层数”是 $L$，即使它实际只包含 $L/p$ 个物理层。
   3. selective recompute 允许优先对最占内存的部分重算，rest full-cache，从而在显存和重算开销间按实际卡容量做折中。

### 3.3 关键假设与适用范围

论文中的推导和结论基于若干重要假设，在实践中需要意识到它们的边界：

1. **只考虑主干 Transformer 块，忽略“小 buffer”**

   * 假设：LayerNorm 的均值/方差（$2sb$）和 bias 等 $O(h)$ 级别 buffer 可以忽略，仅关注 $O(sbh)$ 的激活。
   * 可能失效的场景：极短序列、小 hidden size 或大量额外辅助分支（例如多任务头）时，这些“小 buffer”占比上升，理论模型与实际显存可能有数个百分点偏差。

2. **统一使用 16-bit 激活（每元素 2 bytes），dropout mask 1 byte**

   * 假设：所有激活都以 FP16/BF16 存储，只有 logits 等少量张量使用 FP32。
   * 风险：如果你的栈中仍大量保留 FP32 激活（比如稳定性原因）、或者有自定义 kernel 使用更宽的中间格式，实际显存会高于模型预测。

3. **层结构高度同质，忽略 embedding / output 层贡献**

   * 假设：所有 Transformer 块的结构相同，embedding 和最后一层 FC / loss 的额外激活可以近似忽略。
   * 例外：在非常浅的网络（小 $L$）或 embedding/output 极大（超大 vocab）时，这一近似会变差，需要手工加上额外项。

4. **采用 1F1B 或 interleaved 1F1B 流水调度，首 stage 为瓶颈**

   * 假设：流水调度为 1F1B 或文中的 interleaved 变体，并通过增大 micro-batch 数量把流水“压满”，使首个 stage 的激活显存成为系统瓶颈。
   * 在非典型调度（大量 pipeline bubble、异构 stage、动态分配）或强 offload 场景，这个假设可能不成立，需要重新计算每个 stage 的峰值。

5. **attention 头数 $a$、序列长度 $s$ 足够大，使 $5as/h \gg 34$**

   * 假设：在 GPT-3、MT-NLG 这种规模下，attention 后半段激活（$QK^\top$、softmax 等）占了绝大多数显存。
   * 当 $s$ 很短、$a$ 很少、$h$ 很大时，$5as/h$ 不再显著大于 34，这时 selective recompute 的收益会下降。

### 3.4 数学公式与算法解读

这一小节挑出论文中几个关键公式，分别从“原文形式 → 含义 → 直观操作”三个层次来理解。

#### 3.4.1 单层激活内存（无模型并行）

**原文中的公式（式 (1)）：**

$$
M_{\text{act, layer}} = sbh \left( 34 + 5a \frac{s}{h} \right)
$$

* **在解决什么问题？**
  这是“一个 Transformer 层在前向中需要缓存多少激活”的近似公式，用来估算在不使用任何 TP/SP/PP 时，每层激活占用的显存。

* **符号含义：**

  * $s$：序列长度（sequence length）
  * $b$：micro-batch 大小
  * $h$：hidden 维度
  * $a$：attention 头数
  * $M_{\text{act, layer}}$：这一层的总激活内存（单位是 bytes，因为每元素已经乘上了 2 bytes）

* **如何得到 34 和 $5a s/h$？（直观版）**

  1. 把一个层拆成：两次 LayerNorm、一个 attention 块、一个 MLP 块。
  2. 粗略统计每部分需要缓存的张量数量和大小：

     * attention 块约贡献 $11sbh + 5as^2b$；
     * MLP 约贡献 $19sbh$；
     * 两个 LayerNorm 合计贡献 $4sbh$。
  3. 合起来就是 $(11 + 19 + 4) s b h = 34sbh$，再把 $5as^2b$ 写成 $sbh \cdot 5a s/h$，得到上式。

* **等价重写（仅为直观）：**

$$
M_{\text{act, layer}}
= s b h \cdot 34 ;+; 5 a s^2 b
$$

可以直接看成“**与序列长度线性相关的主干部分** + **与 $s^2$ 相关的 attention 复杂部分**”。

* **直观操作描述：**
  如果你给定 $(s, b, h, a)$，那么：

  1. 先算出“每层主干激活”的大小：$34sbh$；
  2. 再算出“attention 正方形矩阵相关”的大小：$5as^2b$；
  3. 二者相加就是这一层需要缓存的激活字节数。

#### 3.4.2 张量并行下的单层激活（TP）

**原文中的公式（式 (2)）：**

$$
M_{\text{act, layer}}^{\text{TP}}
= sbh \left(
10 + \frac{24}{t} + 5a \frac{s}{ht}
\right)
$$

* **含义：**
  在 $t$ 路张量并行（TP）下，只有 attention 和 MLP 内部“切得动”的那部分激活按 $1/t$ 分摊到了各卡，而 LayerNorm 和若干 Dropout 区域仍然在每卡完整保留，导致常数从 34 变成了 $10 + 24/t$，而 attention 中的 $5as^2b$ 项变成了 $5as^2b/t$。

* **直观理解：**

  * **“10”**：未切分、在每张卡上重复存在的 LayerNorm + Dropout 等激活。
  * **$24/t$**：TP 后真正被均分的部分（大 GEMM 相关）。
  * **$5a s/(ht)$**：attention 中 $s^2$ 级别的激活在 $t$ 卡上平均分摊。

* **直观操作描述：**

  1. 先像式 (1) 那样算一遍“总的主干激活”与 “attention 激活”；
  2. 再把能切的部分除以 $t$，不能切的部分保持不变；
  3. 把它们合起来，就得到了上式。

#### 3.4.3 张量 + 序列并行（TP+SP）

**原文中的公式（式 (4)）：**

$$
\begin{aligned}
M_{\text{act, layer}}^{\text{TP+SP}}
&= sbh \left(
\frac{10}{t} + \frac{24}{t} + 5a \frac{s}{ht}
\right) \
&= \frac{sbh}{t}\left(
34 + 5a \frac{s}{h}
\right)
\end{aligned}
$$

* **含义：**
  把之前 TP 下仍然重复的 10$sbh$ 这块，通过沿序列维的 SP 再切一刀，最终整层激活（包括 attention 的那一块）都被均匀地分摊到了 $t$ 个 TP rank 上——直观就是“**激活内存整体除以 $t$**”。

* **关键点：**

  * 依靠 $g$（all-gather）和 $\bar g$（reduce-scatter）这对“转换算子”把 LayerNorm / Dropout 区域从序列切分域切回张量切分域再切回去。
  * 通信带宽不变：因为原来的 ring all-reduce 本身就是 reduce-scatter + all-gather 的组合。

* **直观操作描述：**

  1. 先按式 (1) 算出无并行时 $M_{\text{act, layer}}$；
  2. 再简单除以 $t$，就得到 TP+SP 下每卡需要的激活内存。

#### 3.4.4 加上流水并行后的总激活内存

**原文中的公式（式 (5)）：**

$$
M_{\text{total}}^{\text{acts}} =
\frac{s b h L}{t}\left(
34 + 5 a \frac{s}{h}
\right)
$$

* **含义：**
  在 1F1B 流水调度下，首个 pipeline stage 尽管只负责 $L/p$ 个物理层，但因为要同时“在飞”$p$ 个 micro-batch，最终 peak 激活量等价于“**$L$ 层都压在这一卡上**”。因此总激活内存等于“单层激活 × $L$ 层 / $t$”。

* **直观操作：**

  1. 用式 (4) 算出单层 TP+SP 下的激活：$\frac{s b h}{t}(34 + 5 a s/h)$；
  2. 乘上需要同时驻留的“等效层数”——在经典 1F1B 中就是 $L$；
  3. 得到上式。

> 如果采用 interleaved pipeline，论文指出需要再乘上一个 $(1 + \frac{p-1}{pm})$ 的修正因子，这里不展开。

#### 3.4.5 全层重算 vs 选择性重算

1. **全层激活重算的内存（简单情形）**

   **原文中的结论：**

   $$M_{\text{full-recompute}} \approx 2 s b h L$$

   * 含义：如果你只 checkpoint 每层输入/输出（假设每层只一组），忽略其它激活，那么每层只需要存两份 $(s,b,h)$，总共就是 $2sbhL$。
   * 问题：显存是下来了，但每次反向要多跑一个完整前向，FLOPs 增加约 33%–40%，在大模型上非常肉疼。

2. **选择性激活重算（重点）**

   **原文中的公式（式 (6)）：**

   $$
   M_{\text{selective}} =
   \frac{34 s b h L}{t}
   $$

   * 含义：在 TP+SP 的基础上，只对 attention 中“大激活、低 FLOPs”的那几块做重算，把 $5 a s^2 b$ 那一坨激活完全从显存中移除，只剩下主干的 $34sbh$，然后再除以 $t$。

   * 直观：

     * 无并行 + 无重算：$L \times sbh(34 + 5as/h)$；
     * TP+SP + 无重算：再除以 $t$；
     * TP+SP + 选择性重算：再把 $5as/h$ 那块整个砍掉，对应就变成式 (6)。

   * 以 GPT-3 / MT-NLG 为例：
     对于 GPT-3 ($a=96,s=2048,h=12288$)，有 $5 a s/h \approx 80$；
     对于 MT-NLG ($a=128,s=2048,h=20480$)，有 $5 a s/h \approx 64$。
     相比主干常数 34，这说明**绝大多数激活其实来自那一小撮 attention 子算子**，砍掉它们能省掉 60%–70% 的激活，而相应重算 FLOPs 仅增加 1.6%–2.7%。

---

**与常见训练栈的对应关系**

从“我的大规模训练栈（如 Megatron / DeepSpeed / vLLM 等）”视角，可以这么理解这些模块对应到哪几层：

* **激活内存模型 → 配置搜索/自动调参层**

  * 用上面的公式快速预估在给定 $(s,b,h,a,L,t,p)$ 下的激活峰值，帮助选择 TP/SP/PP 组合和 micro-batch 大小。
* **TP+SP 组合并行 → 模型并行策略层**

  * 对应框架里“张量并行 + 序列并行”的维度配置，例如 `tensor_model_parallel_size`、`sequence_parallel_size`，以及相关 shard 规则。
* **$g/\bar g$ 算子 → 通信 backend + kernel 层**

  * 实际落地就是把原来的 `all_reduce` 替换成配对的 `all_gather` + `reduce_scatter`，常常与 GEMM kernel 融合在一起以减少中间缓冲拷贝。
* **选择性激活重算 → Checkpoint 策略/自动重算层**

  * 对应框架里的 `activation_checkpoint_method`、`checkpoint_attention` 之类的开关，以及在 Python 图里包一层 `checkpoint(function, *args)`。
* **Pipeline 分析 → 并行调度与作业编排层**

  * 决定每个 stage 放多少层、micro-batch 数量、是否使用 interleaved pipeline，并确保首 stage 的显存峰值满足卡容量。

---

## 四、建模方式与评估指标

### 4.1 问题是如何形式化的？

**核心优化目标**可以简单概括为：

> 在给定设备显存约束与并行配置（TP/SP/PP）的条件下，
> 最小化激活内存峰值与重算带来的额外 FLOPs 开销之和。

论文没有写成严格的优化问题，但通过公式基本隐式完成了建模：

1. **激活内存模型：**

   * 无并行时单层激活：
     $M_{\text{act, layer}} = sbh(34 + 5 a s/h)$。
   * TP+SP+PP 后首 stage 总激活：
     $M_{\text{total}} = \dfrac{s b h L}{t} (34 + 5 a s/h)$。
   * selective recompute 后：
     $M_{\text{selective}} = \dfrac{34 s b h L}{t}$。

2. **FLOPs 模型：**

   表 2 中给出了不同配置下每层 FLOPs，例如无并行时：

   $$
   \text{FLOPs}_{\text{no-parallel}}
   =================================

   72 s b h^2 \left(1 + \frac{s}{6h}\right)
   $$

   其它配置则在此基础上除以 $t$，或增加部分重算相关项（比如 selective recompute 下的 $1 + \frac{2s}{9h}$ 等）。

3. **简化与约束：**

   * 假设所有层同构，忽略 embedding/output 等小头；
   * 只考虑单 precision（16-bit）激活；
   * 只分析 FP 算力，暂不引入通信时延模型（通信通过“bytes communicated”单独报告）。

整个建模的思路是：**先用解析式锁定“理论上最优的内存分摊方式”，再在这个空间内讨论不同 checkpoint 策略的代价**。

### 4.2 核心评估指标

论文里的指标非常工程向，基本可以直接映射到你的监控面板上：

1. **激活内存（Activations Memory）**

   * 含义：单层或整个模型在前向/反向时为激活分配的显存峰值（通常以 GB 或占卡总显存的百分比表示）。
   * 对应关系：直接决定“能不能在一张卡上跑下这个配置”，也是是否需要再打开重算的第一判断依据。

2. **每层前向/反向时延（Forward / Backward Time per Layer）**

   * 含义：固定模型 & batch 设置下，单层 forward + backward 的 wall-clock 时间（ms），文中用单层 22B 模型来做 micro benchmark。
   * 对应关系：用来拆分“重算多耗了多少时间”、“SP/TP 对 LayerNorm/Dropout 加速了多少”。

3. **重算开销（Recompute Overhead）**

   * 含义：在“无重算 baseline”的前提下，重算之后 forward+backward 总时延的相对提升，比如 full recompute 的 +39% vs selective+SP 的 +4%。
   * 对应关系：帮助判断“多省的显存是否值这点时间”，对于整机训练吞吐尤为关键。

4. **端到端迭代时间（Iteration Time / Throughput）**

   * 含义：一次完整迭代（forward+backward+优化器更新）所需时间，论文中报告的是 22B/175B/530B/1T 模型的迭代时间与对应 throughput 提升（约 29%–32%）。
   * 对应关系：这是最贴近“训练总时长”的指标，也最容易映射到预算上。

5. **模型 FLOPs 利用率（MFU）与硬件 FLOPs 利用率（HFU）**

   * 含义：

     * MFU：模型理论 FLOPs / 峰值算力；
     * HFU：实际执行 FLOPs（包括重算）/ 峰值算力。
   * 对应关系：说明在应用 selective recompute 后，虽然实际 FLOPs 稍有增加，但总体算力利用率仍然可以维持甚至略升，比如 1T 模型的 MFU/HFU 在 56% 左右。

6. **通信量（Bytes Communicated）**

   * 含义：每层在不同配置下的总通信字节数，表 2 中以 bytes 形式给出。
   * 对应关系：对比“TP vs TP+SP vs full/partial recompute”是否引入额外 all-gather / reduce-scatter，帮助判断在不同网络拓扑（单机 NVLink、多机 IB）下是否会被通信瓶颈卡住。

---

## 五、主要实验发现

用几条结论把整篇实验的要点串一下：

* **TP+SP / selective recompute 单独使用时，各自都能将激活显存压到 TP 基线的约一半：**
  仅加 sequence parallelism，就能把激活降到原来的 ~50%；仅加 selective recompute，同样约半；两者叠加可达到约 **5×** 的压缩，使得 175B / 530B / 1T 配置在 80GB 卡上变得可行。

* **选择性重算极大降低了重算开销：**
  在 22B 单层实验中：

  * 全层重算：forward+backward 总时延增加 39%；
  * selective 重算：增加 7%；
  * selective + SP：仅增加 4%。

* **对大模型越友好，模型越大收益越高：**
  对 530B 和 1T 模型，full recompute 的重算 overhead 约 36%，而 selective+SP 的 overhead 仅 2%。

* **端到端吞吐提升约 30%：**
  在 4 组模型（22B/175B/530B/1T）上，与“full recompute + 无 SP”相比，本文方案的迭代时间缩短 29%–32%，对应 throughput 同比例提升。

* **FLOPs 利用率稳步提升：**
  随模型规模变大，MFU / HFU 从 40%+ 提升到 56% 左右，说明激活内存优化带来的“更大 batch、更好并行配置”对整体硬件利用率收益显著。

### 5.1 关键图表解读

1. **图 7：不同方案的激活内存占比（相对于 TP baseline）**

   * 现象：随着模型规模增大，sequence parallelism 与 selective recompute 单独使用时都能将激活压到约 50%；合用后能降到不到 20%，而 full recompute 在 10% 左右。
   * 支撑主张：说明在“只付出 2%–7% 重算开销”的前提下，TP+SP+selective 已经非常接近 full recompute 的内存效率，但少了大量无谓的算力开销，是实践中更平衡的方案。

2. **图 8：每层 forward / backward / recompute 时间拆分**

   * 现象：

     * baseline 无重算时，backward 时间远大于 forward；
     * full recompute 给 backward 顶上去一大块；
     * selective+SP 的“重算条”非常细，对整体影响极小。
   * 支撑主张：证明 selective 重算确实只把重算集中在 FLOPs 稍多的那一小块，且通过重叠通信/计算把 overhead 压得很低。

3. **表 5：端到端迭代时间与 FLOPs 利用率**

   * 现象：所有规模模型的 iteration time 都从 “full recompute” 配置中减掉了 ~30%；同时 MFU/HFU 随规模增大而升高，1T 模型可到 56%+。
   * 支撑主张：说明本文不仅仅是“把显存凑够就完事”，而是在实际训练吞吐和硬件利用率上都证明了工程价值。

**结果解读与边界**

总体来看，实验非常有说服力：公式推导和实测数据高度匹配，从单层 micro benchmark 到上百层、上万卡的端到端实验，都展示了 TP+SP+selective 的稳定优势。

但也存在一些未完全覆盖的维度，例如：

* 并未系统评估 **更复杂的重复结构**（MoE、带多路分支的 encoder-decoder）中 selective recompute 的收益与开销；
* 对 **梯度 checkpoint 搜索算法**（如 CVPR 2021 的 Optimal Checkpoint Search）只在相关工作中提及，未做直接对比；
* 实验主要集中在单一硬件平台与网络拓扑，对低带宽多机环境下“all-gather / reduce-scatter 数量增加是否成为瓶颈”缺乏系统评价。

---

## 六、优点与局限

**亮点（Strengths）**

* **问题刻画非常精准**：从“激活内存”而不是“总显存”切入，将 attention / MLP / LN / Dropout 的激活占比拆得非常细，有利于工程上针对性优化。
* **解析模型简单但威力大**：几个短公式就解释了 TP / SP / PP 的内存行为，并自然给出“激活均匀分摊到 TP 组”的最优形式，为后续工作提供了统一的度量尺。
* **TP+SP 设计优雅**：通过 $g/\bar g$ 将 all-reduce 拆成 all-gather + reduce-scatter，不改变通信带宽，仅改变通信算子的形态，就解决了非 TP 区域的激活重复问题。
* **选择性重算非常工程友好**：只需要在 attention 内部加几处 checkpoint 标记，既减少大量激活，又避免像“全层重算”那样动辄 +30% FLOPs。
* **实验覆盖到 trillion-scale**：在 22B–1T 四个量级模型上完整评估，包含单层时延、整模型迭代时间、MFU/HFU，非常贴近实际大模型训练场景。
* **与现有并行栈高度兼容**：TP+SP+PP 的组合可以自然嵌入到主流 3D 并行框架中，不与 ZeRO/FSDP 等参数/优化器切分技术冲突。([arXiv][1])

**局限（Limitations）**

* **结构假设较强**：只针对标准单 stack Transformer，且默认层结构高度一致，对 MoE、encoder-decoder、多任务头等复杂拓扑的适配并未深入讨论。
* **完全手工的 checkpoint 策略**：当前 selective recompute 方案基于人工分析，并未利用图搜索/自动调度算法去进一步逼近理论最优。
* **缺少对激活碎片化问题的定量分析**：虽然结论部分提到碎片和首 stage 内存不均是未来工作方向，但正文未给出系统测量或模型。
* **通信性能假设偏理想**：将 all-reduce = reduce-scatter + all-gather 视作“通信带宽不变”，在跨机、非全连接拓扑下可能不完全成立。
* **与其它内存优化技术的组合分析有限**：例如与 ZeRO/FSDP、offload、FlashAttention 等组合后的整体收益，目前仍需读者自行探索。([arXiv][1])

---

## 七、业内相关工作对比

下面选 3 类代表性工作，与本文做一个工程视角下的对比：

| 工作                                                                           | 问题定义                                                        | 方法路线                                                                | 贡献与实用价值（主观）                                               |
| ---------------------------------------------------------------------------- | ----------------------------------------------------------- | ------------------------------------------------------------------- | --------------------------------------------------------- |
| **本文：减少激活重计算**                                                               | 大规模 Transformer 训练中，**激活显存**成为主要瓶颈，full recompute 带来巨大算力开销。 | 精确建模激活内存，结合 TP + SP 均分激活，并在层内对子算子做 selective recompute。             | 在不改模型结构的前提下，显存压缩 5×，吞吐提升 ~30%，对于已有 3D 并行栈几乎是“必选项”。        |
| **ZeRO / FSDP 系列**([arXiv][1])                                               | 聚焦 **参数+优化器状态** 的内存冗余，使模型规模随设备数线性扩展。                        | 通过切分 optimizer state、gradient、parameter，将数据并行中的冗余全部打散，配合 offload。   | 大幅减小“模型状态”占用，适合在 DP 维度扩展，和本文在维度上高度互补。                     |
| **GSPMD / 通用 SPMD 并行**([arXiv][2])                                           | 提供一种统一的图级 SPMD 并行抽象，支持 TP/PP/DP/混合。                         | 将并行视作对 tensor shape 的“sharding spec”，由编译器自动完成调度与通信插入。               | 在编译层面对各种并行形式进行统一描述，适合作为 TP+SP+selective 这类优化的“载体”。        |
| **Sequence Parallelism from System Perspective**（SP 系列工作）([ResearchGate][3]) | 面向超长序列训练，关注 **沿序列维切分 activations/参数** 的系统设计。                | 提出多种 SP 变体（ring attention 等），通过在 attention 内加入特殊通信模式减少 $s^2$ 存储和计算。 | 对长上下文模型极为重要，与本文的 SP 思路类似但更关注“长序列下 attention 的计算 pattern”。 |

整体而言，本文可以看作是 **“TP-centric 3D 并行栈中针对激活的一块补完”**：

* 在参数/优化器维度，它自然可以与 ZeRO/FSDP 协同；
* 在编译/图调度维度，可以被 GSPMD 等 SPMD 框架实现为一套 sharding 规则与通信重写；
* 在长序列场景下，可与更激进的 SP / context parallel / ring attention 等方案互补。

### 7.1 个人观点

从“如何写一篇系统论文”的角度看，这篇文章的论证路线非常清晰：

* 先用解析模型解释清楚 **“为什么需要 TP+SP + selective”，以及它在公式上的最优性**；
* 再用多组实验验证“模型和现实基本一致”，并贯穿不同模型规模，避免只在单一规模做 cherry-pick。

如果要挑刺，我觉得可以加强的部分有：

* **baseline 更丰富**：目前重算部分主要对比的是“full recompute vs selective”，如果能再加上“一些自动 checkpoint 搜索算法”（例如 Feng & Huang 2021）或现有框架中的默认策略，对工程选型会更有参考意义。
* **与其它内存优化的组合实验**：例如将 TP+SP+selective 与 ZeRO/FSDP/FlashAttention/参数 offload 一起放入同一张对比表中，说明不同维度上的可叠加性。
* **对碎片和调度的更系统分析**：如能在附录中补充 pipeline 首 stage 的内存碎片分布、不同 micro-batch 数量对碎片的影响，会更利于工程落地时做二次权衡。

---

## 八、在实际训练栈中如何落地？

假设你已经有一套“3D 并行 + 激活 checkpoint” 的训练栈（比如某种 Megatron/DeepSpeed 风格），要引入本文方法，大致可以从以下几个层面动手：

1. **并行调度（TP / SP / PP 组合）**

   * 在现有 TP 配置上，新增 **sequence parallel 维度**，例如增加 `sequence_parallel_size`，并为 LN/Dropout/embedding/output 等非 TP 区域指定“按序列切分”的 layout。
   * 在 pipeline 切分时，显式考虑“首 stage 需要 hold $L$ 层激活”的事实，用上面的公式评估不同 `pipeline_model_parallel_size` 下的 peak 显存。
   * 对多机场景，确认 SP 的通信组（通常和 TP 组一致），避免跨节点频繁做 all-gather / reduce-scatter。

2. **kernel / 算子实现**

   * 为 LN/Dropout/write-back 等算子增加 **SP awareness**：输入张量在序列维上是 shard 的，算子应能在局部 shard 上工作。
   * 将原本在 TP 内部使用的 `all_reduce` 改写成成对的 `all_gather` + `reduce_scatter`，并尽可能与 GEMM kernel 融合，减少中间 buffer。
   * 在 attention 中，对 $QK^\top$、softmax、dropout、attention over V 那一段子图增加“便于重算”的边界，比如使用框架内的 `checkpoint` 包一层。

3. **激活 checkpoint / 重算策略**

   * 提供一个**细粒度的重算配置接口**，允许用户单独控制：

     * 是否对 attention 内部做 selective checkpoint；
     * 是否对 MLP 或整层做额外 checkpoint（在更紧张显存下）。
   * 将论文里的“GPU 级 FLOPs overhead 估算公式”固化为工具函数，让用户在配置文件中看到“预估重算 overhead 与激活节省比例”，以帮助选边界。

4. **通信与集体操作 backend**

   * 在通信层额外支持“基于形状与 layout 的 all-reduce ↔ AG+RS 重写”，必要时对 `all_gather` 和 `reduce_scatter` 做专门调优（pipeline overlap、组内拓扑 awareness 等）。
   * 为 SP/TP 的通信 group 提供统一管理，避免出现“一个 rank 同时隶属太多 group 导致 NCCL resource 紧张”的问题。

5. **DataLoader / 预处理与打包策略**

   * 虽然本文不直接改变 DataLoader，但在实践中通常会利用“节省下来的激活显存”去增加 micro-batch 或 global batch，此时需要检查：

     * 数据打包是否支持更大 batch（尤其是多任务混合数据集）；
     * 长序列训练时，是否与 SP / context parallel 等策略冲突。

6. **配置搜索 / 自动调参**

   * 将激活内存模型与 FLOPs 模型做成一个小工具（甚至可以写成 Python 脚本），在给定硬件规格和模型配置的情况下，自动搜索可行的 `(TP, SP, PP, micro-batch)` 组合。
   * 对于自动化的 launcher，可以在提交前直接给出“预估 peak 显存、重算开销、MFU 上限”等信息。

7. **监控与调试**

   * 在框架中增加 per-layer / per-stage 的 **激活内存追踪**（通过 forward/backward hooks），验证是否符合论文公式的预估。
   * 监控“重算区”的时间占比，确认 selective 重算的 overhead 是否接近论文中的 2%–7%，若远高于此需要检查通信 overlap 是否生效。

总的来说，引入本文方法的工程工作量主要集中在 **算子 layout 改写 + 通信模式重写 + checkpoint 策略细化**，对上层模型代码侵入较小。

---

## 九、值得进一步探索的研究方向

1. **自动化激活重算策略搜索**

   * 问题：目前 selective recompute 仍基于手工划分；对于更复杂的网络结构，人肉选择 checkpoint 边界既费时又可能 sub-optimal。
   * 价值：结合已有的“最优 checkpoint 搜索”算法（如 CVPR 2021 Feng & Huang）与本文的激活内存模型，有望自动给出在不同显存预算下的最佳重算策略。

2. **与 ZeRO / FSDP / offload 的统一建模**

   * 问题：当前实践往往同时启用参数/梯度/优化器的切分与 offload，以及激活层面的 TP+SP+selective，缺乏统一的成本模型。
   * 价值：构建一个统一的“显存+FLOPs+通信三元模型”，自动在“加大 DP、加大 TP/SP、加大小重算”之间平衡，指导 trillion-scale 训练栈设计。

3. **面向长上下文的序列并行与重算协同**

   * 问题：随着 128K+ 上下文模型普及，各类 sequence/context parallel（ring attention、Ulysses 等）将 attention 变得更加复杂。([ResearchGate][3])
   * 价值：在这些 SP 变体中引入 selective recompute，分析在 $s\gg h$ 情况下重算开销的精确行为，可能会给长上下文模型带来新的可行配置。

4. **针对 MoE 与稀疏结构的激活内存优化**

   * 问题：MoE 将计算稀疏化，但激活内存仍可能较高，且路由/门控带来新的通信与存储模式。
   * 价值：扩展本文的激活模型到“稀疏激活”场景，定义 per-expert 的激活与重算策略，有助于在保持稀疏计算优势的同时进一步压缩显存。

5. **pipeline 首 stage 内存碎片与动态调度**

   * 问题：论文提到 pipeline 首 stage 的显存不均与碎片化是未来方向之一，但尚无系统方案。
   * 价值：结合 allocator 行为（如 buddy / caching allocator）与 dynamic micro-batching，探索在不改模型结构的前提下，通过调度与分配策略进一步降低首 stage 峰值。

---

## 十、知识图谱思维链

从“大模型系统”的知识图谱来看，这篇论文涉及的连接点大致如下：

* **并行与调度**

  * 提供了一个把 TP+SP+PP 一起放进激活内存公式的框架，让“如何选 TP/SP/PP 组合”从拍脑袋变成可计算的问题。
  * 把 1F1B / interleaved pipeline 的显存峰值特性用简洁公式刻画出来，为之后的流水调度论文（如多种 1F1B 变体）提供了对比基线。([ACL Anthology][4])

* **内存管理与显存优化**

  * 把激活内存拆解成“主干（$34sbh$）+ attention 方阵（$5as^2b$）”，让人一眼看出优化空间在哪里。
  * selective recompute 展示了“通过精细定位 FLOPs 便宜区”来换显存，是一类值得在其他结构上重复使用的模式。

* **通信与集体操作**

  * 显式利用“all-reduce = RS + AG”这一事实，通过 $g/\bar g$ 改写通信图，实现激活切分而不增加总通信量。
  * 对比了不同方案下 per-layer 通信 bytes，为之后的通信优化工作提供了一个可参考的 baseline。

* **kernel 与算子优化**

  * 强调在 LN / Dropout / embedding / output 等算子中也做 SP，让这些“看似简单”的算子真正享受到并行带来的内存与速度收益。
  * 鼓励把通信算子和 GEMM 融合，从而减少中间 buffer 与 kernel launch overhead。

* **模型结构与架构设计**

  * 虽然模型结构本身未修改，但激活内存分析可以直接用来评估“加宽/加深/加头数/加序列长度”对显存的影响，为设计新架构提供量化依据。
  * 对 GPT-3/MT-NLG 的具体参数做了代入，不仅告诉你“公式长啥样”，还告诉你“在真实配置下数值是多大”。

* **数据、预处理与打包策略**

  * 从侧面说明了“节省激活显存之后可以做什么”：可以换成更大 micro-batch、更长序列或更多 global batch，对 DataLoader 与数据打包策略提出了新的需求。

### 10.1 个人收获与反思

对我个人来说，这篇论文最大的启发在于——**很多看似“经验主义”的并行/重算技巧，其实可以被一个非常简洁的解析模型统一描述**。一旦把激活内存拆成 $34sbh$ 和 $5as^2b$ 两块，很多选择就变得显而易见：TP+SP 应该怎么切、attention 中哪一部分值得 checkpoint、pipeline stage 怎么分层等，都可以从公式里直接读出来。

另一个收获是对 **“局部重算”这一模式的再认识**：以前提到 gradient checkpoint，多数人只想到“按层 checkpoint”；本文展示了“按层内子算子 checkpoint”可以更精细地调节显存/算力的 trade-off，而且实现成本并没有想象中那么高——只要你愿意在算子图上多画几条边界。

从实践角度，我认为值得立刻尝试迁移到自己训练栈里的点主要有两个：

* 第一是 **在现有 TP 栈上补齐 SP**，至少要让 LN/Dropout/embedding/output 这些激活也能按序列 shard；
* 第二是在 attention 内部实现类似的 selective recompute，把 $QK^\top$、softmax、attention over V 那块抽成一个 checkpoint 子图，并配合通信/计算 overlap 做些微调。

> 总体评价：这篇工作在不改变模型结构、不过度侵入训练栈的前提下，用一套简洁的理论和一组扎实的大规模实验，给出了一个几乎“默认应当启用”的激活内存优化方案。对于已经运行 3D 并行大模型训练的团队，它更偏工程实践；而对于正在搭建设备/并行栈的人，则提供了一个非常清晰的“并行+重算联合设计”参考范式。

[1]: https://arxiv.org/abs/1910.02054
[2]: https://arxiv.org/abs/2105.04663
[3]: https://www.researchgate.net/publication/372917292_Sequence_Parallelism_Long_Sequence_Training_from_System_Perspective
[4]: https://aclanthology.org/2025.naacl-long.454.pdf
