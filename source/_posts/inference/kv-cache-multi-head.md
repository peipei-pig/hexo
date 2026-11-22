---
title: "从多头注意力到 KV Cache：一次 Token 生成的完整计算过程"
date: 2026-11-22 12:00:00
categories:
  - 推理
tags:
  - KV Cache
  - 大模型推理
  - 注意力机制
description: 以 Qwen2.5-0.5B 为例，追踪单步 Decode 中的 Q、K、V 投影、逐层 KV 读写和注意力计算，并估算缓存容量。
link: /papers/kv-cache-multi-head/
html_source: papers/kv-cache-multi-head/index.html
---

用 `The cat sat` 对照全前缀重算与增量解码，追踪 Qwen2.5-0.5B 第 1 层的张量形状和实际数值，再推算多层 KV Cache 的容量与读取量。

<!-- more -->

[阅读完整图文解析](/papers/kv-cache-multi-head/)
