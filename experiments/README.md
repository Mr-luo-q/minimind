# MiniMind 后训练实验记录

本目录是后训练实验的总入口：**`_tools/` 只有代码与配置，所有产物都在 `results/`**。

> 📄 **文档已单独整理成独立仓库**：[`Mr-luo-q/minimind-posttrain-docs`](https://github.com/Mr-luo-q/minimind-posttrain-docs)
> —— 只放文档与图表，便于单独阅读/分享；代码与原始数据仍在本仓库 `experiments/`。

## 目录结构

```
experiments/
├─ README.md                  ← 本文件（索引）
├─ 后训练实验报告.md            ← 主报告（工具生成，含图表与逐字回答）
├─ EXPERIMENT_PLAN_V2.md      ← 第二阶段方案（竞赛数学 + 函数调用，汇报用）
├─ img/                       ← 报告用的图表（课程实验部分）
├─ _tools/                    ← 只有代码与配置，不放产物
│  ├─ make_charts.py          ← 生成 img/ 下的图表
│  ├─ assemble_report.py      ← 生成 后训练实验报告.md
│  ├─ qwen_sft/               ← LLaMA-Factory SFT（configs / data / 转换与评测脚本）
│  └─ qwen_grpo/              ← TRL GRPO 训练脚本 + bench/ 评测工具
└─ results/                   ← 所有实验产物
   ├─ qwen_4b_TRAINING_LOG.md ← **Qwen3-4B 训练日志（带图，自动生成）**
   ├─ mini_sft/               ← MiniMind 64M：exp1(全参1轮) / exp2(全参2轮) / exp3(LoRA)
   ├─ qwen_sft/               ← Qwen SFT：loss 曲线、probe 回答、训练日志、summary
   ├─ qwen_grpo/              ← Qwen GRPO：训练日志、显存采样、probe、边界扫描结果
   └─ bench/                  ← 三个模型（base/SFT/GRPO）的 IFEval-lite + GSM8K 对比
      ├─ index.html           ← 结果总表（浏览器直接打开）
      ├─ RESULTS.md / ANALYSIS.md / table.csv
      ├─ WHY_BASE_WINS.md     ← 归因分析：为什么 base 反而最好
      ├─ raw/                 ← 最终数据集（v3 判分口径）+ manifest.json
      ├─ batches/             ← 各次跑批的原始拉取
      └─ archive/             ← 旧判分口径的跑批（勿与最终结果混用）
```

> 每个结果目录里的 `manifest.json` 记录该批结果用的判分口径。
> `bench/` 的判分器修过两个缺陷（中文长度判分、结尾判分过严），**旧口径与修正版分数不可直接比较**，
> 详见 `bench/README.md` 与 `_tools/qwen_grpo/bench/README.md`。

## MiniMind 尺度实验（results/mini_sft/）

| exp | 方法 | 数据 | 轮数 | lr | loss末值 | loss最小 | 8问对比 |
|---|---|---|---|---|---|---|---|
| exp1 | 全参SFT | sft_t2t_mini | 1 | 1e-5 | 1.5287 | 1.2895 | probe_before/after |
| exp2 | 全参SFT | sft_t2t_mini | 2 | 1e-5 | 1.6683 | 1.0141 | probe_after_sft_ep2 |
| exp3 | LoRA | sft_t2t_mini | 1 | 1e-4 | 1.6561 | 1.5412 | probe_lora |

**观察要点**

- exp1 vs exp2：多训 1 轮（epochs 2）loss 是否继续下降、末值差异
- exp1 vs exp3：全参(63.9M 全训) vs LoRA(仅 adapter) 同数据 1 epoch 的 loss 形态与参数量差异
- 各目录 `probe_*.txt` 为同一 8 问的模型回答（pretrain=SFT前底座；after_sft=全参SFT后；lora=LoRA版）

## Qwen3-4B 尺度的三模型对比（results/bench/）

| 模型 | IFEval instruction-level | IFEval prompt-level | GSM8K |
|---|---|---|---|
| base（Qwen3-4B 底座） | **98.1%** | **98.0%** | 94.0% |
| SFT（e1 LoRA, 1 epoch） | 88.7% | 88.0% | 84.0% |
| GRPO（通用 RM, 250 步） | 96.2% | 96.0% | **95.0%** |

结论：base 指令遵循最强；SFT 在 IFEval(−9.4pt) 与 GSM8K(−10.0pt) 上都明显退化；
GRPO 基本追平 base。**IFEval-lite 为自实现轻量版，分数仅用于模型间横向对比，不代表官方 IFEval 榜单成绩。**

**为什么 base 反而最好？** 逐题归因见 [`results/bench/WHY_BASE_WINS.md`](results/bench/WHY_BASE_WINS.md)：
SFT 是"训错了目标"（用预训练能力换 MiniMind 文风，IFEval 单向退化 0 翻盘 / 5 翻车，GSM8K 损伤集中在"大数"）；
GRPO 是"奖励不针对目标"（KL≈0.02 几乎没动底座，与 base 的 GSM8K 预测一致率 97/100）。
要真正提升 GSM8K，得把奖励从"整段回答讨不讨喜"换成可验证规则（RLVR）。

## 下一步：第二阶段方案

因 4B 基线已饱和（GSM8K 94.0% / IFEval-lite 98.1%，训练空间小），第二阶段改用 **Qwen3-1.7B**，
转向两个「单轮 + 低基线 + 规则可验证」的方向：**竞赛数学**（AIME24/25 基线 11.7%，文献实测同模型可 +11~16pt）
与 **函数调用**（BFCL 基线 55.49%，短板在 Parallel 44.44%）。

完整方案（含基线数据、评测集、奖励设计、风险与待确认项）见 **[`EXPERIMENT_PLAN_V2.md`](EXPERIMENT_PLAN_V2.md)**。

## 权重说明

`*.pth` 不入库（>100MB），保留在实例 `/root/autodl-tmp/minimind/out/`：

| 文件 | 说明 |
|---|---|
| sft_mini_ep1_768.pth | exp1 全参 SFT 1 epoch |
| sft_mini_ep2_768.pth | exp2 全参 SFT 2 epochs |
| lora_sft_ep1_768.pth (797KB) | exp3 LoRA adapter（需配 pretrain_768.pth 使用）|
