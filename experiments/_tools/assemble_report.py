# -*- coding: utf-8 -*-
"""Assemble the lab report with all probe answers embedded verbatim."""
import html
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # experiments/
PROBES = [
    ("results/mini_sft/exp1_ep1/probe_before_pretrain.txt", "SFT 前底座（pretrain_768.pth，无指令模板，直接续写）"),
    ("results/mini_sft/exp1_ep1/probe_after_sft.txt", "exp1：全参 SFT 1 epoch 后"),
    ("results/mini_sft/exp2_ep2/probe_after_sft_ep2.txt", "exp2：全参 SFT 2 epochs 后"),
    ("results/mini_sft/exp3_lora_ep1/probe_lora.txt", "exp3：LoRA 1 epoch 后（adapter 叠加 pretrain）"),
]

HEADER = """# MiniMind 后训练实验报告

> **实验日期**：2026-09-02
> **硬件**：AutoDL RTX 5090 (32GB) · torch 2.8.0+cu128 · Python 3.12 · transformers 4.57.6
> **项目**：jingyaogong/minimind（实验分支位于本 `experiments/` 目录，已同步 GitHub `Mr-luo-q/minimind`）

---

## 1. 实验对象：什么模型

本次所有实验基于 **MiniMind minimind-3（Dense）**，参数量 **63.91M（约 6400 万，≈0.064B）**：

| 配置项 | 值 |
|---|---|
| 隐藏维度 d_model | 768 |
| Transformer 层数 | 8 |
| 词表大小 | 6400 |
| Q heads / KV heads | 8 / 4（GQA） |
| max_position_embeddings | 32768（rope_theta=1e6） |
| 前馈层 | SwiGLU |
| **总参数量** | **63.91M**（fp16 权重约 128MB） |

**后训练的含义**：从官方预训练底座 `pretrain_768.pth` 出发，用指令数据继续训练，让模型学会"按指令格式回答问题"（SFT 阶段 label 只计算 `assistant` 回答部分，`user` 提问与模板 token 以 `-100` 屏蔽）。

---

## 2. 用了什么方法：三种后训练方案

| 实验 | 方法 | 脚本 | 数据 | 轮数 | batch | seq | lr | 可训练参数 |
|---|---|---|---|---|---|---|---|---|
| exp1 | **全参 SFT**（Full Fine-tuning） | `train_full_sft.py` | sft_t2t_mini（905,718 条对话） | 1 | 16 | 768 | 1e-5 | **63.91M（100%）** |
| exp2 | **全参 SFT** | `train_full_sft.py` | 同上 | **2** | 16 | 768 | 1e-5 | 63.91M（100%） |
| exp3 | **LoRA**（低秩适配） | `train_lora.py` | 同上 | 1 | 32 | 768 | 1e-4 | **~0.797M（约 1.2% adapter）** |

- exp1 / exp2 对比 → 考察**训练轮数（数据利用次数）**对 SFT 效果的影响；
- exp1 / exp3 对比 → 考察**全参更新 vs 参数高效微调（LoRA）**在同等数据、同起点下的差异；
- 三者共享同一起点 `pretrain_768.pth`、同一数据、同一评测集（8 个固定问题），除上述变量外其余配置一致。

---

## 3. 跑了几次实验

**3 次训练实验 + 4 组模型评测**（同一 8 问对 4 个模型依次提问）：

| # | 模型 | 来源 | 评测输出 |
|---|---|---|---|
| 1 | pretrain（SFT 前底座） | 官方权重 | `results/mini_sft/exp1_ep1/probe_before_pretrain.txt` |
| 2 | exp1：全参 SFT 1 epoch | 本次训练产物 | `results/mini_sft/exp1_ep1/probe_after_sft.txt` |
| 3 | exp2：全参 SFT 2 epochs | 本次训练产物 | `results/mini_sft/exp2_ep2/probe_after_sft_ep2.txt` |
| 4 | exp3：LoRA 1 epoch | 本次训练产物（adapter 叠加在 pretrain 上） | `results/mini_sft/exp3_lora_ep1/probe_lora.txt` |

每轮训练均记录了完整日志（`train.log`，全参 SFT 每 100 步一条、LoRA 每 10 步一条）与 loss 序列（`loss_curve.csv`）。

---

## 4. 实验结果可视化

### 4.1 三种方法的 loss 曲线对比

横轴按各实验自身总步数归一化为训练进度百分比；曲线做了轻量平滑便于观察趋势（蓝=全参 1ep，红=全参 2ep，绿=LoRA 1ep，圆点标记各自 loss 最低点）：

![loss 曲线对比](./img/loss_curves.png)

**读图要点**：
- exp1（蓝）与 exp2（红）前 50% 阶段几乎重合——同一数据第一轮学习行为一致；exp2 在第二个 epoch（50%~100%）继续下降，loss 最低点 **1.014** 明显低于 exp1 的 1.29；
- exp3（绿）从 pretrain 出发、以更高 lr（1e-4）训练，初期波动更大（见 4.4 放大图），最终收敛到接近 exp1 的水平；
- 三个实验末尾的 loss 末值都有回升/波动——这是 lr 余弦衰减到底 + 采样间隔的叠加现象，不代表模型变差（评估应以 loss 最低点附近的 checkpoint 为准）。

### 4.2 loss 首值 / 最小值 / 末值对比

![loss 统计](./img/loss_stats.png)

| 实验 | loss 首值* | loss 最小值 | loss 末值 |
|---|---|---|---|
| exp1 全参 1ep | 1.987 | **1.29** | 1.529 |
| exp2 全参 2ep | 1.987 | **1.014** | 1.668 |
| exp3 LoRA 1ep | 2.712 | 1.541 | 1.656 |

> *首值 = 日志首个打点，注意打点间隔不同：exp1/exp2 首个打点在 step 100，exp3 在 step 10（详见第 7 节"为什么 LoRA 初始 loss 更大"）。

### 4.3 实际更新的参数量：全参 vs LoRA

![可训练参数量](./img/trainable_params.png)

LoRA 只训练 **0.797M** 参数（adapter 权重文件仅 797KB），约为全参微调的 **1/80**，却能在 1 epoch 内把 loss 压到 1.54——这就是参数高效微调"小参数、大效果"的直观体现（代价是容量上限低于全参，见 4.2 中 exp1 最小值更低）。

### 4.4 初始阶段放大：为什么 LoRA 的首个记录 loss 更高

![初始阶段 loss](./img/early_loss.png)

exp3（LoRA，lr=1e-4）在 step 20 冲到最高 **2.86** 后迅速回落；exp1（全参，lr=1e-5）全程平稳在 ~2.0。原因分析见第 7 节。

---

## 5. 定性对比小结

对同一 8 问，SFT 前后的回答差异主要体现在**格式纪律**（编号列表、加粗标题、礼貌收尾），而不是知识量本身（64M 模型 + 1 epoch 的知识提升有限，回答仍常有事实错误与车轱辘话——这正是后续 RL 阶段想改善的方向）。全部回答原文见第 6 节。

**示例 Q1「你有什么特长？」（SFT 前 vs SFT 后，节选）：**

SFT 前（pretrain 底座，直接续写）：
> 作为AI助手，我没有个人特长或特长。我被设计为根据提供的信息和指令执行任务……然而，我没有个人特长或能力，因为我没有自我意识或情感。

SFT 后（exp1，全参 SFT 1 epoch）：
> 作为AI助手，我的主要职责是提供准确、高效的信息和帮助。我的特长包括但不限于：
> 1. **知识丰富**：能够从不同领域的知识和信息中找到解答……
> ……
> 我的特长是快速响应、知识、多语言……如果您有任何问题或需要帮助，随时告诉我，我会尽力提供帮助。

---

## 6. 全部 probe 回答全文（8 问 × 4 模型，逐字原文）

> 说明：为保持逐字原文（回答中本身包含 markdown 加粗、列表与 ```python 代码块），以下回答以 HTML `<pre>` 原样块呈现。评测设定：固定每问 seed（1000+i）、top_p=0.95、temperature=0.85、max_new_tokens=512（脚本 `_tools/eval_probe.py`，部署于实例 `/root/autodl-tmp/eval_probe.py`）。

{{PROBES}}

---

## 7. 讨论：为什么 LoRA 的初始 loss 会更大

### 现象

| 实验 | 首个打点 | 值 | 打点间隔 |
|---|---|---|---|
| exp1 全参 SFT（lr=1e-5） | step 100 | 1.987 | 每 100 步 |
| exp3 LoRA（lr=1e-4） | step 10 | 2.712（且 step 20 冲到 **2.863**，为全程最高） | 每 10 步 |

两者**起点是同一份 `pretrain_768.pth` 权重**，理论上 step-0 的 loss 应相同（~2.0 量级）。差异是训练开始后几十步内产生的，主要有三个原因：

### 原因 1：学习率差 10 倍 → 初始"过冲"（主因）

- LoRA 脚本默认 lr=**1e-4**，全参 SFT 用 lr=**1e-5**（两者都是各自脚本的默认值）。
- SFT 数据分布（带 `<|im_start|>` chat 模板的指令格式）对只见过预训练语料的模型是**轻微分布外（OOD）**，loss 曲面在起点附近较陡。
- lr=1e-4 意味着前几步"迈"得很大，直接翻过 loss 谷底冲到高处：exp3 从 step 10 的 2.71 继续涨到 step 20 的 **2.86**（全程最大值），随后才回落（step 50 ≈ 2.20，step 150 ≈ 2.04）——教科书式的**学习率过大导致的早期过冲**。
- 全参 SFT 的 lr=1e-5 步长小，loss 始终平稳在 ~2.0 附近再缓慢下行，不会出现这个尖峰。
- 佐证：待 lr 余弦衰减到 ~1e-4 以下后（step ≥ ~2000），exp3 的 loss 才进入与 exp1 相近的下降通道——两条曲线在 4.1 图中后段走势一致。

### 原因 2：可调参数空间受限 → 前期"纠正 OOD"更慢

- 全参 SFT 每步可同时更新全部 **63.91M** 参数（embedding、注意力、MLP 全动），拟合指令分布的自由度大；
- LoRA 每步只能沿低秩子空间更新 **0.797M** adapter 参数（冻结的 98.8% 只能被动传导梯度），等效表达能力受限，同样步数内把分布外输入"拉回"的速度更慢，因此前期 loss 停留在高位的时间更长。
- 注意：这影响的是**前期下降速度**，并不直接改变"起点值"。

### 原因 3：打点时机不同（统计口径，易被忽略）

- exp1/exp2 的 `log_interval=100`（首个记录点是 step 100），exp3 的 `log_interval=10`（首个记录点是 step 10）。
- 因此表里的"首值"不是同一时刻的 loss：exp1 的 1.987 是训练了 100 步之后（初始波动已平复），exp3 的 2.712 是刚起步 10 步（正撞上过冲期）。
- 若给 exp1 也按 step 10 打点，它的值会高于 1.99，但不会像 exp3 那样冲到 2.86——因为它的 lr 小、过冲弱。
- 收敛后两者接近（exp3 末值 1.656 vs exp1 末值 1.529），也印证起点差异不是本质问题。

### 结论

> **LoRA 初始 loss 更高 ≈ "10 倍学习率的早期过冲 + 低秩容量限制导致前期下降慢 + 打点时机更早"三者叠加**，与 LoRA 方法本身的收敛上限无关。若把 LoRA 的 lr 降到 1e-5 并同样在 step 100 才打点，其"首值"会与全参 SFT 相当。实践中 LoRA 用更高 lr（1e-4 量级）是常见做法——它参数少、不易过拟合，只是要容忍初始阶段的尖峰（必要时可加 warmup 缓解）。

---

## 8. 结论要点

1. **训练轮数有效**：exp2（2 epochs）loss 最低点比 exp1（1 epoch）低约 0.28，说明同数据多过一轮仍能学到新东西；代价是训练时间翻倍（~2h vs ~1h）。
2. **全参 > LoRA（同预算下）**：1 epoch 内全参 loss 更低（1.29 vs 1.54）；但 LoRA 用 1/80 的参数量达到了接近的效果，且训练更快、显存更低——**参数受限或数据量小时 LoRA 性价比极高**。
3. **loss 之外必须看回答**：loss 下降 ≠ 回答变好，SFT 最显著的作用是"学会按指令格式组织回答"（见第 5/6 节），质量评价需要 probe 式问答对比。
4. **对比实验要注意口径**：打点间隔、lr、起点、数据顺序都会影响"首值/末值"这类统计量（见第 7 节），跨实验比较前先对齐口径。

---

## 9. 附录

### 9.1 复现命令

```bash
# exp1 / exp2（全参 SFT）
cd trainer
PYTHONUNBUFFERED=1 python train_full_sft.py --epochs 1 --save_weight sft_mini_ep1   # exp1
PYTHONUNBUFFERED=1 python train_full_sft.py --epochs 2 --save_weight sft_mini_ep2   # exp2

# exp3（LoRA，从 pretrain 出发，同数据同截断长度）
PYTHONUNBUFFERED=1 python train_lora.py --lora_name lora_sft_ep1 --epochs 1 \
    --learning_rate 1e-4 --max_seq_len 768 \
    --data_path ../dataset/sft_t2t_mini.jsonl --from_weight pretrain

# 同 8 问评测（eval_probe.py，在实例上执行）
python eval_probe.py --weight sft_mini_ep1      # 全参模型
python eval_probe.py --weight pretrain --lora lora_sft_ep1   # LoRA 模型
```

### 9.2 文件清单

```
experiments/
├── README.md                 # 索引
├── 后训练实验报告.md          # 本文档
├── img/                      # 图表（loss_curves / loss_stats / trainable_params / early_loss）
├── _tools/                   # 只有代码与配置
│   ├── make_charts.py        # 生成 img/ 下的图表
│   ├── assemble_report.py    # 生成本文档
│   ├── qwen_sft/             # LLaMA-Factory SFT（configs / data / 脚本）
│   └── qwen_grpo/            # TRL GRPO 脚本 + bench/ 评测工具
└── results/                  # 所有实验产物
    ├── mini_sft/             # 本节内容：exp1_ep1 / exp2_ep2 / exp3_lora_ep1
    │   ├── exp1_ep1/         # train.log + loss_curve.csv + probe_before/after + README
    │   ├── exp2_ep2/         # train.log + loss_curve.csv + probe_after_sft_ep2 + README
    │   └── exp3_lora_ep1/    # train.log + loss_curve.csv + probe_lora + README
    ├── qwen_sft/             # Qwen SFT 产物
    ├── qwen_grpo/            # Qwen GRPO 产物
    └── bench/                # base/SFT/GRPO 的 IFEval-lite + GSM8K 对比
```

### 9.3 模型权重（不入库，保存在实例数据盘）

| 文件 | 说明 |
|---|---|
| `sft_mini_ep1_768.pth`（137MB） | exp1 产物，后续 RL 实验的基线 |
| `sft_mini_ep2_768.pth`（137MB） | exp2 产物 |
| `lora_sft_ep1_768.pth`（797KB） | exp3 产物，需叠加 `pretrain_768.pth` 使用 |
"""


def build_probes_section():
    parts = []
    for rel, desc in PROBES:
        with open(os.path.join(BASE, rel), encoding="utf-8") as f:
            text = f.read().rstrip()
        parts.append(f"### 模型：{desc}\n")
        parts.append(f"来源文件：`experiments/{rel}`\n")
        # split into Q/A blocks
        blocks = text.split("\n\n")
        cur_q = None
        for b in blocks:
            b = b.strip()
            if not b:
                continue
            if b.startswith("Q") and ":" in b.split("\n")[0][:8]:
                parts.append(f"\n**{b}**\n")
            else:
                parts.append("<pre>" + html.escape(b) + "</pre>\n")
    return "\n".join(parts)


def main():
    body = build_probes_section()
    md = HEADER.replace("{{PROBES}}", body)
    out = os.path.join(BASE, "后训练实验报告.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write(md)
    print("report written:", out)
    print("probe <pre> blocks:", md.count("<pre>"))


if __name__ == "__main__":
    main()
