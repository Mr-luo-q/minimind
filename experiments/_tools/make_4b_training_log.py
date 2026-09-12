# -*- coding: utf-8 -*-
"""
Qwen3-4B 训练日志：从原始日志抽取指标 → 制图 → 输出带图文档。

数据源（全部是仓库里已有的原始日志，不引入新数据）：
  results/qwen_sft/trainer_e1_qwen3-4b_lora.jsonl   SFT 冷启动（313 步）
  results/qwen_grpo/grpo_formal.log                 GRPO 正式训练（250 步，25 个指标）
  results/qwen_grpo/grpo_formal_vram.csv            GRPO 显存采样（1s 间隔）

产物：
  img/log4b_*.png                        图表
  results/qwen_4b_TRAINING_LOG.md        带图文档

用法: python make_4b_training_log.py
"""
import ast
import csv
import json
import os
import re
import statistics as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

for fp in (r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simhei.ttf"):
    if os.path.exists(fp):
        font_manager.fontManager.addfont(fp)
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 130

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # experiments/
IMG = os.path.join(BASE, "img")
RES = os.path.join(BASE, "results")
os.makedirs(IMG, exist_ok=True)

SFT_JSONL = os.path.join(RES, "qwen_sft", "trainer_e1_qwen3-4b_lora.jsonl")
GRPO_LOG = os.path.join(RES, "qwen_grpo", "grpo_formal.log")
VRAM_CSV = os.path.join(RES, "qwen_grpo", "grpo_formal_vram.csv")
DOC = os.path.join(RES, "qwen_4b_TRAINING_LOG.md")


# ---------------- 抽取 ----------------
def load_sft():
    rows = []
    for line in open(SFT_JSONL, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        o = json.loads(line)
        if "loss" in o:
            rows.append(o)
    return rows


def load_grpo():
    """日志每步打印一个 Python dict（单引号），用 ast 解析比正则稳。"""
    rows = []
    for line in open(GRPO_LOG, encoding="utf-8", errors="replace"):
        line = line.strip()
        if not line.startswith("{") or "'step_time'" not in line:
            continue
        try:
            rows.append(ast.literal_eval(line))
        except (ValueError, SyntaxError):
            pass
    return rows


def load_vram():
    rows = []
    with open(VRAM_CSV, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try:
                rows.append((int(r["time_utc"]), float(r["util_pct"]), float(r["vram_mb"])))
            except (ValueError, KeyError):
                pass
    rows.sort()
    return rows


def smooth(xs, w):
    """居中滑动平均；w<=1 原样返回。"""
    if w <= 1 or len(xs) <= w:
        return list(xs)
    out, h = [], w // 2
    for i in range(len(xs)):
        lo, hi = max(0, i - h), min(len(xs), i + h + 1)
        out.append(sum(xs[lo:hi]) / (hi - lo))
    return out


# ---------------- 图表 ----------------
def chart_sft(sft):
    steps = [r["current_steps"] for r in sft]
    loss = [r["loss"] for r in sft]
    lr = [r["lr"] for r in sft]
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.plot(steps, loss, "o-", ms=3.5, lw=1.5, color="#1f77b4", label="训练 loss（每 20 步打点）")
    ax.plot(steps, smooth(loss, 5), "-", lw=2.2, color="#d62728", alpha=.85, label="滑动平均（窗口5）")
    lo = min(range(len(loss)), key=lambda i: loss[i])
    # 标注统一放到曲线下方的空白区，避免出界或压住曲线/标题
    ax.annotate(f"首值 {loss[0]:.4f} @ step {steps[0]}（首个打点）", xy=(steps[0], loss[0]),
                xytext=(steps[0] + 30, loss[0] - 0.085), fontsize=8.5, color="#333",
                arrowprops=dict(arrowstyle="->", color="#666", lw=.9))
    ax.annotate(f"最低 {loss[lo]:.4f} @ step {steps[lo]}", xy=(steps[lo], loss[lo]),
                xytext=(steps[lo] + 34, loss[lo] - 0.012), fontsize=8.5, color="#333",
                arrowprops=dict(arrowstyle="->", color="#666", lw=.9))
    ax.set_ylim(min(loss) - 0.11, max(loss) + 0.035)
    ax.set_xlabel("step"); ax.set_ylabel("loss")
    ax.set_title(f"SFT 冷启动：Qwen3-4B LoRA（共 {steps[-1]} 步 / 1 epoch / 1 万条，打点间隔 20 步）")
    ax.grid(alpha=.3); ax.legend(fontsize=9)
    ax2 = ax.twinx()
    ax2.plot(steps, lr, "--", lw=1.1, color="#7f7f7f", alpha=.7, label="learning rate")
    ax2.set_ylabel("learning rate", color="#7f7f7f"); ax2.tick_params(axis="y", colors="#7f7f7f")
    ax2.legend(fontsize=8, loc="center right")
    fig.tight_layout(); p = os.path.join(IMG, "log4b_sft_loss.png"); fig.savefig(p); plt.close(fig)
    return p


def chart_grpo_overview(g):
    steps = list(range(1, len(g) + 1))
    reward = [r["reward"] for r in g]
    kl = [r["kl"] for r in g]
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7))

    ax = axes[0][0]
    ax.plot(steps, reward, lw=.8, color="#c8c8c8", alpha=.8, label="每步")
    ax.plot(steps, smooth(reward, 21), lw=2.2, color="#1f77b4", label="滑动平均（窗口21）")
    ax.axhline(0, color="#999", lw=.8, ls=":")
    f20, l20 = sum(reward[:20]) / 20, sum(reward[-20:]) / 20
    # 均值直接用图例说明，避免文字标注压住曲线
    ax.axhline(f20, color="#d62728", lw=1.2, ls="--", label=f"前20步均值 {f20:.2f}")
    ax.axhline(l20, color="#2ca02c", lw=1.2, ls="--", label=f"后20步均值 {l20:.2f}（+{l20-f20:.2f}）")
    ax.set_ylim(min(reward) - .25, max(reward) + .45)
    ax.set_title("① 奖励（InternLM2-1.8B-Reward，范围 ±3）")
    ax.set_xlabel("step"); ax.set_ylabel("reward"); ax.grid(alpha=.3)
    ax.legend(fontsize=8, loc="lower right", framealpha=.95)

    ax = axes[0][1]
    ax.plot(steps, kl, lw=.9, color="#bbb")
    ax.plot(steps, smooth(kl, 21), lw=2.2, color="#ff7f0e")
    ax.set_title(f"② KL 散度（β=0.04）—— 全程 ≤{max(kl):.4f}，策略几乎没离开底座")
    ax.set_xlabel("step"); ax.set_ylabel("kl"); ax.grid(alpha=.3)

    ax = axes[1][0]
    L = [r["completions/mean_length"] for r in g]
    # y 轴放宽到 340，否则贴着 256 的曲线会被坐标轴和虚线压住看不见
    ax.plot(steps, L, lw=1.8, color="#9467bd", marker="o", ms=1.6, label="生成长度")
    ax.axhline(256, color="#d62728", lw=1.2, ls="--")
    ax.text(6, 262, "上限 max_completion_length = 256", color="#d62728", fontsize=9)
    ng = [r["completions/mean_terminated_length"] for r in g]
    ax.plot(steps, ng, lw=1.4, color="#2ca02c", label="自然结束长度（恒为 0）")
    ax.set_ylim(-15, 340)
    ax.set_title("③ 生成长度：全程贴在上限（100% 被截断）")
    ax.set_xlabel("step"); ax.set_ylabel("tokens"); ax.legend(fontsize=8, loc="center right"); ax.grid(alpha=.3)

    ax = axes[1][1]
    gn = [r["grad_norm"] for r in g]
    ax.plot(steps, gn, lw=.8, color="#bbb")
    ax.plot(steps, smooth(gn, 21), lw=2.2, color="#17becf")
    ax.set_title("④ 梯度范数")
    ax.set_xlabel("step"); ax.set_ylabel("grad_norm"); ax.grid(alpha=.3)

    fig.suptitle("GRPO 正式训练总览：Qwen3-4B LoRA + 通用奖励模型（250 步）", fontsize=12.5)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    p = os.path.join(IMG, "log4b_grpo_overview.png"); fig.savefig(p); plt.close(fig)
    return p


def chart_grpo_detail(g):
    steps = list(range(1, len(g) + 1))
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4))

    ax = axes[0]
    mean = [r["reward"] for r in g]; std = [r["reward_std"] for r in g]
    ax.plot(steps, smooth(std, 21), lw=2.2, color="#8c564b")
    ax.axhline(0, color="#999", lw=.8, ls=":")
    z = [r["frac_reward_zero_std"] for r in g]
    ax.set_title(f"① 奖励标准差（全程 >0，均值 {st.mean(std):.3f}）\n"
                 f"组内零方差比例 frac_reward_zero_std = {st.mean(z):.3f}")
    ax.set_xlabel("step"); ax.set_ylabel("reward std"); ax.grid(alpha=.3)

    ax = axes[1]
    ent = [r["entropy"] for r in g]
    ax.plot(steps, ent, lw=.8, color="#bbb")
    ax.plot(steps, smooth(ent, 21), lw=2.2, color="#e377c2")
    ax.set_title(f"② 策略熵（均值 {st.mean(ent):.3f}）\n熵未见持续下降 → 未发生模式崩溃")
    ax.set_xlabel("step"); ax.set_ylabel("entropy"); ax.grid(alpha=.3)

    ax = axes[2]
    # 注意: 日志里的 num_tokens 是【累计值】(0 -> 56万)，直接画会是一条假直线。
    # 必须做差分才是"每步 token"。每步 ≈ batch2 x num_gen4 x 256上限 + prompt token
    nt = [r["num_tokens"] for r in g]
    per_step = [nt[0]] + [b - a for a, b in zip(nt, nt[1:])]
    ax.plot(steps, smooth(per_step, 21), lw=2.2, color="#bcbd22", label="每步 token（差分）")
    ax.axhline(2048, color="#d62728", lw=1.2, ls="--", label="生成下限 2×4×256=2048")
    ax.set_ylim(1900, max(per_step) * 1.05)
    ax.set_title(f"③ 每步 token（均值 {st.mean(per_step):,.0f}）\n"
                 f"num_tokens 为累计值，此处已差分")
    ax.set_xlabel("step"); ax.set_ylabel("tokens/step"); ax.grid(alpha=.3)
    ax.legend(fontsize=8, loc="upper left")

    fig.suptitle("GRPO 训练信号诊断：奖励方差 / 策略熵 / 吞吐", fontsize=12.5)
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    p = os.path.join(IMG, "log4b_grpo_detail.png"); fig.savefig(p); plt.close(fig)
    return p


def chart_vram(vram, runtime_s=None):
    if not vram:
        return None
    t0 = vram[0][0]
    mins = [(t - t0) / 60 for t, _, _ in vram]
    vram_gb = [v / 1024 for _, _, v in vram]
    util = [u for _, u, _ in vram]
    fig, ax = plt.subplots(figsize=(11, 4.3))
    ax.plot(mins, vram_gb, lw=.9, color="#1f77b4", alpha=.8, label="显存占用 (GB)")
    ax.axhline(96, color="#333", lw=1.1, ls=":", label="卡容量 96 GB")
    ax.plot(mins, [u / 100 * 96 for u in util], lw=.8, color="#ff7f0e", alpha=.4,
            label="GPU 利用率（折算到 96GB 轴）")

    # 如果知道真实训练时长，把训练窗口标出来——好把"训练期峰值"与"跑批前后开销"分开
    if runtime_s:
        tmin = runtime_s / 60
        ax.axvspan(0, tmin, color="#2ca02c", alpha=.07)
        in_win = [v for m, v in zip(mins, vram_gb) if m <= tmin]
        if in_win:
            pk = max(in_win)
            ax.axhline(pk, color="#2ca02c", lw=1.2, ls="--",
                       label=f"训练期峰值 {pk:.1f} GB（≤{tmin:.0f} min）")
            ax.text(tmin + 1.5, pk + 1.5, f"训练期峰值 {pk:.1f} GB", fontsize=9, color="#2ca02c")

    peak = max(vram_gb); pk_i = vram_gb.index(peak)
    ax.annotate(f"全窗口峰值 {peak:.1f} GB @ {mins[pk_i]:.1f} min（训练已结束）",
                xy=(mins[pk_i], peak), xytext=(mins[pk_i] - 46, peak + 3.5), fontsize=9,
                color="#c0392b", arrowprops=dict(arrowstyle="->", color="#c0392b", lw=.9))
    ax.set_xlabel("采样时间（分钟，自采样开始）"); ax.set_ylabel("GB / 折算利用率")
    ax.set_ylim(-3, 104)
    ax.set_title(f"GRPO 训练显存与利用率（采样 {len(vram)} 点，间隔 1s，窗口 {mins[-1]:.1f} 分钟）")
    ax.grid(alpha=.3); ax.legend(fontsize=8.5, loc="center right", framealpha=.95)
    fig.tight_layout(); p = os.path.join(IMG, "log4b_grpo_vram.png"); fig.savefig(p); plt.close(fig)
    return p


# ---------------- 文档 ----------------
def table_steps(g, every=25):
    L = ["| step | reward | reward_std | KL | 熵 | 生成长度 | 截断比 | step_time |",
         "|---|---|---|---|---|---|---|---|"]
    for i in range(every - 1, len(g), every):
        r = g[i]
        L.append("| {} | {:.3f} | {:.3f} | {:.4f} | {:.3f} | {:.0f} | {:.2f} | {:.1f}s |".format(
            i + 1, r["reward"], r["reward_std"], r["kl"], r["entropy"],
            r["completions/mean_length"], r["completions/clipped_ratio"], r["step_time"]))
    return L


def main():
    sft, g, vram = load_sft(), load_grpo(), load_vram()
    assert sft and g, "数据抽取失败"

    # 先从日志取真实训练时长——显存图要靠它把"训练期"与"跑批前后"分开
    runtime = None
    for line in open(GRPO_LOG, encoding="utf-8", errors="replace"):
        m = re.search(r"'train_runtime':\s*([\d.]+)", line)
        if m:
            runtime = float(m.group(1))

    p_sft = chart_sft(sft)
    p_ov = chart_grpo_overview(g)
    p_dt = chart_grpo_detail(g)
    p_vr = chart_vram(vram, runtime)

    # --- 统计量 ---
    loss = [r["loss"] for r in sft]
    steps_sft = [r["current_steps"] for r in sft]
    reward = [r["reward"] for r in g]
    kl = [r["kl"] for r in g]
    mean_len = [r["completions/mean_length"] for r in g]
    clipped = [r["completions/clipped_ratio"] for r in g]
    st_time = [r["step_time"] for r in g]
    f20, l20 = sum(reward[:20]) / 20, sum(reward[-20:]) / 20
    vpeak = max(v / 1024 for _, _, v in vram)
    vmean = st.mean(v / 1024 for _, _, v in vram)
    umean = st.mean(u for _, u, _ in vram)
    vmins = (vram[-1][0] - vram[0][0]) / 60
    # 训练窗口内的显存峰值（与《边界扫描结果.md》口径一致，应接近 22.7 GB）
    vpeak_train = max(v / 1024 for t, _, v in vram if runtime and (t - vram[0][0]) / 60 <= runtime / 60)

    rel = lambda p: "../img/" + os.path.basename(p)

    D = [f"""# Qwen3-4B 训练日志

> 本文档由 `_tools/make_4b_training_log.py` 从原始日志自动生成，图表与数字均可复现。
> 数据源：`results/qwen_sft/trainer_e1_qwen3-4b_lora.jsonl`、`results/qwen_grpo/grpo_formal.log`、
> `results/qwen_grpo/grpo_formal_vram.csv`。
> 评测结果与归因分析见 [`bench/RESULTS.md`](bench/RESULTS.md) 与 [`bench/WHY_BASE_WINS.md`](bench/WHY_BASE_WINS.md)。

## 0. 一览

| | SFT 冷启动 | GRPO 正式训练 |
|---|---|---|
| 底座 | Qwen3-4B（真 base） | Qwen3-4B（真 base，**跳过 SFT**） |
| 方法 | LoRA rank16 / alpha32 / 全 7 个投影层 | LoRA rank32 + GRPO |
| 数据 | MiniMind `sft_t2t_mini` 抽 1 万条 | `prompts_2k.jsonl`（2 万 prompts） |
| 训练信号 | 模仿对话 | InternLM2-1.8B-Reward 通用偏好分 |
| 超参 | lr **2e-4**，1 epoch，cutoff 1024 | lr 1e-5，β=0.04，num_gen 4 |
| 步数 / 时长 | 313 步（打点 15 次）/ 20 分 58 秒 | 250 步（每步打点）/ **{runtime/60:.1f} 分钟** |
| 硬件 | AutoDL RTX 5090 32GB | **RTX PRO 6000 Blackwell 96GB** |
| loss / reward | {loss[0]:.4f} → {loss[-1]:.4f}（最低 {min(loss):.4f}） | {reward[0]:.3f} → {reward[-1]:.3f}（前20均值 {f20:.2f} → 后20均值 {l20:.2f}） |

## 1. SFT 冷启动

![SFT loss 曲线]({rel(p_sft)})

- loss **{loss[0]:.4f} → {loss[-1]:.4f}**（首末打点，`logging_steps=20`），最低 **{min(loss):.4f}**（step {steps_sft[loss.index(min(loss))]}）；
- 前 60 步下降最陡（{loss[0]:.3f} → {loss[2]:.3f}），此后 lr 余弦衰减、loss 在 0.80–0.85 之间震荡不再下探；
- **1 万条数据上 loss 压到 {min(loss):.4f} 属过拟合式拟合**，是后续"用预训练能力换文风"的直接原因
  （见 `bench/WHY_BASE_WINS.md` 第 2 节）。

## 2. GRPO 正式训练

![GRPO 总览]({rel(p_ov)})

**① 奖励**：前 20 步均值 **{f20:.2f}** → 后 20 步均值 **{l20:.2f}**（+{l20-f20:.2f}）。
注意这是**通用偏好分**在涨（回答更像样），**不等于任务能力在涨**——同一 checkpoint 在
GSM8K 上 95.0%（base 94.0%）、IFEval 96.2%（base 98.1%），基本原地不动。

**② KL**：全程 **≤ {max(kl):.4f}**（均值 {st.mean(kl):.4f}）。策略几乎没有离开底座，
这既解释了"为什么没训坏"，也解释了"为什么没有提升"。

**③ 生成长度**：**每一步都贴在 256 token 上限**，`clipped_ratio` 全程 = **{st.mean(clipped):.2f}**，
`mean_terminated_length` = 0 —— **没有任何一条 rollout 自然结束**，训练信号建立在被截断的文本上。

**④ 梯度范数**：均值 {st.mean([r['grad_norm'] for r in g]):.3f}，无异常尖峰。

![GRPO 信号诊断]({rel(p_dt)})

## 3. 显存与吞吐

![显存曲线]({rel(p_vr)})

| 指标 | 值 |
|---|---|
| GPU | NVIDIA RTX PRO 6000 Blackwell Server Edition（96 GB） |
| **训练期显存峰值** | **{vpeak_train:.1f} GB**（与《边界扫描结果.md》记录的 22.7 GB 同口径） |
| 全采样窗口峰值 | {vpeak:.1f} GB（出现在训练结束之后，属跑批收尾开销） |
| 显存均值（采样） | {vmean:.1f} GB |
| GPU 利用率均值 | {umean:.0f}% |
| 采样窗口 | {vmins:.1f} 分钟（{len(vram)} 点，间隔 1s） |
| 单步耗时 | 均值 {st.mean(st_time):.2f}s（min {min(st_time):.2f}s / max {max(st_time):.2f}s） |
| **训练总时长** | **{runtime:.0f} 秒 = {runtime/60:.1f} 分钟** |

> ⚠️ **两个峰值数字口径不同，别混用**：
> 采样窗口（{vmins:.1f} 分钟）比训练本身（{runtime/60:.1f} 分钟）**多出 {vmins - runtime/60:.1f} 分钟**，
> 包含模型加载与跑批前后开销 —— 图中绿色阴影是真实训练期，可以看到 {vpeak:.1f} GB 那个尖峰落在阴影之外。
> **汇报时用训练期峰值 {vpeak_train:.1f} GB（≈报告记录的 22.7 GB）。**

## 4. 逐步指标（每 25 步采样）

{chr(10).join(table_steps(g))}

## 5. 硬件与配置摘要

```
GPU      : NVIDIA RTX PRO 6000 Blackwell Server Edition (96 GB)
SFT      : Qwen3-4B + LoRA r16/a32, lr 2e-4, 1 epoch, 1万条, cutoff 1024,
           313 步（日志末打点 300）, 20分58秒, 显存峰值约 24 GB (RTX 5090 32GB)
GRPO     : Qwen3-4B + LoRA r32, lr 1e-5, beta 0.04, num_generations 4,
           max_completion_length 256, batch 2 x grad_accum 4,
           250 步, {runtime/60:.1f} 分钟, 峰值 22.7 GB, 步时约 22 s
奖励模型 : InternLM2-1.8B-Reward (fp16)
```

## 6. 从这批日志读出的三个问题

1. **奖励与目标无关**：奖励分在涨，但 GSM8K/IFEval 不动 —— 通用偏好分不等于任务能力；
2. **KL 太小（≤{max(kl):.4f}）**：250 步几乎没有位移，底座能力保住了，但也没学到东西；
3. **rollout 100% 截断**：`clipped_ratio` 恒为 1、终止长度恒为 0，所有候选都被截成半句话，
   会压缩组内奖励方差（实测均值 {st.mean([r['reward_std'] for r in g]):.3f}），削弱 GRPO 的优势信号。

> 这三点已写入第二阶段方案 [`EXPERIMENT_PLAN_V2.md`](../EXPERIMENT_PLAN_V2.md)：
> 改用**规则奖励（RLVR）**、**加大策略位移**、并把 `clipped_ratio` 与
> `frac_reward_zero_std` 列为阶段 0 的前置检查项。
"""]

    open(DOC, "w", encoding="utf-8").write("\n".join(D))
    print("charts:", os.path.basename(p_sft), os.path.basename(p_ov), os.path.basename(p_dt), os.path.basename(p_vr))
    print("doc   :", DOC)
    print(f"SFT {len(sft)} 点 | GRPO {len(g)} 步 | VRAM {len(vram)} 点 | runtime {runtime}s")


if __name__ == "__main__":
    main()
