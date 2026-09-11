# -*- coding: utf-8 -*-
"""Generate loss-curve charts for the MiniMind post-training lab report."""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

for fp in (r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simhei.ttf"):
    if os.path.exists(fp):
        font_manager.fontManager.addfont(fp)
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # experiments/
IMG = os.path.join(BASE, "img")
os.makedirs(IMG, exist_ok=True)

EXPS = [
    ("exp1_sft_mini_ep1", "exp1 全参 SFT 1 epoch (lr=1e-5)"),
    ("exp2_sft_mini_ep2", "exp2 全参 SFT 2 epochs (lr=1e-5)"),
    ("exp3_lora_sft_ep1", "exp3 LoRA 1 epoch (lr=1e-4)"),
]

def load(name):
    path = os.path.join(BASE, name, "loss_curve.csv")
    steps, losses = [], []
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            steps.append(float(row["step"]))
            losses.append(float(row["loss"]))
    return steps, losses

def smooth(seq, window):
    if window <= 1:
        return seq
    out = []
    half = window // 2
    for i in range(len(seq)):
        lo, hi = max(0, i - half), min(len(seq), i + half + 1)
        out.append(sum(seq[lo:hi]) / (hi - lo))
    return out

def norm01(xs, total):
    return [x / total * 100 for x in xs]

# ---------- Chart 1: overlaid loss curves (x = % of training) ----------
fig, ax = plt.subplots(figsize=(10, 5.5), dpi=150)
colors = {"exp1_sft_mini_ep1": "#1f77b4", "exp2_sft_mini_ep2": "#d62728", "exp3_lora_sft_ep1": "#2ca02c"}
for name, label in EXPS:
    steps, losses = load(name)
    total = steps[-1]
    w = max(3, len(losses) // 250)
    ax.plot(norm01(steps, total), smooth(losses, w), color=colors[name], lw=1.8, label=f"{label}  (loss_min={min(losses):.3f})")
    imin = losses.index(min(losses))
    ax.plot(norm01(steps, total)[imin], min(losses), "o", color=colors[name], ms=5)
ax.set_xlabel("训练进度 (%)")
ax.set_ylabel("loss")
ax.set_title("三种后训练方法的 loss 曲线对比（横轴按各自总步数归一化）")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(IMG, "loss_curves.png"))
plt.close(fig)

# ---------- Chart 2: first / last / min loss grouped bar ----------
fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
names = [e[0] for e in EXPS]
labels = ["exp1\n全参1ep", "exp2\n全参2ep", "exp3\nLoRA1ep"]
firsts, lasts, mins = [], [], []
for name, _ in EXPS:
    _, losses = load(name)
    firsts.append(losses[0]); lasts.append(losses[-1]); mins.append(min(losses))
import numpy as np
x = np.arange(len(names))
w = 0.26
b1 = ax.bar(x - w, firsts, w, label="loss 首值", color="#ff9896")
b2 = ax.bar(x, mins, w, label="loss 最小值", color="#98df8a")
b3 = ax.bar(x + w, lasts, w, label="loss 末值", color="#aec7e8")
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_ylabel("loss")
ax.set_title("各实验 loss 首值 / 最小值 / 末值对比")
for bars in (b1, b2, b3):
    for b in bars:
        ax.annotate(f"{b.get_height():.2f}", (b.get_x() + b.get_width() / 2, b.get_height()),
                    ha="center", va="bottom", fontsize=8)
ax.legend()
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(IMG, "loss_stats.png"))
plt.close(fig)

# ---------- Chart 3: LoRA param share (log-scale bar: trainable params) ----------
fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
cats = ["exp1/exp2 全参 SFT\n可训练参数", "exp3 LoRA\n可训练参数(adapter)"]
vals = [63.91, 0.797]
bars = ax.bar(cats, vals, color=["#1f77b4", "#2ca02c"], width=0.5)
for b in bars:
    ax.annotate(f"{b.get_height():.2f} M", (b.get_x() + b.get_width() / 2, b.get_height()),
                ha="center", va="bottom", fontsize=11)
ax.set_ylabel("可训练参数量 (M)")
ax.set_title("全参微调 vs LoRA：实际更新的参数量")
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(IMG, "trainable_params.png"))
plt.close(fig)

# ---------- Chart 4: early-phase zoom (why LoRA starts higher) ----------
fig, ax = plt.subplots(figsize=(10, 5.5), dpi=150)
for name, label in EXPS:
    steps, losses = load(name)
    ax.plot(steps, losses, color=colors[name], lw=1.2, alpha=0.9, label=label,
            marker="o" if len(steps) < 60 else None, ms=3)
ax.set_xlim(0, 2500)
ax.set_xlabel("step")
ax.set_ylabel("loss")
ax.set_title("初始阶段 loss 对比（前 2500 步，未经平滑）")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
# annotate exp3 spike
s3, l3 = load("exp3_lora_sft_ep1")
i20 = s3.index(20)
ax.annotate(f"LoRA 最高点 {l3[i20]:.3f} @ step 20（lr=1e-4 初始过冲）",
            xy=(20, l3[i20]), xytext=(700, 2.75),
            arrowprops=dict(arrowstyle="->", color="#2ca02c"), color="#2ca02c", fontsize=9)
ax.annotate("全参 SFT lr=1e-5，平稳起步",
            xy=(300, 1.91), xytext=(900, 2.1),
            arrowprops=dict(arrowstyle="->", color="#1f77b4"), color="#1f77b4", fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(IMG, "early_loss.png"))
plt.close(fig)

print("charts written to", IMG)
