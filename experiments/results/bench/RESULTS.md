# base vs SFT vs GRPO 评测结果

> 生成时间：2026-09-11 21:42:55 ｜ 评测口径：IFEval-lite 50 条中文指令（规则自动判分）+ GSM8K test 前 100 题，贪心解码，**统一关闭思考模式**（`--no-think`）。

> 本次口径（来自 `manifest.json`）：长度判分 `count_units (CJK 逐字计) + end_with 容许尾部标点/引号/markdown 符`；chat_template=yes (gsm8k 也走 chat template)；no_think=True；创建于 2026-09-11 21:42:54；备注：最终结果集：IFEval 用修正版判分器（长度 count_units + 结尾容错），GSM8K 沿用首次跑批

> **说明**：IFEval-lite 为自实现轻量版，分数用于**模型间横向对比**，不代表官方 IFEval 榜单成绩。

| 模型 | IFEval instruction-level | IFEval prompt-level | GSM8K | 状态 |
|---|---|---|---|---|
| base（Qwen3-4B 底座） | 98.1% (53 约束) | 98.0% (50 题) | 94.0% (94/100) | ✅ 完成 |
| SFT（e1 LoRA, 1 epoch） | 88.7% (53 约束) | 88.0% (50 题) | 84.0% (84/100) | ✅ 完成 |
| GRPO（通用 RM, 250 步） | 96.2% (53 约束) | 96.0% (50 题) | 95.0% (95/100) | ✅ 完成 |

## 原始数据

| 文件 | 说明 |
|---|---|
| `ifeval_results.csv` / `gsm8k_results.csv` | 汇总分数（追加写） |
| `ifeval_results_<tag>.jsonl` / `gsm8k_results_<tag>.jsonl` | 每题明细（含模型回答片段 / ref vs pred） |
| `ANALYSIS.md` | 约束类型分析（哪类约束最容易被违反） |
| `run_main.log` | 完整运行日志 |
| `index.html` | 网页版结果表（含每题明细） |

