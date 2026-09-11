# base vs SFT vs GRPO 评测结果

> 生成时间：2026-09-11 21:02:11 ｜ 评测口径：IFEval-lite 50 条中文指令（规则自动判分）+ GSM8K test 前 100 题，贪心解码，**统一关闭思考模式**（`--no-think`）。

> ⚠️ 未找到 `manifest.json`：无法确认这批分数用的哪版判分口径。**v1（split 计长，对中文失效）与修正版的结果不可直接比较**。

> **说明**：IFEval-lite 为自实现轻量版，分数用于**模型间横向对比**，不代表官方 IFEval 榜单成绩。

| 模型 | IFEval instruction-level | IFEval prompt-level | GSM8K | 状态 |
|---|---|---|---|---|
| base（Qwen3-4B 底座） | 84.9% (53 约束) | 84.0% (50 题) | 94.0% (94/100) | ✅ 完成 |
| SFT（e1 LoRA, 1 epoch） | 81.1% (53 约束) | 80.0% (50 题) | 84.0% (84/100) | ✅ 完成 |
| GRPO（通用 RM, 250 步） | 83.0% (53 约束) | 82.0% (50 题) | 95.0% (95/100) | ✅ 完成 |

## 原始数据

| 文件 | 说明 |
|---|---|
| `ifeval_results.csv` / `gsm8k_results.csv` | 汇总分数（追加写） |
| `ifeval_results_<tag>.jsonl` / `gsm8k_results_<tag>.jsonl` | 每题明细（含模型回答片段 / ref vs pred） |
| `ANALYSIS.md` | 约束类型分析（哪类约束最容易被违反） |
| `run_main.log` | 完整运行日志 |
| `index.html` | 网页版结果表（含每题明细） |

