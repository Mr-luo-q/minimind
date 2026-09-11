# 评测结果（base vs SFT vs GRPO）

本目录是**结果的唯一归档处**——跑批结束后把结果集中放这里，便于查找与复用。

## 结果表（最终）

| 模型 | IFEval instruction-level | IFEval prompt-level | GSM8K |
|---|---|---|---|
| base（Qwen3-4B 底座） | **98.1%** (52/53) | **98.0%** (49/50) | 94.0% (94/100) |
| SFT（e1 LoRA, 1 epoch） | 88.7% (47/53) | 88.0% (44/50) | 84.0% (84/100) |
| GRPO（通用 RM, 250 步） | 96.2% (51/53) | 96.0% (48/50) | **95.0%** (95/100) |

> 权威版本以 `RESULTS.md` / `index.html` / `table.csv` 为准（本表是快照）。

> **为什么 base 反而最好？** 归因分析见 **[`WHY_BASE_WINS.md`](./WHY_BASE_WINS.md)** ——
> 逐题比对了三模型的得失：SFT 是"训错了目标"（用能力换文风），GRPO 是"奖励不针对目标"（没训坏也没提升）。

## 先看哪个文件

| 想看什么 | 打开 |
|---|---|
| **一张对比表 + 每题明细（推荐）** | `index.html` —— 单文件网页，浏览器直接打开，无需服务器 |
| 贴进报告的 Markdown 表 | `RESULTS.md` |
| **为什么 base 反而最好（归因）** | `WHY_BASE_WINS.md` |
| **哪一类约束最容易被违反** | `ANALYSIS.md`（约束类型通过率 + 全员都挂的题） |
| 原始分数（表格数据） | `table.csv` |
| 逐题原始明细（含模型回答片段） | `raw/ifeval_results_<tag>.jsonl`、`raw/gsm8k_results_<tag>.jsonl` |
| 各次跑批的原始拉取 | `batches/` |
| 首次跑批（旧判分口径） | `archive/run1_v1_split_checker/` |

## ⚠️ 判分口径修过两次，三次跑批的分数不可混用

IFEval 的判分器修了两个**会让对比结论失真**的缺陷。这不是调优，是不修就会得出错误结论：

| 跑批 | 判分器 | 缺陷与影响 | base |
|---|---|---|---|
| **v1** `archive/run1_v1_split_checker/` | `len(text.split())` + 严格结尾 | ① 中文无空格分词：127 汉字的回答被算成 **1 个词** → `min_words` **恒不通过**、`max_words` **恒通过**（10/53 约束在测空东西）<br>② 结尾过严：`……此致敬礼。` 因多一个句号被判 **False** | 84.9% |
| **v2** `batches/results_v2-ifeval/` | `count_units()` | 长度口径修正（CJK 逐字计） | 92.5% |
| **v3** `raw/`（最终） | `count_units()` + `end_with` 容错 | 结尾剥掉尾部标点/引号/markdown 符后再比对 | **98.1%** |

影响有多大：base 的 IFEval 从 84.9% 一路修到 98.1%——**13 个百分点全是判分缺陷吃掉的**，
且每个缺陷都同时压低了三个模型（属系统性偏差，不是偏袒某个模型）。

**三次跑批的分数不能直接比较。** 每个结果目录的 `manifest.json` 记录了它属哪版口径，
`gen_table.py` 会把口径打印在表头，`consistency_problems()` 还会校验"明细反算的通过数"与
"CSV 的汇总值"是否一致——不一致会直接在表上标红。

## 结论要点

1. **base 的指令遵循最强**（98.1% / 98.0%），SFT 和 GRPO 都没能超过它；
2. **SFT 是唯一明显退化的**：IFEval −9.4pt、GSM8K −10.0pt。
   训练数据是通用对话、只训 1 epoch，约束纪律与数学能力都被稀释——"SFT 让回答更像样" ≠ "能力变强"；
3. **GRPO 几乎追平 base 的指令遵循**（96.2%），GSM8K 略高（95% vs 94%，100 题下 1 题差无统计意义）；
4. **最容易被违反的约束是 `end_with`**：SFT/GRPO 67%、base 100%。
   模型常"说了指定短语但后面又补一句"（如 GRPO 的 `书中自有黄金屋，亦有心灵的宝藏。`）；
5. `max_words`（超长）与 `no_commas` 是 SFT 的另两个弱项——都是**约束纪律**类，不是知识类。

## 评测口径（必须随结果一起引用，否则数字会被误读）

- **IFEval-lite**：50 条中文指令、53 个可验证约束、规则自动判分 → instruction-level（约束通过率）与 prompt-level（整题全过率）。
  **这是自实现轻量版，分数只用于 base/SFT/GRPO 之间的横向对比，不代表官方 IFEval 榜单成绩。**
- **GSM8K**：官方 test 集前 100 题、贪心解码、`####` 后数字精确比对。
- **统一关闭思考模式**（`--no-think`）：Qwen3 chat template 默认开启思考模式，实测 base/GRPO 会把 600 token 预算耗在思考段上、答案被截断而答错；三个模型统一在非思考模式作答，口径才一致。
- **prompt 格式统一走 chat template**：SFT 用 `template: qwen`、GRPO 用 `apply_chat_template` 训练，给它们喂纯文本提示会低估这两个模型。

## 怎么重新生成

```powershell
# 一键：从实例拉取 → 合并 → 渲染表 + 分析（需要实例开机、SSH 端口正确）
# 端口在 AutoDL 控制台每次开机都会变，记得改后面的值
.\fetch_and_render.ps1 -Port 39365

# 只重新渲染本地表（不联网、不需要实例、不需要 torch）
python .\gen_table.py --dir .\raw
python .\analyze_constraints.py --dir .\raw
```

管线是"实例上只负责跑推理，本地负责合并与渲染"：

```
实例: batches/results_v3-ifeval/*/   (IFEval 重跑，每模型一个子目录)
      batches/results-gsm8k/         (GSM8K，首次跑批)
   ↓ merge_results.py  --ifeval-v2 ... --gsm8k-v1 ... --out raw/
raw/  (最终数据集 + manifest.json)
   ↓ gen_table.py / analyze_constraints.py
RESULTS.md / index.html / table.csv / ANALYSIS.md
```

> `gen_table.py`、`analyze_constraints.py` 与仓库 `experiments/_tools/qwen_grpo/bench/` 下的同名脚本是同一份，
> 本地副本便于离线重渲。两者都**不需要 torch**（`analyze_constraints.py` 用 `ast` 静态解析题集）。
> 脚本幂等：跑批中途执行会如实标出"⏳ 部分完成 / ⬜ 未跑"，不会把没跑的模型显示成 0 分。

## 目录约定

```
bench_results/
├─ README.md                    ← 本文件
├─ index.html                   ← 网页版结果表（含每题明细）
├─ RESULTS.md                   ← Markdown 对比表
├─ ANALYSIS.md                  ← 约束类型分析
├─ table.csv                    ← 表格数据
├─ gen_table.py                 ← 汇总/渲染（本地副本）
├─ analyze_constraints.py       ← 约束类型分析（本地副本）
├─ merge_results.py             ← 合并"IFEval 重跑 + GSM8K 首跑"（本地副本）
├─ fetch_and_render.ps1         ← 一键：拉取 → 合并 → 渲染
├─ raw/                         ← 最终结果集（v3 判分口径）+ manifest.json
├─ batches/                     ← 各次跑批的原始拉取
│  ├─ results_v3-ifeval/
│  ├─ results_v2-ifeval/
│  └─ results-gsm8k/
└─ archive/
   └─ run1_v1_split_checker/    ← v1 跑批（判分缺陷未修，勿与最终结果混用）
```
