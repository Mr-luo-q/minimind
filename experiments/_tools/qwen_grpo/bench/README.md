# Bench：IFEval-lite + GSM8K 评测（base vs SFT vs GRPO）

本目录是**模型间横向对比**的评测工具，不是官方榜单复现。

| 文件 | 作用 |
|---|---|
| `ifeval_lite.py` | 指令遵循评测：50 条中文指令 + 10 类规则判分约束 → instruction-level / prompt-level 通过率 + 每题明细 |
| `gsm8k_eval.py` | 数学评测：GSM8K test 前 N 题，`#### 后数字` 精确比对 → 正确率 + 每题明细；`extract_predicted_answer` 可复用为 RLVR 的规则奖励 |
| `run_bench.sh` | **一键跑 3 个模型 × 2 个 benchmark**，并汇总成 `bench_summary.md` 对比表 |

## 0. 定位（保持诚实）

- IFEval-lite 是**自实现轻量版**：约束类型对齐 IFEval 官方常见形态（关键词 / JSON / 长度 / 段落数 / 起止 / 禁逗号 / 句数），但题集只有 50 条中文、且为规则自动判分；
- **分数只用于 base vs SFT vs GRPO 的横向对比，不代表官方 IFEval 榜单成绩**；
- GSM8K 用官方 test 集前 N 题 + 贪心解码，分数与常见公开数字同量级但非严格复现（prompt 模板不同）。

## 1. 一键运行（在实例上执行）

```bash
cd /root/autodl-tmp/qwen_grpo/bench      # 或你同步脚本的目录
DRY=1 bash run_bench.sh                   # 先干跑：只打印命令 + 检查路径，不加载模型
bash run_bench.sh                         # 正式跑：3 模型 × (IFEval-50 + GSM8K-100)
```

### 常用开关

```bash
bash run_bench.sh ifeval        # 只跑 IFEval-lite
bash run_bench.sh gsm8k         # 只跑 GSM8K
GSM8K_NUM=200 bash run_bench.sh # GSM8K 题数（默认 100）
DRY=1 bash run_bench.sh         # 干跑
BASE_OVERRIDE=/root/autodl-tmp/qwen_sft/models/Qwen3-4B-Instruct bash run_bench.sh
OUTDIR=/root/autodl-tmp/bench_out bash run_bench.sh   # 换结果目录
PYTHON=/root/miniconda3/bin/python bash run_bench.sh  # 指定解释器
```

**先干跑再正式跑**：`DRY=1` 会检查底座/adapter 路径是否存在并打印真实命令，避免白等一次 8GB 模型加载。

### 脚本里的模型配置

`run_bench.sh` 顶部一段集中配置，实例上路径变了只改这里：

```bash
BASE_QWEN=${BASE_OVERRIDE:-$SFT/models/Qwen3-4B}   # 底座
ADAPTER_SFT=$SFT/out/e1_qwen3-4b_lora              # Qwen3-4B SFT(LoRA, e1)
ADAPTER_GRPO=$GRPO/out/grpo_qwen3-4b_formal        # GRPO 250 步(通用 RM)

MODELS=( "qwen3-4b_base|" "qwen3-4b_sft|$ADAPTER_SFT" "qwen3-4b_grpo|$ADAPTER_GRPO" )
```

- base 与 SFT/GRPO 共用同一底座，后两者只是叠加 **LoRA adapter**（避免再合并 8GB 权重）；
- SFT 那行用的是 **e1 的 LoRA**（`out/e1_qwen3-4b_lora`），等价于 `out/merged_e1_qwen3-4b_lora`；若实例上没这个 adapter，从 `MODELS` 里删掉该行即可；
- 加模型就加一行 `"tag|/path/to/adapter"`（adapter 留空 = 纯底座）。

### 底座到底是 Base 还是 Instruct？

仓库里记录的是 `/root/autodl-tmp/qwen_sft/models/Qwen3-4B`（见 `experiments/_tools/qwen_sft/configs/e1_qwen3_4b_lora.yaml`）。若实例上该目录实际是 **Instruct** 权重，base 基线会强不少（指令遵循天然更好），三个模型口径仍然一致、横向对比有效——但**报告里要写清楚是哪个**。用 `grep -i instruct <model>/config.json` 或看 `tokenizer_config.json` 里的 chat_template 可确认。

## 2. 输出产物

结果默认写到 `experiments/results/bench/`：

| 文件 | 内容 |
|---|---|
| `bench_summary.md` | **对比表**（IFEval 两级通过率 + GSM8K 正确率），直接贴进实验报告 |
| `ifeval_results.csv` / `gsm8k_results.csv` | 追加写，多模型自动成表 |
| `ifeval_results_<tag>.jsonl` / `gsm8k_results_<tag>.jsonl` | 每题明细（含模型回答片段 / ref vs pred） |
| `ifeval_<tag>.log` / `gsm8k_<tag>.log` | 每个模型完整 stdout（含进度与 GPU 信息） |

> **重复跑会追加**：`*_results.csv` 是 append 模式，同一个 tag 跑两次会在表里出现两行。
> 重跑同一批模型前先删掉旧 csv（或换 `OUTDIR`），否则汇总表会重复。
> 任一模型跑挂时脚本**不会**生成汇总表（只留各模型 log），避免半张表被误读为完整结果。

### 汇总表形态

```
| 模型 | IFEval instruction-level | IFEval prompt-level | GSM8K (greedy) |
|---|---|---|---|
| qwen3-4b_base | 70.1% (67约束) | 52.0% (50题) | 78.0% (78/100) |
| qwen3-4b_sft  | 85.0% (67约束) | 74.0% (50题) | 80.0% (80/100) |
| qwen3-4b_grpo | 88.0% (67约束) | 78.0% (50题) | 79.0% (79/100) |
```

（上表数字是**格式示例**，不是真实结果；真实数字以跑完的 csv 为准。）

## 3. 时长与显存

- IFEval-lite：50 题 × `max_new_tokens=600` 贪心解码，**每题 ~1-3s**（4B + 单卡）→ 3 模型约 5-10 分钟；
- GSM8K-100：推理链更长，**每题 ~2-5s** → 3 模型约 10-20 分钟；
- 显存：仅推理，bf16 4B ≈ **10-12GB**，一张 24GB 卡足够（不需要之前 GRPO 训练用的 96GB 卡）；

## 4. 已知边界 / 注意

1. **【已修】长度类约束曾对中文完全失效**（v1 缺陷，务必知道）：
   v1 用 `len(text.split())` 计长度。中文不用空格分词，于是 —— 实测 base 回答"至少 100 字"的题目写出了 **127 个汉字**，
   被 `split()` 算成 **1 个词**：
   - `min_words` 对中文**恒不通过**（实测 base/SFT 都是 0%(0/5)）；
   - `max_words` 对中文**恒通过**（实测都是 100%(5/5)）。
   53 个约束里有 10 个（19%）当时在测"空"东西。**这不是模型能力差异，是判分口径 bug。**
   现在改用 `count_units()`：CJK 汉字各算 1 个单位，连续西文/数字算 1 个词（对纯中文 ≈ 汉字数，对齐题面"N 字"）。
   ⚠️ 因此 **v1 与修正版的 IFEval 分数不可直接比较**，`min_words`/`max_words` 列的差异尤其不能当真；
   要么整体用修正版重跑，要么只用修正版之后的结果做横向对比。
2. **【已修】结尾类约束曾误判 5/6**（v2 缺陷，务必知道）：
   `ck_end_with` 原来是 `text.strip().endswith(chars)`，过严 —— 实测模型输出
   `"……未来。此致敬礼。"` 因结尾多一个句号被判 **False**；`"……成功。坚持就是胜利。"` 同理。
   3 道题（#14/#24/#48）因此呈现"三个模型全挂"，其中 **5/6 是误判**（模型其实照做了）。
   现在改为：先严格比对，不通过再剥掉尾部标点/空白/引号/markdown 符后比对。
   影响：base IFEval 92.5% → **98.1%**，SFT 86.8% → 88.7%，GRPO 92.5% → 96.2%。
   `end_with` 从"全员 33-50%"变成真正的区分项（base 100%、SFT/GRPO 67%）。
3. `ifeval_lite.py` 第 41 条（"不超过120字总结这篇文章"）**没给原文**，模型正常会反问，该题几乎必挂——所有模型同等受影响，属题面缺陷；
4. `no_commas` 只查中英文逗号，不查顿号/分号（题面也只说"不用逗号"）；
5. **思考模式**：Qwen3 chat template 默认开思考，实测 base/GRPO 会把 600 token 预算耗在思考段、答案被截断而答错。
   统一用 `--no-think`（`run_bench.sh` 默认已加）；若要看思考模式表现，用 `THINK=1 bash run_bench.sh`；
6. `gsm8k_eval.py` 统一走 chat template —— SFT 是 `template: qwen`、GRPO 是 `apply_chat_template` 训的，
   给它们喂纯文本提示会系统性低估这两个模型；
7. GSM8K 首次运行需要联网下 `openai/gsm8k`，脚本已设 `HF_ENDPOINT=https://hf-mirror.com`（可用环境变量覆盖）。

> **教训**：两个缺陷都表现为"某个约束类型通过率异常低/异常高"。
> 所以拿到结果先跑 `analyze_constraints.py` 看**按约束类型的分布**——
> 某类约束三个模型齐刷刷 0%/100%，先怀疑判分器，再怀疑模型。

## 4.1 约束类型分析

`analyze_constraints.py` 把"哪个模型更听话"细化成"哪一类约束更不会被遵守"：

```bash
python analyze_constraints.py --dir <结果目录>      # -> <结果目录>/ANALYSIS.md
```

它会按约束类型统计通过率，并列出**所有模型都过不去**的题（优先怀疑题面/判分，而不是模型）。
正是这个分析暴露了上面第 1 条的 `min_words` 缺陷（0% vs 其它类型 100% 的反常分布）。
它用 `ast` 静态解析 `ifeval_lite.py` 里的 `PROMPTS`，**不导入 torch**，所以在没装 torch 的机器上也能跑。

## 5. 单模型手动调用

```bash
python ifeval_lite.py --model /path/to/base [--adapter /path/to/lora] \
    --tag grpo_after --out ifeval_results.csv [--max-new-tokens 600]

python gsm8k_eval.py --model /path/to/base [--adapter /path/to/lora] \
    --num 100 --tag grpo_250steps --out gsm8k_results.csv
```

## 6. GSM8K 相关的后续实验（RLVR）

见本文件下方的"效果演示"思路：用 GSM8K 训练集抽 ~500 题，reward = 答案精确匹配（0/1）+ 格式小奖励，GRPO 训 100~200 步，再复评同一 100 题 → 对比 base / 通用 GRPO / **GSM8K-RLVR** 三行，最后一行上升即训练效果可视化。

| 模型 | GSM8K 预期 | 说明 |
|---|---|---|
| MiniMind 64M 系列 | ~0-3% | 容量不足，仅作尺度阶梯展示 |
| Qwen2.5-1.5B-Instruct | ~50-60% | |
| Qwen3-4B（Instruct） | ~75-82% | 基线较强 |
| Qwen3-4B + GRPO(通用 RM, 250 步) | ≈基线 | 没训数学，预期无明显提升（对照） |
| **Qwen3-4B + RLVR(GSM8K 规则奖励)** | **应有明显提升** | 真正的"训练效果"实验 |

> GSM8K 公开数据（`openai/gsm8k`，HF 直连需代理或 `HF_ENDPOINT` 镜像）约 7.5k train / 1.3k test。
