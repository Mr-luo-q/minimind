#!/bin/bash
# ============================================================
# 一键评测：base vs SFT vs GRPO（IFEval-lite + GSM8K）
#   - 在【实例】上执行，不是本地
#   - 结果：
#       experiments/results/bench/ifeval_results.csv   (+ 每模型 jsonl 明细)
#       experiments/results/bench/gsm8k_results.csv    (+ 每模型 jsonl 明细)
#       experiments/results/bench/bench_summary.md     (对比表，直接贴进报告)
#   - 评测脚本已随本仓库同步到实例，不再依赖 /root/autodl-tmp 下的副本
#
# 用法:
#   bash run_bench.sh                 # 跑全部 3 个模型 × 2 个 benchmark
#   bash run_bench.sh ifeval          # 只跑 IFEval-lite
#   bash run_bench.sh gsm8k           # 只跑 GSM8K
#   GSM8K_NUM=200 bash run_bench.sh   # GSM8K 题数(默认 100)
#   DRY=1 bash run_bench.sh           # 只打印将要执行的命令，不加载模型
#   BASE_OVERRIDE=/path/to/model bash run_bench.sh   # 换底座(例如 Instruct 版)
#   PYTHON=/path/to/python bash run_bench.sh         # 指定解释器
#   OUTDIR=/path/to/out bash run_bench.sh            # 换结果目录
#   THINK=1 bash run_bench.sh                        # 保留 Qwen3 默认思考模式(默认已关，见下)
#
# 注1: 默认给两个评测器都加 --no-think。原因(实测): Qwen3 chat template 默认开启思考模式，
#      base 与 GRPO 在 GSM8K 首题会把 600 token 预算耗在思考段上、答案被截断而答错；
#      三个模型统一在非思考模式作答，口径才一致。THINK=1 可还原官方默认行为。
# 注2: SFT 那一行用 e1 的 LoRA adapter 叠加 base（等价于 merged_e1_qwen3-4b_lora，
#      省去 8GB 权重合并）。若实例上没有这个 adapter，把 MODELS 里 sft 那行删掉即可。
# ============================================================
set -uo pipefail

export PATH=/root/miniconda3/bin:$PATH
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}   # datasets 下 GSM8K 用
export HF_HUB_DISABLE_TELEMETRY=1

WHAT="${1:-all}"
GSM8K_NUM="${GSM8K_NUM:-100}"
# 本次跑批的备注（写进 manifest.json，便于日后区分口径；例: NOTE="v2 长度判分修正后重跑"）
NOTE="${NOTE:-}"

# 默认关闭思考模式；THINK=1 可关掉这个开关（即保留 Qwen3 官方默认行为）
THINK_FLAG="--no-think"
[ "${THINK:-0}" = "1" ] && THINK_FLAG=""

# ---------------- 路径配置（实例上路径变了只改这一段） ----------------
SFT=/root/autodl-tmp/qwen_sft
GRPO=/root/autodl-tmp/qwen_grpo
BASE_QWEN=${BASE_OVERRIDE:-$SFT/models/Qwen3-4B}
ADAPTER_SFT=$SFT/out/e1_qwen3-4b_lora
ADAPTER_GRPO=$GRPO/out/grpo_qwen3-4b_formal

# 评测脚本位置: 本仓库 experiments/_tools/qwen_grpo/bench
# 允许从任意 cwd 调用本脚本
BENCH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# 结果落到实验结果目录 experiments/results/bench/
OUTDIR="${OUTDIR:-$(cd "$BENCH/../../.." && pwd)/results/bench}"
mkdir -p "$OUTDIR"
# ---------------- 路径配置结束 ----------------

# 模型清单:  tag|adapter(可为空)
MODELS=(
  "qwen3-4b_base|"
  "qwen3-4b_sft|$ADAPTER_SFT"
  "qwen3-4b_grpo|$ADAPTER_GRPO"
)

# python: 优先 $SFT/venv（llamafactory 那套, 已含 transformers+peft+datasets）
if [ -n "${PYTHON:-}" ]; then
  B="$PYTHON"
elif [ -x "$SFT/venv/bin/python" ]; then
  B="$SFT/venv/bin/python"
else
  B="$(command -v python3 || command -v python)"
fi

echo "==================== ENV ===================="
echo "bench dir   : $BENCH"
echo "out dir     : $OUTDIR"
echo "python      : $B"
echo "which       : $WHAT     gsm8k_num: $GSM8K_NUM"
echo "base        : $BASE_QWEN"
echo "adapter sft : $ADAPTER_SFT"
echo "adapter grpo: $ADAPTER_GRPO"
echo "============================================="

# 环境探针（失败只提示，不中断——真跑的时候错误信息更完整）
"$B" -c "import torch,transformers,peft,datasets as d;print('torch',torch.__version__,'| transformers',transformers.__version__,'| datasets ok');print('cuda',torch.cuda.is_available(),'|',torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU ONLY')" 2>&1 || echo "!! 环境探针失败：请看上面的报错（缺包 / 无 CUDA / python 路径不对）"
echo "============================================="

# 路径预检: 缺一个就提示（不中断，方便一次看全）
require() { [ -e "$2" ] || { echo "!! 缺失 $1: $2"; NBAD=$((NBAD + 1)); }; return 0; }
NBAD=0
require "base 底座"    "$BASE_QWEN"
require "SFT adapter"  "$ADAPTER_SFT"
require "GRPO adapter" "$ADAPTER_GRPO"
if [ "$NBAD" != "0" ]; then
  echo ">> 有 $NBAD 个路径不存在。缺失的 adapter 会在加载时报错并中断该模型那一行；"
  echo ">> 先改好顶部路径配置，再用 DRY=1 复查一遍。"
fi

FAILED=0   # 失败计数（在函数使用前初始化，set -u 下必须）

# ---------------- 口径清单 ----------------
# 把"这批分数是在什么口径下跑出来的"落成文件，避免日后分不清 v1/v2 口径而混用分数。
MANIFEST="$OUTDIR/manifest.json"
if [ ! -f "$MANIFEST" ]; then
  cat > "$MANIFEST" <<EOF
{
  "created": "$(date '+%Y-%m-%d %H:%M:%S')",
  "note": "${NOTE}",
  "ifeval_checker": "count_units (CJK 逐字计; v1 的 split() 对中文恒真/恒假，已修)",
  "chat_template": "yes (gsm8k 也改走 chat template)",
  "no_think": $([ -n "$THINK_FLAG" ] && echo true || echo false),
  "gsm8k_num": $GSM8K_NUM,
  "base": "$BASE_QWEN"
}
EOF
  echo "manifest 写入: $MANIFEST"
else
  echo "manifest 已存在（沿用本次口径记录）: $MANIFEST"
fi

run_ifeval() {
  local tag="$1" adapter="$2" cmd
  if [ -n "$adapter" ]; then
    cmd=("$B" "$BENCH/ifeval_lite.py" --model "$BASE_QWEN" --adapter "$adapter"
         --tag "$tag" --out "$OUTDIR/ifeval_results.csv" ${THINK_FLAG:+"$THINK_FLAG"})
  else
    cmd=("$B" "$BENCH/ifeval_lite.py" --model "$BASE_QWEN"
         --tag "$tag" --out "$OUTDIR/ifeval_results.csv" ${THINK_FLAG:+"$THINK_FLAG"})
  fi
  echo; echo "---------- IFEval-lite: $tag ----------"
  printf '  %q' "${cmd[@]}"; echo
  [ "${DRY:-0}" = "1" ] && return 0
  "${cmd[@]}" 2>&1 | tee "$OUTDIR/ifeval_$tag.log"
  if [ "${PIPESTATUS[0]}" = "0" ]; then
    echo "EXIT=0  OK"
  else
    echo "EXIT=${PIPESTATUS[0]}  !! 这一行失败，最终汇总会标为不完整"
    FAILED=$((FAILED + 1))
  fi
}

run_gsm8k() {
  local tag="$1" adapter="$2" cmd
  if [ -n "$adapter" ]; then
    cmd=("$B" "$BENCH/gsm8k_eval.py" --model "$BASE_QWEN" --adapter "$adapter"
         --num "$GSM8K_NUM" --tag "$tag" --out "$OUTDIR/gsm8k_results.csv" ${THINK_FLAG:+"$THINK_FLAG"})
  else
    cmd=("$B" "$BENCH/gsm8k_eval.py" --model "$BASE_QWEN"
         --num "$GSM8K_NUM" --tag "$tag" --out "$OUTDIR/gsm8k_results.csv" ${THINK_FLAG:+"$THINK_FLAG"})
  fi
  echo; echo "---------- GSM8K($GSM8K_NUM): $tag ----------"
  printf '  %q' "${cmd[@]}"; echo
  [ "${DRY:-0}" = "1" ] && return 0
  "${cmd[@]}" 2>&1 | tee "$OUTDIR/gsm8k_$tag.log"
  if [ "${PIPESTATUS[0]}" = "0" ]; then
    echo "EXIT=0  OK"
  else
    echo "EXIT=${PIPESTATUS[0]}  !! 这一行失败，最终汇总会标为不完整"
    FAILED=$((FAILED + 1))
  fi
}

FAILED=0
for entry in "${MODELS[@]}"; do
  tag="${entry%%|*}"
  adapter="${entry#*|}"
  if [ "$WHAT" = "all" ] || [ "$WHAT" = "ifeval" ]; then run_ifeval "$tag" "$adapter"; fi
  if [ "$WHAT" = "all" ] || [ "$WHAT" = "gsm8k" ]; then run_gsm8k  "$tag" "$adapter"; fi
done

if [ "${DRY:-0}" = "1" ]; then
  echo; echo "DRY=1: 上面是全部将要执行的真实命令，未加载任何模型。"
  exit 0
fi

if [ "$FAILED" != "0" ]; then
  echo; echo "!! 有 $FAILED 次运行失败，因此【不生成汇总表】——避免半张表被误读为完整结果。"
  echo "!! 失败的完整报错在各模型的 *_<tag>.log 里；修好后重跑（已成功的模型该跑还是要重跑）。"
  exit 1
fi

# ---------------- 汇总成 markdown 对比表 ----------------
echo; echo "==================== SUMMARY ===================="
OUTDIR="$OUTDIR" "$B" - <<'PY'
import csv, os
out = os.environ["OUTDIR"]

def read_csv(name):
    p = os.path.join(out, name)
    if not os.path.exists(p):
        return []
    with open(p, encoding="utf-8") as f:
        return list(csv.DictReader(f))

if_rows = read_csv("ifeval_results.csv")
gs_rows = read_csv("gsm8k_results.csv")
if_map = {r["tag"]: r for r in if_rows}
gs_map = {r["tag"]: r for r in gs_rows}
tags = [t for t in ["qwen3-4b_base", "qwen3-4b_sft", "qwen3-4b_grpo"] if t in if_map or t in gs_map]
tags += [t for t in list(if_map) + list(gs_map) if t not in tags]

lines = ["# base vs SFT vs GRPO 评测对比", "",
         "> IFEval-lite 为自实现轻量版（50 条中文指令 + 规则判分，10 类约束），",
         "> 分数用于**模型间横向对比**，不代表官方 IFEval 榜单成绩。", "",
         "| 模型 | IFEval instruction-level | IFEval prompt-level | GSM8K (greedy) |",
         "|---|---|---|---|"]
for t in tags:
    i = if_map.get(t); g = gs_map.get(t)
    ic = f"{float(i['instruction_acc'])*100:.1f}% ({i['constraints']}约束)" if i else "—"
    pc = f"{float(i['prompt_acc'])*100:.1f}% ({i['prompts']}题)" if i else "—"
    gc = f"{float(g['accuracy'])*100:.1f}% ({g['correct']}/{g['total']})" if g else "—"
    lines.append(f"| {t} | {ic} | {pc} | {gc} |")
lines += ["", "## 原始产物", "",
          "- `ifeval_results.csv` / `gsm8k_results.csv`（追加写，多模型自动成表）",
          "- `ifeval_results_<tag>.jsonl` / `gsm8k_results_<tag>.jsonl`（每题明细）",
          "- `ifeval_<tag>.log` / `gsm8k_<tag>.log`（完整 stdout）"]
md = "\n".join(lines) + "\n"
with open(os.path.join(out, "bench_summary.md"), "w", encoding="utf-8") as f:
    f.write(md)
print(md)
PY

echo "DONE -> $OUTDIR/bench_summary.md"
