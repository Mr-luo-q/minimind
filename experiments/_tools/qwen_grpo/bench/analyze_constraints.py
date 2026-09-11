#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IFEval-lite 失败的约束类型分析：把"哪个模型更听话"细化成"哪一类约束更不会被遵守"。

读 IFEval-lite 的每题明细 jsonl，按约束类型（json / keywords / max_words ...）统计
每个模型的通过率，并找出"三个模型都过不去"的题（多半是题面/判分问题，不是模型问题）。

用法:
  python analyze_constraints.py --dir /root/autodl-tmp/qwen_grpo/bench/results
  python analyze_constraints.py --dir . --out ANALYSIS.md
"""
import argparse
import json
import os
import sys
from collections import defaultdict

# 与 ifeval_lite.py 保持一致的题面-约束映射；优先从 ifeval_lite 导入，导入不了再用内置副本
def safe_print(s):
    try:
        print(s)
    except UnicodeEncodeError:
        enc = sys.stdout.encoding or "utf-8"
        print(s.encode(enc, errors="replace").decode(enc))


def load_prompts():
    """从 ifeval_lite.py 里取 PROMPTS。

    用 ast 静态解析源码，而不是 import —— ifeval_lite.py 顶层 import torch，
    在没装 torch 的机器上导入会直接失败。PROMPTS 是纯字面量，静态解析更稳，
    也不会因为导入而触发任何副作用。
    """
    here = os.path.dirname(os.path.abspath(__file__))
    cands = [os.path.join(here, "ifeval_lite.py"),
             os.path.join(here, "..", "..", "experiments", "_tools", "qwen_grpo", "bench", "ifeval_lite.py"),
             os.path.join(here, "..", "_tools", "qwen_grpo", "bench", "ifeval_lite.py")]
    for p in cands:
        p = os.path.abspath(p)
        if not os.path.exists(p):
            continue
        try:
            import ast
            tree = ast.parse(open(p, encoding="utf-8").read())
            for node in tree.body:
                if isinstance(node, ast.Assign):
                    for tgt in node.targets:
                        if isinstance(tgt, ast.Name) and tgt.id == "PROMPTS":
                            return ast.literal_eval(node.value)
            safe_print(f"!! {p} 里没找到 PROMPTS")
        except Exception as e:
            safe_print(f"!! 解析 {p} 失败: {e}")
    return None


MODELS = [("qwen3-4b_base", "base"), ("qwen3-4b_sft", "SFT"), ("qwen3-4b_grpo", "GRPO")]


def read_jsonl(p):
    if not os.path.exists(p):
        return []
    out = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", default=None, help="输出 markdown 路径（默认 <dir>/ANALYSIS.md）")
    args = ap.parse_args()
    d = args.dir

    PROMPTS = load_prompts()
    if not PROMPTS:
        safe_print("!! 找不到 ifeval_lite.py（需要它的 PROMPTS 做 题面->约束 映射）")
        return 1

    # 题型分布先行：哪些约束被考了多少次
    type_total = defaultdict(int)
    for _, checks in PROMPTS:
        for kind, _p in checks:
            type_total[kind] += 1

    per_model = {}
    for tag, label in MODELS:
        det = read_jsonl(os.path.join(d, f"ifeval_results_{tag}.jsonl"))
        if not det:
            continue
        by_type = defaultdict(lambda: [0, 0])   # kind -> [pass, total]
        failed_prompts = []
        for row in det:
            i = row.get("i")
            oks = row.get("checks_ok", [])
            if i is None or i >= len(PROMPTS):
                continue
            checks = PROMPTS[i][1]
            if len(oks) != len(checks):
                continue          # 明细与定义不一致（题集改过）时跳过，不猜
            allok = True
            for (kind, _p), ok in zip(checks, oks):
                by_type[kind][1] += 1
                by_type[kind][0] += int(bool(ok))
                if not ok:
                    allok = False
            if not allok:
                failed_prompts.append(i)
        per_model[tag] = {"label": label, "by_type": by_type,
                          "failed": set(failed_prompts), "n": len(det)}

    L = ["# IFEval-lite 约束类型分析", "",
         "> 把\"哪个模型更听话\"细化为\"哪一类约束更不会被遵守\"。",
         "> 约束定义与 `ifeval_lite.py` 的 PROMPTS 一一对应，明细条数与定义不一致的题会被跳过（不猜）。", ""]

    kinds = sorted(type_total, key=lambda k: (-type_total[k], k))
    L += ["## 各约束类型通过率", "",
          "| 约束类型 | 题数 | " + " | ".join(lbl for _t, lbl in MODELS if _t in per_model) + " |",
          "|---" * (2 + sum(1 for t, _ in MODELS if t in per_model)) + "|"]
    for k in kinds:
        row = [f"`{k}`", str(type_total[k])]
        for tag, _lbl in MODELS:
            if tag not in per_model:
                continue
            p, t = per_model[tag]["by_type"].get(k, [0, 0])
            row.append("—" if t == 0 else f"{p/t*100:.0f}% ({p}/{t})")
        L.append("| " + " | ".join(row) + " |")

    # 全员都挂的题 -> 多半是题面缺陷
    if per_model:
        common = None
        for tag in per_model:
            common = per_model[tag]["failed"] if common is None else (common & per_model[tag]["failed"])
        L += ["", "## 所有模型都未通过的题（优先怀疑题面/判分，而非模型）", ""]
        if common:
            for i in sorted(common):
                L.append(f"- **#{i}** {PROMPTS[i][0]}")
                L.append(f"  - 约束：`{PROMPTS[i][1]}`")
        else:
            L.append("（无）")

    L += ["", "## 各模型未通过的题号", ""]
    for tag, label in MODELS:
        if tag not in per_model:
            continue
        L.append(f"- **{label}**（{len(per_model[tag]['failed'])}/{per_model[tag]['n']} 题有约束未过）："
                 + (", ".join(f"#{i}" for i in sorted(per_model[tag]["failed"])) or "无"))

    md = "\n".join(L) + "\n"
    out = args.out or os.path.join(d, "ANALYSIS.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write(md)
    safe_print(md)
    safe_print(f"[analyze] -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
