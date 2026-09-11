#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
把两次跑批合并成最终结果集：

  IFEval  <- v2 跑批（results_v2/<tag>/，count_units 修正后的长度判分）
  GSM8K   <- v1 跑批（长度判分只影响 IFEval，GSM8K 判分逻辑没变，无需重跑）

合并成 gen_table.py 能直接读的扁平目录，并写一份说明来源的 manifest.json。

用法:
  python merge_results.py --ifeval-v2 <results_v2目录> --gsm8k-v1 <v1结果目录> --out <最终目录>
"""
import argparse
import csv
import glob
import json
import os
import shutil
import sys
import time

TAGS = ["qwen3-4b_base", "qwen3-4b_sft", "qwen3-4b_grpo"]


def safe_print(s):
    try:
        print(s)
    except UnicodeEncodeError:
        enc = sys.stdout.encoding or "utf-8"
        print(s.encode(enc, errors="replace").decode(enc))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ifeval-v2", required=True, help="results_v2 目录（每个 tag 一个子目录）")
    ap.add_argument("--gsm8k-v1", required=True, help="v1 结果目录（含 gsm8k_results*.csv/jsonl）")
    ap.add_argument("--out", required=True, help="合并后的扁平输出目录")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    copied_i, copied_g = [], []

    # ---- IFEval: 从 v2 各子目录合并成一个 csv（每模型一行）+ 逐题 jsonl ----
    rows, header = [], None
    for tag in TAGS:
        p = os.path.join(args.ifeval_v2, tag, "ifeval_results.csv")
        if not os.path.exists(p):
            safe_print(f"!! 缺 v2 IFEval: {p}")
            continue
        with open(p, encoding="utf-8") as f:
            r = list(csv.reader(f))
        if not r:
            continue
        header = header or r[0]
        rows += [x for x in r[1:] if x and x[0] == tag]
        src = os.path.join(args.ifeval_v2, tag, f"ifeval_results_{tag}.jsonl")
        if os.path.exists(src):
            shutil.copy(src, os.path.join(args.out, os.path.basename(src)))
            copied_i.append(tag)
            safe_print(f"  IFEval v2  {tag}: {r[1][1] if len(r) > 1 else '?'} / {r[1][2] if len(r) > 1 else '?'}")
    if header and rows:
        with open(os.path.join(args.out, "ifeval_results.csv"), "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(rows)
        safe_print(f"  合并 IFEval csv: {len(rows)} 行 -> {args.out}/ifeval_results.csv")

    # ---- GSM8K: 从 v1 直接搬 ----
    for pat, dst in [("gsm8k_results_*.jsonl", None), ("gsm8k_results.csv", None),
                     ("gsm8k_*.log", None)]:
        for src in glob.glob(os.path.join(args.gsm8k_v1, pat)):
            shutil.copy(src, os.path.join(args.out, os.path.basename(src)))
            copied_g.append(os.path.basename(src))
    for tag in TAGS:
        if os.path.exists(os.path.join(args.out, f"gsm8k_results_{tag}.jsonl")):
            safe_print(f"  GSM8K v1   {tag}: 已搬入")

    manifest = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "note": "最终结果集：IFEval 用修正版判分器（长度 count_units + 结尾容错），GSM8K 沿用首次跑批",
        "ifeval_checker": "count_units (CJK 逐字计) + end_with 容许尾部标点/引号/markdown 符",
        "ifeval_source": args.ifeval_v2 + "（修正后重跑）",
        "gsm8k_source": "首次跑批（两处修正都只影响 IFEval，GSM8K 判分逻辑未变）",
        "fixes": [
            "旧版长度判分用 text.split()，中文无空格 → 127 汉字被算成 1 个词: min_words 恒 False、max_words 恒 True",
            "旧版结尾判分过严，'……此致敬礼。' 因多一个句号被判 False（3 题三模型全挂，其中 5/6 是误判）",
        ],
        "chat_template": "yes (gsm8k 也走 chat template)",
        "no_think": True,
        "gsm8k_num": 100,
    }
    with open(os.path.join(args.out, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    safe_print(f"\nmanifest 写入 {args.out}/manifest.json")
    safe_print(f"IFEval 模型 {len(copied_i)}/3, GSM8K 文件 {len(copied_g)} 个")
    return 0 if len(copied_i) == 3 else 1


if __name__ == "__main__":
    raise SystemExit(main())
