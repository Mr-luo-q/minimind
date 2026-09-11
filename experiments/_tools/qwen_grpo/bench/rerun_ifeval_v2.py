#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v2 IFEval 重跑（长度判分修正后），单一目的：在与 v1 不同的目录里产出三模型分数，
且让每个模型的结果落在各自的子目录里，避免 append 模式互相污染。

用法（在实例上，bench 目录里）:
  python rerun_ifeval_v2.py --outroot /root/autodl-tmp/qwen_grpo/bench/results_v2

产出:
  <outroot>/<tag>/ifeval_results.csv          每模型一行
  <outroot>/<tag>/ifeval_results_<tag>.jsonl  每题明细
  <outroot>/<tag>/manifest.json               口径记录（含 count_units 说明）
"""
import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))

MODELS = [("qwen3-4b_base", None),
          ("qwen3-4b_sft", "/root/autodl-tmp/qwen_sft/out/e1_qwen3-4b_lora"),
          ("qwen3-4b_grpo", "/root/autodl-tmp/qwen_grpo/out/grpo_qwen3-4b_formal")]
BASE = "/root/autodl-tmp/qwen_sft/models/Qwen3-4B"

MANIFEST = {
    "note": "IFEval 用修正后的判分器重跑（长度 count_units + 结尾容错）",
    "ifeval_checker": "count_units (CJK 逐字计) + end_with 容许尾部标点/引号/markdown 符",
    "chat_template": "yes",
    "no_think": True,
    "gsm8k": "不重跑（这两处修正都只影响 IFEval；GSM8K 判分逻辑未变）",
    "why": "v1: split() 计长对中文恒真/恒假; v2: end_with 过严，'…此致敬礼。' 被判 False（5/6 为误判）",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outroot", required=True)
    ap.add_argument("--python", default=sys.executable)
    args = ap.parse_args()

    rc_all = 0
    for tag, adapter in MODELS:
        d = os.path.join(args.outroot, tag)
        os.makedirs(d, exist_ok=True)
        if not os.path.exists(os.path.join(d, "manifest.json")):
            with open(os.path.join(d, "manifest.json"), "w", encoding="utf-8") as f:
                json.dump(dict(MANIFEST, created=time.strftime("%Y-%m-%d %H:%M:%S"), tag=tag),
                          f, ensure_ascii=False, indent=2)
        cmd = [args.python, os.path.join(HERE, "ifeval_lite.py"), "--model", BASE,
               "--tag", tag, "--out", os.path.join(d, "ifeval_results.csv"), "--no-think"]
        if adapter:
            cmd += ["--adapter", adapter]
        print("\n" + "=" * 70)
        print("RUN", " ".join(cmd), flush=True)
        rc = subprocess.call(cmd)
        print("EXIT", rc, "for", tag, flush=True)
        if rc != 0:
            rc_all = 1
    print("\nALL DONE rc=%d" % rc_all)
    return rc_all


if __name__ == "__main__":
    raise SystemExit(main())
