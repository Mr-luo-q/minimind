# -*- coding: utf-8 -*-
"""
GSM8K 评测器：自动判分（规则提取最终答案比对），可评测任意 transformers checkpoint 或 base+LoRA。

设计意图:
  1) 对比 SFT/GRPO 前后数学正确率 (体现训练效果)
  2) extract_answer() 同时可复用为 GRPO 的规则奖励 (RLVR) 判分器

用法(实例, base python):
  python gsm8k_eval.py --model /path/to/base --num 100 \
      --adapter /path/to/grpo_adapter (可选) --tag grpo_250steps \
      --out gsm8k_results.csv

说明: 默认零样本 + 贪心解码; 参考答案取 "#### 后数字", 预测取生成文本最后一个数字。
"""
import argparse
import csv
import json
import os
import re
import time

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def extract_reference_answer(text):
    """参考答案: '... #### 123' -> '123'"""
    m = re.search(r"####\s*([-+]?\d+(?:,\d{3})*(?:\.\d+)?)", text)
    if m:
        return m.group(1).replace(",", "")
    nums = re.findall(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?", text)
    return nums[-1].replace(",", "") if nums else None


def extract_predicted_answer(text):
    """预测答案: 取生成文本里最后一个数字(去掉逗号/百分号场景)"""
    text = re.sub(r"[,，]", "", text)
    m = re.findall(r"-?\d+(?:\.\d+)?", text)
    return m[-1] if m else None


def normalize(s):
    if s is None:
        return None
    try:
        f = float(s)
        if f == int(f):
            return str(int(f))
        return f"{f:.6f}".rstrip("0").rstrip(".")
    except ValueError:
        return s.strip()


def build_prompt(question):
    # 零样本中文提示，引导逐步推理（非 chat 模型的兜底纯文本格式）
    return f"请解答下面的数学应用题，先逐步推理，最后单独一行给出最终答案。\n问题：{question}\n答案是："


def build_input(tok, question, no_think=False):
    """优先用 chat template —— SFT 是 template: qwen、GRPO 也是 apply_chat_template 训的，
    给它们喂纯文本提示会低估这两个模型（口径不公平）。没有 chat template 才退回纯文本。"""
    body = f"请解答下面的数学应用题，先逐步推理，最后单独一行给出最终答案。\n问题：{question}"
    try:
        return tok.apply_chat_template([{"role": "user", "content": body}],
                                       tokenize=False, add_generation_prompt=True,
                                       **({"enable_thinking": False} if no_think else {}))
    except Exception:
        return build_prompt(question)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="transformers 模型目录")
    ap.add_argument("--adapter", default=None, help="可选: peft LoRA adapter 目录(叠加在 base 上)")
    ap.add_argument("--num", type=int, default=100, help="评测题目数(默认抽前100条, -1=全部1319)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-new-tokens", type=int, default=600)
    ap.add_argument("--no-think", action="store_true",
                    help="给 chat template 传 enable_thinking=False（Qwen3 专用；否则 600 token 会被思考段吃光）")
    ap.add_argument("--tag", default="model")
    ap.add_argument("--out", default="gsm8k_results.csv")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("gpu:", torch.cuda.get_device_name(0) if dev == "cuda" else "cpu")

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map=dev, trust_remote_code=True)
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
        print("adapter loaded:", args.adapter)
    model.eval()
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    ds = load_dataset("openai/gsm8k", "main", split="test")
    print("gsm8k test size:", len(ds))
    if args.num and args.num > 0:
        ds = ds.select(range(min(args.num, len(ds))))

    correct = 0
    rows = []
    t0 = time.time()
    for i, ex in enumerate(ds):
        q, ref_full = ex["question"], ex["answer"]
        ref = normalize(extract_reference_answer(ref_full))
        prompt = build_input(tok, q, no_think=args.no_think)
        inp = tok(prompt, return_tensors="pt").to(dev)
        with torch.no_grad():
            gen = model.generate(
                input_ids=inp.input_ids, attention_mask=inp.attention_mask,
                max_new_tokens=args.max_new_tokens, do_sample=False,
                pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
        out_text = tok.decode(gen[0][inp.input_ids.shape[1]:], skip_special_tokens=True)
        pred = normalize(extract_predicted_answer(out_text))
        ok = (pred == ref)
        correct += int(ok)
        rows.append({"tag": args.tag, "i": i, "ok": ok,
                     "ref": ref, "pred": pred, "question": q[:80]})
        if (i + 1) % 20 == 0 or i == len(ds) - 1:
            print(f"[{i+1}/{len(ds)}] acc={correct/(i+1):.3f} ({time.time()-t0:.0f}s)", flush=True)

    acc = correct / len(ds)
    print(f"\n===== RESULT {args.tag}: {acc*100:.1f}% ({correct}/{len(ds)}) =====")
    new = not os.path.exists(args.out)
    with open(args.out, "a", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["tag", "correct", "total", "accuracy"])
        w.writerow([args.tag, correct, len(ds), f"{acc:.4f}"])
    with open(args.out.replace(".csv", f"_{args.tag}.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
