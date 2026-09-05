# -*- coding: utf-8 -*-
"""
对 Qwen 系列(transformers 格式)跑与 MiniMind 实验完全相同的 8 问评测。
用法:
  python eval_qwen_probe.py --model /path/to/merged_or_base --tag qwen3-4b-lora --outdir /root/autodl-tmp/qwen_sft/probes
"""
import argparse
import os
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPTS = [
    "你有什么特长？",
    "为什么天空是蓝色的",
    "请用Python写一个计算斐波那契数列的函数",
    "解释一下\"光合作用\"的基本过程",
    "如果明天下雨，我应该如何出门",
    "比较一下猫和狗作为宠物的优缺点",
    "解释什么是机器学习",
    "推荐一些中国的美食",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="模型目录(transformers 格式, 已合并或底座)")
    ap.add_argument("--tag", required=True, help="输出文件名标记")
    ap.add_argument("--outdir", default="/root/autodl-tmp/qwen_sft/probes")
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.85)
    ap.add_argument("--top_p", type=float, default=0.95)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map=dev, trust_remote_code=True
    )
    model.eval()
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    out = open(os.path.join(args.outdir, f"probe_{args.tag}.txt"), "w", encoding="utf-8")
    for i, q in enumerate(PROMPTS):
        torch.manual_seed(1000 + i)
        messages = [{"role": "user", "content": q}]
        inp = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        ids = tok(inp, return_tensors="pt").to(dev)
        t0 = time.time()
        with torch.no_grad():
            gen = model.generate(
                input_ids=ids["input_ids"], attention_mask=ids["attention_mask"],
                max_new_tokens=args.max_new_tokens, do_sample=True,
                top_p=args.top_p, temperature=args.temperature,
                pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id,
            )
        ans = tok.decode(gen[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
        out.write(f"Q{i+1}: {q}\nA{i+1}: {ans.strip()}\n\n")
        print(f"[{i+1}/{len(PROMPTS)}] {time.time()-t0:.1f}s", flush=True)
    out.close()
    print("PROBE_DONE", args.tag)


if __name__ == "__main__":
    main()
