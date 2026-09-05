# -*- coding: utf-8 -*-
"""
从 sharegpt 格式指令集抽取 prompt，生成 GRPO 训练数据集（jsonl: {"prompt": "..."}）。
用法:
  python build_prompts.py \
      --input  /root/autodl-tmp/qwen_sft/data/sft_10k.jsonl \
      --output /root/autodl-tmp/qwen_grpo/data/prompts_2k.jsonl \
      --num 2000 --seed 42
"""
import argparse
import json
import random


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--num", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    prompts = []
    with open(args.input, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            convs = obj.get("conversations", [])
            if not convs or convs[0].get("from") != "human":
                continue
            text = (convs[0].get("value") or "").strip()
            # 只要"像指令"的短问题，过滤超长/对话式杂讯
            if 4 <= len(text) <= 400:
                prompts.append(text)
    print("usable prompts:", len(prompts))
    random.shuffle(prompts)
    prompts = prompts[: args.num]
    with open(args.output, "w", encoding="utf-8") as f:
        for p in prompts:
            f.write(json.dumps({"prompt": p}, ensure_ascii=False) + "\n")
    print("written:", len(prompts), "->", args.output)


if __name__ == "__main__":
    main()
