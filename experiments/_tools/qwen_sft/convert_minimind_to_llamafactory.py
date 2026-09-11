# -*- coding: utf-8 -*-
"""
MiniMind sft_t2t_mini.jsonl -> LLaMA-Factory sharegpt 格式转换 + 固定采样。

用法:
  python convert_minimind_to_llamafactory.py \
      --input /root/autodl-tmp/minimind/dataset/sft_t2t_mini.jsonl \
      --output /root/autodl-tmp/qwen_sft/data/sft_10k.jsonl \
      --num 10000 --seed 42

规则:
  - 只保留纯 user/assistant 对话; 含 tools / tool_calls / reasoning_content 的样本丢弃
    (避免 Qwen 模板与 MiniMind 扩展字段不兼容, 且保证全参/LoRA 用同一份干净数据)
  - 固定 seed 采样, 保证 64M/1.5B/4B 三档实验用同一批数据(可比性)
"""
import argparse
import json
import random


def convert(input_path, output_path, num, seed):
    random.seed(seed)
    kept, dropped = 0, 0
    with open(input_path, encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            convs = obj.get("conversations", [])
            # 丢弃含工具/思考扩展字段的样本
            if any(m.get("tools") or m.get("tool_calls") or m.get("reasoning_content")
                   for m in convs):
                dropped += 1
                continue
            turns = []
            ok = True
            for m in convs:
                role = m.get("role")
                content = m.get("content")
                if not content:
                    ok = False
                    break
                if role == "user":
                    turns.append({"from": "human", "value": content})
                elif role == "assistant":
                    turns.append({"from": "gpt", "value": content})
                elif role == "system":
                    continue  # LLaMA-Factory 用 dataset_info 的 system_prompt 或忽略
                else:
                    ok = False
                    break
            if not ok or not turns:
                dropped += 1
                continue
            # 过滤掉只剩 user 没有 assistant 的样本
            if not any(t["from"] == "gpt" for t in turns):
                dropped += 1
                continue
            kept += 1
            fout.write(json.dumps({"conversations": turns}, ensure_ascii=False) + "\n")
    print(f"kept={kept} dropped={dropped}")
    # 固定采样 num 条
    with open(output_path, encoding="utf-8") as f:
        lines = f.readlines()
    print(f"total usable={len(lines)}")
    if len(lines) > num:
        random.shuffle(lines)
        lines = lines[:num]
    with open(output_path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    print(f"final={len(lines)} -> {output_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--num", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    convert(args.input, args.output, args.num, args.seed)
