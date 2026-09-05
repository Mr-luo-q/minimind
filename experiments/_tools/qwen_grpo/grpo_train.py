# -*- coding: utf-8 -*-
"""
Qwen3-4B LoRA + GRPO 训练（trl 1.12 GRPOTrainer, 单卡 96GB 验证版）
奖励函数: InternLM2-1.8B-Reward judge（RLAIF 路线; 智能家居终版可换规则奖励 RLVR）

用法(实例 /root/autodl-tmp/qwen_grpo/):
  python grpo_train.py \
      --model /root/autodl-tmp/qwen_sft/models/Qwen3-4B \
      --data  /root/autodl-tmp/qwen_grpo/data/prompts_2k.jsonl \
      --rm    /root/autodl-tmp/internlm2-1_8b-reward \
      --num-generations 4 --max-completion-length 384 \
      --max-steps 12 --output ./out/grpo_qwen3-4b_lora
"""
import argparse
import math
import os
import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModel, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer


def load_reward_model(path, device="cuda"):
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    rm = AutoModel.from_pretrained(path, torch_dtype=torch.float16,
                                   trust_remote_code=True).to(device).eval()
    for p in rm.parameters():
        p.requires_grad_(False)
    return tok, rm


def make_reward_func(rm_tok, rm, device="cuda"):
    def reward_func(prompts, completions, **kwargs):
        scores = []
        with torch.no_grad():
            for prompt, comp in zip(prompts, completions):
                messages = [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": comp},
                ]
                score = rm.get_score(rm_tok, messages)
                score = max(min(score, 3.0), -3.0)
                scores.append(float(score))
        return scores
    return reward_func


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--rm", required=True)
    ap.add_argument("--num-generations", type=int, default=4)
    ap.add_argument("--max-completion-length", type=int, default=384)
    ap.add_argument("--max-steps", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--output", default="./out/grpo_qwen3-4b")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device, "| gpu:", torch.cuda.get_device_name(0) if device == "cuda" else "cpu")

    # ---- 奖励模型 ----
    rm_tok, rm = load_reward_model(args.rm, device)
    reward_func = make_reward_func(rm_tok, rm, device)
    print("reward model loaded:", args.rm)

    # ---- 数据: {"prompt": str} ----
    ds = load_dataset("json", data_files=args.data, split="train")
    print("train prompts:", len(ds), "| example:", str(ds[0]["prompt"])[:60])

    # ---- LoRA ----
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_r * 2,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )

    training_args = GRPOConfig(
        output_dir=args.output,
        run_name=os.path.basename(args.output),
        do_train=True,                      # trl 1.x 需显式开启
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,     # prompt 数
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        bf16=True,
        logging_steps=1,
        save_steps=1000,
        save_total_limit=2,
        report_to="none",
        seed=args.seed,
        gradient_checkpointing=True,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        generation_batch_size=math.lcm(args.batch_size, args.num_generations),  # trl 1.x 约束: 同时整除 batch 与 num_gen
        temperature=0.9,
        top_p=0.95,
        beta=0.04,                          # KL 惩罚系数
        log_completions=False,              # 关闭样本打印(避免 pandas/numpy 兼容问题, 纯装饰功能)
        remove_unused_columns=False,
        model_init_kwargs={
            "torch_dtype": torch.bfloat16,
            "attn_implementation": "sdpa",
            "trust_remote_code": True,
        },
    )

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    trainer = GRPOTrainer(
        model=args.model,
        reward_funcs=[reward_func],
        args=training_args,
        train_dataset=ds,
        processing_class=tok,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(args.output)
    print("GRPO_DONE")


if __name__ == "__main__":
    main()
