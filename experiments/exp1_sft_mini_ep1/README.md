# Exp1: Full SFT on sft_t2t_mini (1 epoch)

- 日期: 2026-09-02
- 硬件: RTX 5090 32GB (AutoDL), torch 2.8.0+cu128, Python 3.12
- 脚本: trainer/train_full_sft.py (全参数微调 Full SFT, 非 LoRA)
- 命令: python train_full_sft.py --epochs 1 --save_weight sft_mini_ep1
- 起点权重: out/pretrain_768.pth (官方预训练底座, 63.91M)
- 数据: dataset/sft_t2t_mini.jsonl (905,718 条指令-回答对话)
- 关键参数: batch=16, max_seq_len=768, lr=1e-5 (cosine衰减到0.1e-5), grad_clip=1.0, bf16 autocast
- 产物: out/sft_mini_ep1_768.pth (fp16, ~137MB) — 后续 DPO/GRPO 的基线

## 观察
- 采样到的 loss 点数: 567 (每100步一条)
- 首个记录 loss: 1.9869 @ step 100
- 最后记录 loss: 1.5287 @ step 56608
- loss 最小值: 1.2895
- loss 整体从 ~2.0 区间下降到 ~1.5-1.6 区间 (SFT 正常水平, 该模型官方 full_sft 相近)
- aux_loss 恒为 0 (dense 模型无 MoE 路由项)
- 日志尾部的 epoch_time 为剩余分钟数估算

## 备注
- loss_curve.csv 是 (step, loss) 原始序列, 可本地画曲线
- 全参 SFT 与 LoRA (train_lora.py) 的对比留作后续实验

## 定性对比（同一 8 问，SFT 前后）
- probe_before_pretrain.txt : pretrain 底座（SFT 前，无指令模板，直接续写）
- probe_after_sft.txt       : sft_mini_ep1（全参 SFT 1 epoch 后）
- 评测脚本: /root/autodl-tmp/eval_probe.py（固定每问 seed=1000+i，top_p=0.95 temp=0.85）
