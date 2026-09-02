# MiniMind Post-training Lab - Experiment Log

## 今日实验总览

| exp | 方法 | 数据 | epochs | lr | 产物 |
|---|---|---|---|---|---|
| exp1 | 全参 SFT (train_full_sft.py) | sft_t2t_mini | 1 | 1e-5 | out/sft_mini_ep1_768.pth |
| exp2 | 全参 SFT | sft_t2t_mini | 2 | 1e-5 | out/sft_mini_ep2_768.pth |
| exp3 | LoRA (train_lora.py, from pretrain) | sft_t2t_mini | 1 | 1e-4 | out/lora_sft_ep1_768.pth (+pretrain) |

## loss 摘要

| exp | loss 末值 | loss 最小值 |
|---|---|---|
| exp1_sft_mini_ep1 | loss 最小值: 1.2895 loss 整体从 ~2.0 区间下降到 ~1.5-1.6 区间 (SFT 正常水平, 该模型官方 full_sft 相近) |
| exp2_sft_mini_ep2 | |
| exp3_lora_sft_ep1 | |

## 定性对比（8 问 probe）

每个 exp 目录下 probe_*.txt：pretrain=SFT前底座；after_sft=全参SFT后；lora=LoRA版。
逐问对比可见：SFT 让模型学会遵循指令格式；pretrain 只会无格式续写。

> 权重 *.pth 不入库（>100MB），保留在实例数据盘 /root/autodl-tmp/minimind/out/
