# MiniMind Post-training Lab - Experiment Log

## 今日实验总览

| exp | 方法 | 数据 | 轮数 | lr | loss末值 | loss最小 | 8问对比 |
|---|---|---|---|---|---|---|---|
| exp1 | 全参SFT | sft_t2t_mini | 1 | 1e-5 | 1.5287 | 1.2895 | probe_before/after |
| exp2 | 全参SFT | sft_t2t_mini | 2 | 1e-5 | 1.6683 | 1.0141 | probe_after_sft_ep2 |
| exp3 | LoRA | sft_t2t_mini | 1 | 1e-4 | 1.6561 | 1.5412 | probe_lora |

## 观察要点

- exp1 vs exp2: 看多训 1 轮（epochs 2）loss 是否继续下降、末值差异
- exp1 vs exp3: 全参(63.9M 全训) vs LoRA(仅 adapter) 在同一数据 1 epoch 的 loss 形态与参数量差异（exp3 日志含 LoRA 参数量占比）
- 每目录 probe_*.txt 为同一 8 问的模型回答（pretrain=SFT前底座；after_sft=全参SFT后；lora=LoRA版）

## 权重说明

*.pth 不入库（>100MB），保留在实例 /root/autodl-tmp/minimind/out/
| 文件 | 说明 |
|---|---|
| sft_mini_ep1_768.pth | exp1 全参 SFT 1 epoch |
| sft_mini_ep2_768.pth | exp2 全参 SFT 2 epochs |
| lora_sft_ep1_768.pth (797KB) | exp3 LoRA adapter（需配 pretrain_768.pth 使用）|
