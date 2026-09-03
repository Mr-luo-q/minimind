#!/bin/bash
# Phase 2: E1(4B LoRA) -> E2(1.5B 全参) -> E3(1.5B LoRA) -> 合并 -> 8问评测
export PATH=/root/miniconda3/bin:$PATH
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
Q=/root/autodl-tmp/qwen_sft
source $Q/venv/bin/activate
cd $Q

echo "===== E1: Qwen3-4B LoRA ====="
llamafactory-cli train $Q/configs/e1_qwen3_4b_lora.yaml > $Q/logs/e1.log 2>&1
echo "E1_EXIT=$?"
tail -3 $Q/logs/e1.log

echo "===== E2: Qwen2.5-1.5B full ====="
llamafactory-cli train $Q/configs/e2_qwen15b_full.yaml > $Q/logs/e2.log 2>&1
echo "E2_EXIT=$?"
tail -3 $Q/logs/e2.log

echo "===== E3: Qwen2.5-1.5B LoRA ====="
llamafactory-cli train $Q/configs/e3_qwen15b_lora.yaml > $Q/logs/e3.log 2>&1
echo "E3_EXIT=$?"
tail -3 $Q/logs/e3.log

echo "===== merge adapters ====="
llamafactory-cli export $Q/configs/export_e1.yaml > $Q/logs/export_e1.log 2>&1
echo "EXP1_EXIT=$?"
llamafactory-cli export $Q/configs/export_e3.yaml > $Q/logs/export_e3.log 2>&1
echo "EXP3_EXIT=$?"

echo "===== probes (8 questions) ====="
python $Q/scripts/eval_qwen_probe.py --model $Q/models/Qwen3-4B --tag base_qwen3-4b --outdir $Q/probes
python $Q/scripts/eval_qwen_probe.py --model $Q/out/merged_e1_qwen3-4b_lora --tag e1_qwen3-4b_lora --outdir $Q/probes
python $Q/scripts/eval_qwen_probe.py --model $Q/models/Qwen2.5-1.5B-Instruct --tag base_qwen15b --outdir $Q/probes
python $Q/scripts/eval_qwen_probe.py --model $Q/out/e2_qwen15b_full --tag e2_qwen15b_full --outdir $Q/probes
python $Q/scripts/eval_qwen_probe.py --model $Q/out/merged_e3_qwen15b_lora --tag e3_qwen15b_lora --outdir $Q/probes

echo "===== summary ====="
ls -la $Q/out/ | head
du -sh $Q/out/* 2>/dev/null
echo "PHASE2_DONE"
