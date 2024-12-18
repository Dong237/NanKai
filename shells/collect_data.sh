#!/bin/bash

export CUDA_VISIBLE_DEVICES="1"

# Bash script to run the data collection script with specified arguments
# --role_amount_per_fact has an impact on processing speed, unrecommended to set it too HIGH
# --chunk_size_by_token defines granularity, has an impact on memory usage, unrecommended to set it too LOW


python data_processor/collect_data_local.py \
    --data_path "红色文本入库/红色文本入库002" \
    --file_type "docx" \
    --model_name_or_path "/data/youxiang/huggingface/Qwen2.5-7B-Instruct" \
    --model_type "vllm" \
    --chunk_size_by_token 1024 \
    --qa_amount_per_fact 10 \
    --role_amount_per_fact 2 \
    --save_dir "datasets/qa_pairs_002.jsonl" \