#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
export CUDA_VISIBLE_DEVICES=1,6
DIR=`pwd`

# Define the list of dataset files and corresponding output directories
DATASETS=("datasets/qa_pairs_001.jsonl" "datasets/qa_pairs_002.jsonl" "datasets/qa_pairs_003.jsonl")
OUTPUT_DIRS=("output/output_model_001" "output/output_model_002" "output/output_model_003")
MODEL="/data/youxiang/huggingface/Qwen2.5-7B-Instruct" 
DS_CONFIG_PATH="ds_config_zero2.json"

# Number of GPUs per GPU worker
GPUS_PER_NODE=$(python -c 'import torch; print(torch.cuda.device_count())')

# Number of GPU workers, for single-worker training, please set to 1
NNODES=${NNODES:-1}

# The rank of this worker, should be in {0, ..., WORKER_CNT-1}, for single-worker training, please set to 0
NODE_RANK=${NODE_RANK:-0}

# The ip address of the rank-0 worker, for single-worker training, please set to localhost
MASTER_ADDR=${MASTER_ADDR:-localhost}

# The port for communication
MASTER_PORT=${MASTER_PORT:-6001}

DISTRIBUTED_ARGS="\
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT"

# Initialize variables for tracking time
TOTAL_TIME=0
RUN_COUNT=0

for i in "${!DATASETS[@]}"; do
    DATA=${DATASETS[i]}
    OUTPUT_DIR=${OUTPUT_DIRS[i]}

    echo "******************************************"
    echo "微调启动, 数据集: $DATA"
    echo "******************************************"

    # Record the start time
    START_TIME=$(date +%s)

    # Run the training script
    torchrun $DISTRIBUTED_ARGS finetune.py \
        --model_name_or_path $MODEL \
        --data_path $DATA \
        --bf16 True \
        --output_dir $OUTPUT_DIR \
        --num_train_epochs 5 \
        --per_device_train_batch_size 8 \
        --per_device_eval_batch_size 32 \
        --gradient_accumulation_steps 8 \
        --validation False \
        --validation_size 1000 \
        --logging_strategy "steps" \
        --logging_steps 2 \
        --eval_strategy "steps" \
        --eval_steps 20 \
        --save_strategy "steps" \
        --save_total_limit 2 \
        --learning_rate 3e-4 \
        --weight_decay 0.1 \
        --adam_beta2 0.95 \
        --warmup_ratio 0.01 \
        --lr_scheduler_type "cosine" \
        --model_max_length 512 \
        --lazy_preprocess True \
        --use_lora \
        --gradient_checkpointing \
        --deepspeed ${DS_CONFIG_PATH}

    # Record the end time
    END_TIME=$(date +%s)

    # Calculate the elapsed time
    ELAPSED_TIME=$((END_TIME - START_TIME))

    # Accumulate the total time and increment run count
    TOTAL_TIME=$((TOTAL_TIME + ELAPSED_TIME))
    RUN_COUNT=$((RUN_COUNT + 1))
done

# Calculate the average time in hours with two decimal places
AVERAGE_TIME_HOURS=$(echo "scale=2; $TOTAL_TIME / $RUN_COUNT / 3600" | bc) #
echo "微调全部结束"
echo "平均微调总耗时: $AVERAGE_TIME_HOURS 小时."
