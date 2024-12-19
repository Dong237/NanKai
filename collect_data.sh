#!/bin/bash

# Set up environment variables
export CUDA_VISIBLE_DEVICES=1
export PYTHONUNBUFFERED=1  # Ensures real-time logging


# 根据实际情况更改以下设置
DATA_PATH="datasets/红色文本入库003"  # Default data folder
MODEL_PATH="/data/youxiang/huggingface/Qwen2.5-7B-Instruct"  # Default model path
SAVE_DIR="datasets/qa_pairs_003.jsonl"  # Output path for generated data

FILE_TYPE="docx"  # Input file format
MODEL_TYPE="vllm"  # Model type (vllm or hf)
CHUNK_SIZE=1024  # Chunk size in tokens
QA_PER_FACT=10  # QA pairs per fact
ROLE_PER_FACT=2  # Roles per fact

LOG_DIR="logs"
LOG_FILE="$LOG_DIR/run_$(date '+%Y%m%d_%H%M%S').log"

# Create the log directory if it doesn't exist
mkdir -p "$LOG_DIR"

# Parse arguments for customization
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --data_path) DATA_PATH="$2"; shift ;;
        --model_path) MODEL_PATH="$2"; shift ;;
        --save_dir) SAVE_DIR="$2"; shift ;;
        --file_type) FILE_TYPE="$2"; shift ;;
        --model_type) MODEL_TYPE="$2"; shift ;;
        --chunk_size) CHUNK_SIZE="$2"; shift ;;
        --qa_per_fact) QA_PER_FACT="$2"; shift ;;
        --role_per_fact) ROLE_PER_FACT="$2"; shift ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

# Log start time
START_TIME=$(date +%s)
echo "Process started at: $(date)" | tee -a "$LOG_FILE"

# Command to execute the Python script
CMD="python collect_data.py \
    --data_path $DATA_PATH \
    --file_type $FILE_TYPE \
    --model_name_or_path $MODEL_PATH \
    --model_type $MODEL_TYPE \
    --chunk_size_by_token $CHUNK_SIZE \
    --qa_amount_per_fact $QA_PER_FACT \
    --role_amount_per_fact $ROLE_PER_FACT \
    --save_dir $SAVE_DIR"

# Execute the Python script and capture the output in the log file
echo "Executing command:" | tee -a "$LOG_FILE"
echo "$CMD" | tee -a "$LOG_FILE"
eval "$CMD" 2>&1 | tee -a "$LOG_FILE"

# Log end time
END_TIME=$(date +%s)
echo "Process ended at: $(date)" | tee -a "$LOG_FILE"

# Calculate total runtime
ELAPSED_TIME=$((END_TIME - START_TIME))
ELAPSED_TIME_HOURS=$(echo "scale=2; $ELAPSED_TIME / 3600" | bc)
echo "总运行时长: $ELAPSED_TIME_HOURS 小时" | tee -a "$LOG_FILE"

# Notify user
echo "数据生成完毕，日志存储于 $LOG_FILE."
