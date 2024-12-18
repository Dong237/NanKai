# KnowledgeUpdate

This is the Repo for updating the knowledge of LLMs through LoRA finetuning with time limit of 1 day

## Table of Contents

- [Setup](#setup)
- [Data Collection](#data-collection)

  - [Arguments](#arguments)
  - [Processing Steps](#processing-steps)
  - [Example](#example)
- [Knowledge Updating &amp; Fine-tuning](#knowledge-updating--finetuning)

  - [Overview](#overview)
  - [Fine-tuning Process](#fine-tuning-process)
  - [Running the Fine-tuning Script](#running-the-fine-tuning-script)
    - [Example Command in `finetune_lora_ds.sh`](#example-command-in-finetune_lora_dssh)

## Setup

 In a conda env with pytorch available, run:

```
pip install -r requirements.txt
```

## Data Collection

To generate a dataset of Q&A pairs from text files, use the `data_processor/data_collection_local.py` script (recommended), which automates processing `.txt` files into a JSON format suited for supervised fine-tuning (SFT) with a local LLM. This script loops through all `.txt` files in the specified directory, extracts facts, and generates Q&A pairs with diversified roles based on each fact.

Besides, we provide an extra script called `data_processor/data_collection_api.py` which leverages an LLM API and achieve the same goal as above. We split the two method into two scripts for better comparison, and we suggest to use whichever is more stable in your environment. Note that '**it is highly unrecommended to use concurrent processing in both cases** since concurrent process may bring extra computing resource consumption and encounter rate limit error when calling the API.

The data collection process originates from the fact-based approach in [this work](https://arxiv.org/abs/2404.00213). To run the data collection process, use the `collect_data.sh` script located in the `shells` folder. This script starts `data_collection.py` with all necessary arguments. You can customize options like the number of Q&A pairs per fact, the number of roles for diversifying each Q&A, and the chunk size for token-based processing.

### Processing Steps

1. **Text Chunking**: The script tokenizes and chunks each `.txt` file based on `chunk_size_by_token`.
2. **Theme Summarization**: Summarizes each chunk’s theme to inform fact extraction.
3. **Fact Extraction**: Extracts discrete facts from the chunk content.
4. **Q&A Generation**: Creates Q&A pairs based on each fact.
5. **Role-based Diversification**: Assigns roles to diversify Q&A pairs for each fact.

### Example

To start the data collection with default settings, run:

```bash
bash shells/collect_data.sh
```

**Note that arguments `chunk_size_by_token` and `role_amount_per_fact` have a great impact on the data processing time. Smaller chunk size will lead to finer granularity of fact extraction, while also slowing down the collection process. Bigger values are recommanded when you have too many text files. Also setting `role_amount_per_fact` too high will slow down the generation process**

## Knowledge Updating & Finetuning

### Overview

The fine-tuning process involves customizing a pre-trained model to improve its performance on a specific dataset. This setup allows you to fine-tune using DeepSpeed with multi-GPU support. We use **LoRA** (Low-Rank Adaptation of Large Language Models) for more efficient training, as it enables model adaptation with minimal added parameters. The process integrates with **wandb** (Weights and Biases) for experiment tracking, and **DeepSpeed** for efficient memory and computational resource management.

### Fine-tuning Process

The fine-tuning Python script (`finetune.py`) relies on the Hugging Face Transformers library and several other libraries like `deepspeed`, `wandb`, and `peft` for LoRA support. Here is a summary of the main script features:

1. **Argument Parsing**: Defines configurations for the model, data, training, and optional LoRA parameters, all handled through data classes.
2. **Data Preprocessing**: The script processes data either eagerly or lazily, offering flexibility in memory usage. The `preprocess` function prepares conversation-based datasets into input and target tensors.
3. **Training Dataset**: Supports both a pre-loaded `SupervisedDataset` and a lazy-loaded `LazySupervisedDataset`, which can handle larger datasets efficiently.
4. **LoRA Configuration**: If `use_lora` is enabled, the script applies LoRA configurations to reduce memory requirements, making it suitable for fine-tuning on limited resources.
5. **Training**: The main `train` function sets up the trainer using Hugging Face's `Trainer` class, enabling distributed training with DeepSpeed and allowing seamless wandb integration for logging and monitoring.
6. **Saving**: At the end of training, the model's state is saved using a `safe_save_model_for_hf_trainer` function.

### Running the Fine-tuning Script

To launch the fine-tuning process, use the shell script `finetune_lora_ds.sh` located in the `shells` folder. This script configures the environment and launches the Python script with distributed training options. Below are the steps for using `finetune_lora_ds.sh`.

1. **Configure Multi-GPU Training**:

   - `GPUS_PER_NODE`: Number of GPUs per node.
   - `NNODES`: Number of GPU nodes.
   - `NODE_RANK`: Rank of this node.
   - `MASTER_ADDR`: IP address of the main node.
   - `MASTER_PORT`: Port for communication.
2. **Model and Data Paths**:

   - Update `MODEL` to point to the model path.
   - Update `DATA` to the path of your dataset.
3. **Run with Optional Arguments**:

   - `DS_CONFIG_PATH`: Path to your DeepSpeed configuration.
   - `WANDB_KEY`: Wandb API key for logging.

The shell script can be run as follows:

```bash
bash shells/finetune_lora_ds.sh -m /path/to/model -d /path/to/data --deepspeed /path/to/deepspeed/config --wandb_key your_wandb_key
```

This setup initiates training with the specified configurations, utilizing DeepSpeed’s capabilities to handle large models across multiple GPUs. Additionally, wandb logging allows for tracking metrics and monitoring progress in real-time.

#### Example Command in `finetune_lora_ds.sh`

```bash
torchrun $DISTRIBUTED_ARGS finetune.py \
    --model_name_or_path $MODEL \
    --key $WANDB_KEY \
    --use_wandb True \
    --wandb_run_name "run-lora" \
    --data_path $DATA \
    --bf16 True \
    --output_dir output \
    --num_train_epochs 20 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 8 \
    --validation True \
    --validation_size 1000 \
    --logging_strategy "steps" \
    --logging_steps 4 \
    --eval_strategy "steps" \
    --eval_steps 4 \
    --save_strategy "steps" \
    --save_total_limit 3 \
    --learning_rate 1e-5
```
