# KnowledgeUpdate

[中文版本](README.zh-CN.md) | [English Version](README.md)

This is the Repo for updating the knowledge of LLMs through LoRA finetuning with time limit of 1 day

## Table of Contents

- [Setup](#setup)
- [Data Collection](#data-collection)
- [Knowledge Updating &amp; Fine-tuning](#knowledge-updating--finetuning)

  - [Overview](#overview)
  - [Fine-tuning Process](#fine-tuning-process)
  - [Running the Fine-tuning Script](#running-the-fine-tuning-script)

- [Running Test Case 1](#running-test-case-1)
  - [Objective](#objective)
  - [Test Setup](#test-setup)
  - [Procedure](#procedure)
  - [Pass/Fail Criteria](#passfail-criteria)
  - [Command to Execute](#command-to-execute)
- [Running Test Case 2](#running-test-case-2)
  - [Objective](#objective-1)
  - [Test Setup](#test-setup-1)
  - [Procedure](#procedure-1)
  - [Execution Steps](#execution-steps)
  - [Pass/Fail Criteria](#passfail-criteria-1)
  - [Output](#output-1)


## Setup

In a conda env, run:

```
pip install -r requirements.txt
```

## Data Collection

The data collection process generates a dataset of Q&A pairs from text files by automaticly processing text files into a ``.josnl`` format suited for supervised fine-tuning (SFT) of an LLM. IT loops through all text files in the specified directory, extracts facts, and generates Q&A pairs with diversified roles based on each fact.

The data collection process originates from the fact-based approach in [this work](https://arxiv.org/abs/2404.00213). To run the data collection process, use the `collect_data.sh` script. This script starts `collect_data.py` with all necessary arguments. The processing steps are as follows:

   1. **Text Chunking**: The script tokenizes and chunks each `.txt` file based on `chunk_size_by_token`.
   2. **Theme Summarization**: Summarizes each chunk’s theme to inform fact extraction.
   3. **Fact Extraction**: Extracts discrete facts from the chunk content.
   4. **Q&A Generation**: Creates Q&A pairs based on each fact.
   5. **Role-based Diversification**: Assigns roles to diversify Q&A pairs for each fact.

The above process can be configured with following arguments in `collect_data.sh`:

- **`--data_path`**: Path to the directory containing input files.
- **`--file_type`**: Type of input files, either `'docx'` or `'txt'`.
- **`--model_name_or_path`**: Path to the pre-trained model or Hugging Face model ID.
- **`--model_type`**: Model backend, either `'vllm'` or `'hf'` or `'api'`. When setting as `'api'`, you are calling the api of the LLM, please  make sure you have the `api_chat` function in `utils/helper.py` correctly configured. 
- **`--chunk_size_by_token`**: Token limit for splitting content into chunks; smaller values yield more granular extractions but also slow down processing.
- **`--qa_amount_per_fact`**: Number of Q&A pairs to generate for each fact.
- **`--role_amount_per_fact`**: Number of roles simulated to diversify Q&A pairs.
- **`--save_dir`**: File path to save the generated `.jsonl` dataset (default: `datasets/qa_pairs.jsonl`).

To start the data collection with default settings, run:

```bash
bash collect_data.sh
```

## Knowledge Updating & Finetuning

### Overview

The fine-tuning process involves customizing a pre-trained model to improve its performance on a specific dataset. This setup allows one to fine-tune using DeepSpeed with multi-GPU support. We use **LoRA** (Low-Rank Adaptation of Large Language Models) for more efficient training, as it enables model adaptation with minimal added parameters. The process integrates with **wandb** (Weights and Biases) for experiment tracking, and **DeepSpeed** for efficient memory and computational resource management.

### Fine-tuning Process

The fine-tuning Python script (`finetune.py`) relies on the Hugging Face Transformers library and several other libraries like `deepspeed` and `peft` for LoRA support. Here is a summary of the main script features:

1. **Argument Parsing**: Defines configurations for the model, data, training, and optional LoRA parameters, all handled through data classes.
2. **Data Preprocessing**: The script processes data either eagerly or lazily, offering flexibility in memory usage. The `preprocess` function prepares conversation-based datasets into input and target tensors.
3. **Training Dataset**: Supports both a pre-loaded `SupervisedDataset` and a lazy-loaded `LazySupervisedDataset`, which can handle larger datasets efficiently.
4. **LoRA Configuration**: If `use_lora` is enabled, the script applies LoRA configurations to reduce memory requirements, making it suitable for fine-tuning on limited resources.
5. **Training**: The main `train` function sets up the trainer using Hugging Face's `Trainer` class, enabling distributed training with DeepSpeed and allowing seamless wandb integration for logging and monitoring.
6. **Saving**: At the end of training, the model's state is saved using a `safe_save_model_for_hf_trainer` function.

### Running the Fine-tuning Script

To launch the fine-tuning process, use the shell script `finetune_lora_ds.sh` located in the `shells` folder. This script configures the environment and launches the Python script with distributed training options. Below are the steps for using `finetune_lora_ds.sh`.

1. **Configure Multi-GPU Training (normally no changes required)**:

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
   - `WANDB_KEY`: Wandb API key for logging, is wandb not used, training will be logged with tensorboard.

The shell script can be run as follows:

```bash
bash shells/finetune_lora_ds.sh 
```

**Note: before starting the finetuning, make sure to play with the ``per_device_train_batch_size`` argument to find the optimal batch size for your ressource setting.**


## Running Test Case 1

### **Objective**
This test evaluates whether the LoRA fine-tuning process using the generated dataset can be completed within **8 hours** on the specified hardware setup. The test involves running three fine-tuning experiments with different datasets and averaging the total runtime to assess time efficiency.

### **Test Setup**
- **Hardware Configuration**:
  - GPUs used: `CUDA_VISIBLE_DEVICES=x,x,x,x` (4 A100 GPUs)
  - `GPUS_PER_NODE` is dynamically determined by the script.

- **Datasets**:
  - Three datasets (`qa_pairs_001.jsonl`, `qa_pairs_002.jsonl`, and `qa_pairs_003.jsonl`) are used for separate experiments.
  - Each dataset is processed individually during each run.

- **Fine-Tuning Script**:
  - The fine-tuning script is executed using `torchrun` for distributed multi-GPU training.
  - Key parameters:
    - Batch size per GPU: `32`
    - Gradient accumulation steps: `4`
    - Epochs: `5`
    - Learning rate: `3e-4`
    - Lazy preprocessing: Enabled.
    - LoRA fine-tuning: Enabled.

### **Procedure**
1. **Execution**:
   - The test runs the fine-tuning process three times, each using one of the specified datasets and saving the output model to a separate directory.
   - The elapsed time for each run is recorded.

2. **Time Measurement**:
   - Start time and end time are logged for each run.
   - Total runtime is calculated as the sum of elapsed times for all three runs.
   - Average runtime is computed.

3. **Output**:
   - The script prints the average runtime in hours at the end of all runs.

### **Pass/Fail Criteria**
- **Pass**: If the average runtime for the three runs is **≤ 8 hours**.
- **Fail**: If the average runtime exceeds 8 hours.

### **Command to Execute**
Run the test script as follows:
```bash
bash test_case_1.sh
```

## Running Test Case 2

### **Objective**
This test verifies whether the fine-tuning losses from the three runs in **Test Case 1** converge. The convergence is evaluated using the **Cauchy test**, which checks if the difference between consecutive loss values becomes smaller than a specified threshold.

### **Test Setup**
- **Input**:
  - Three `.npy` files containing loss values from the fine-tuning runs, stored in the `losses` folder.
- **Threshold**:
  - The convergence threshold is set to **0.001** by default.
  - You can customize this value when running the script.

### **Procedure**
1. **Script Execution**:
   - The test is executed using the Python script `cauchy_test.py`, which can be started with `bash cauchy_test.sh`
   - It applies the Cauchy test to each `.npy` file in the specified loss folder.

2. **Convergence Check**:
   - For each loss file, the script checks whether the absolute difference between consecutive loss values falls below the threshold.
   - If all files pass the Cauchy test, the fine-tuning losses are considered convergent.

3. **Output**:
   - The script outputs a table showing whether each loss file passes the convergence test.

### **Execution Steps**
1. **Run the Bash Script**:
   Use the provided `cauchy_test.sh` script to start the Python convergence test:
   ```bash
   bash cauchy_test.sh
   ```
   - **Options**:
     - `--folder` or `-f`: Path to the folder containing `.npy` files (default: `losses`).
     - `--threshold` or `-t`: Convergence threshold for the Cauchy test (default: `0.001`).
     - `--script` or `-s`: Python script path (default: `cauchy_test.py`).

   Example command with custom options:
   ```bash
   bash cauchy_test.sh -f custom_losses -t 0.0005
   ```

2. **Run the Python Script Directly**:
   Alternatively, you can directly execute the Python script:
   ```bash
   python cauchy_test.py --loss_folder losses --threshold 0.001
   ```

### **Pass/Fail Criteria**
- **Pass**:
  - All three `.npy` loss files pass the Cauchy test, indicating that the losses converge.
- **Fail**:
  - Any `.npy` file fails the Cauchy test, indicating non-convergent losses.


### **Output**
After running the script, a table is displayed with the results:

| **File Name**        | **Converges?** |
|-----------------------|----------------|
| `qa_pairs_001.npy`    | ✅ Yes         |
| `qa_pairs_002.npy`    | ✅ Yes         |
| `qa_pairs_003.npy`    | ✅ Yes         |

If all files pass, the test passes. Otherwise, it fails.