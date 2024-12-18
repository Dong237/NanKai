#!/usr/bin/env python

"""


This script performs data collection and processing tasks by automatically looping through
all existing .txt files in a folder and generating Q&A pairs based on the documents. 

The above process is performed sequentially for each document, and requires a local LLM.

Methods are inspired by the following works: 
    [Injecting New Knowledge into Large Language Models via Supervised Fine-Tuning](https://arxiv.org/abs/2404.00213)
    [Assisting in Writing Wikipedia-like Articles From Scratch with Large Language Models](https://arxiv.org/abs/2402.14207)

"""

import os
import re


import logging
import argparse
import jieba
from tqdm import tqdm
from typing import List, Literal, Callable
from vllm import LLM
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.helper import (
    read_txt_file, 
    read_docx_file,
    vllm_chat,
    hf_chat,
    closest_power_of_2,
    setup_logging,
    jsonl_append
)
from utils.prompts import (
    SYSTEM_PROMPT_DATA_GEN,
    SYSTEM_PROMPT_ROLE_GEN,
    THEME_SUMMARIZATION, 
    FACT_DISTILLATION,
    ROLE_GENERATION,
    ROLE_BASED_QA_DIVERSIFY,
    FACT_BASED_QA_GEN_SKIP,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Script for processing text into SFT-ready format"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="datasets/txt_data",
        help="Path to the directory containing the data files."
    )
    parser.add_argument(
        "--file_type",
        type=str,
        default="docx",
        help="input data types, either 'docx' or 'txt'."
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="/data/hf_model",
        help="Path to the pre-trained model for generating data."
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="vllm",
        help="type of model to load, either 'vllm' or 'hf'"
    )
    parser.add_argument(
        "--chunk_size_by_token",
        type=int,
        default=512,
        help="Chunk size measured by tokens, smaller chunk size leads to finer granularity when extracting (more facts being extracted)"
    )
    parser.add_argument(
        "--qa_amount_per_fact",
        type=int,
        default=10,
        help="Number of QA pairs to generate per fact, default to 10 as in Mecklenburg et al, 2024 ."
    )
    parser.add_argument(
        "--role_amount_per_fact",
        type=int,
        default=3,
        help="Number of roles to assign per fact, roles are played by an LLM to augment/diversify the QA pairs for a given fact"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="datasets/qa_pairs.jsonl",
        help="Path to save the final json dataset for finetuning"
    )
    return parser.parse_args()


def initialize_model_and_tokenizer(model_name_or_path: str, method: Literal["vllm", "hf"] = "vllm"):
    """Initializes and returns the model and tokenizer."""
    if method == "vllm":
        model = LLM(model_name_or_path, tensor_parallel_size=1)
    elif method == "hf":
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype="auto",
            device_map="auto"
        )
    else:
        raise ValueError(f"Invalid method: {method}")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    return model, tokenizer


def get_generate_function(model_type):
    if model_type == "vllm":
        generate_function = vllm_chat
    elif model_type == "hf":
        generate_function = hf_chat
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    return generate_function

def get_file_reading_function(file_type):
    if file_type == "txt":
        file_reading_function = read_txt_file
    elif file_type == "docx":
        file_reading_function = read_docx_file
    return file_reading_function

def chunking_by_token_size(content: str, overlap_token_size=64, max_token_size=1024):
    tokens = list(jieba.cut(content))
    results = []
    for index, start in enumerate(range(0, len(tokens), max_token_size - overlap_token_size)):
        chunk_content = ''.join(tokens[start: start + max_token_size])
        results.append({
            "tokens": min(max_token_size, len(tokens) - start),
            "content": chunk_content.strip(),
            "chunk_order_index": index,
        })
    return results


def parse_diversified_qa_response(diversified_qa_response: str) -> list:
    elements = re.findall(
        r'Q:.*?A:.*?(?=\d+\.|$)', 
        diversified_qa_response, 
        re.DOTALL
    )
    elements = [element.strip() for element in elements]
    return elements


def parse_numbered_elements(response: str) -> list:
    return re.findall(r'\d+\.\s*(.*?)(?=\n\d+\.|$)', response)


def generate_qa_pairs_from_one_file(
    data_path: str, 
    data_name: str, 
    qa_amount_per_fact: int, 
    role_amount_per_fact: int, 
    chunk_size_by_token: int,
    model: AutoModelForCausalLM, 
    tokenizer: AutoTokenizer,
    read_file_function: Callable[[str], str],
    generate_function: Callable[[str], str],
    ) -> List[dict]:
    
    logging.info(f"Collecting data from the single txt file: {data_name}")
    data = read_file_function(os.path.join(data_path, data_name))
    overlap_token_size = closest_power_of_2(chunk_size_by_token)
    chunking_results = chunking_by_token_size(
        content=data, 
        overlap_token_size=overlap_token_size,
        max_token_size=chunk_size_by_token 
    )

    qa_pairs_all_chunks = []
    facts = []
    for chunk in tqdm(chunking_results, desc="Processing chunks..."):
        content = chunk["content"]

        # Summarizing the theme of the chunk
        theme_response = generate_function(
            model, 
            tokenizer,
            user_prompt=THEME_SUMMARIZATION.format(passage=content), 
            system_prompt=SYSTEM_PROMPT_DATA_GEN
        )
        if len(theme_response) == 0:
            logging.warning("No theme found for the chunk")
            continue
        
        # Extract facts from the chunk
        facts_response = generate_function(
            model, 
            tokenizer,
            user_prompt=FACT_DISTILLATION.format(
                theme=theme_response, 
                passage=content
            ), 
            system_prompt=SYSTEM_PROMPT_DATA_GEN
        )
        try:
            parsed_facts = parse_numbered_elements(facts_response)
        except:
            logging.warning("Fact extraction failed for the chunk, skipping now...")
            continue
        
        # Processing each fact in all extracted facts
        for fact in tqdm(parsed_facts, desc="Processing facts in the chunk"):
            # Generate a standard Q&A pair based on the fact
            qa_response = generate_function(
                model, 
                tokenizer,
                user_prompt=FACT_BASED_QA_GEN_SKIP.format(
                    theme=theme_response, 
                    fact=fact
                ), 
                system_prompt=SYSTEM_PROMPT_DATA_GEN
            )
            if qa_response != "SKIP":
                # Generate possible roles
                roles_response = generate_function(
                    model, 
                    tokenizer,
                    user_prompt=ROLE_GENERATION.format(
                        amount=role_amount_per_fact,
                        theme=theme_response
                    ),
                    system_prompt=SYSTEM_PROMPT_ROLE_GEN
                )
                try:
                    parsed_roles_response = parse_numbered_elements(roles_response)
                except:
                    logging.warning("Generation of different roles failed, skipping this fact now...")
                    continue
                
                role_amount_per_fact_actual = len(parsed_roles_response)
                assert qa_amount_per_fact > role_amount_per_fact_actual, "Each role should generate at least one QA pair"
                scheduler = divmod(qa_amount_per_fact, role_amount_per_fact_actual)
                qa_amount_per_role_schedule = [scheduler[0]] * (role_amount_per_fact_actual - 1) + [scheduler[0] + scheduler[1]]

                for idx, role in enumerate(parsed_roles_response): 
                    # Diversify the standard Q&A pair given a role   
                    diversified_qa_response = generate_function(
                        model,
                        tokenizer,
                        user_prompt=ROLE_BASED_QA_DIVERSIFY.format(
                            role=role,
                            theme=theme_response,
                            amount=qa_amount_per_role_schedule[idx],
                            qa_pair=qa_response,
                            fact=fact
                        ), 
                        system_prompt=SYSTEM_PROMPT_DATA_GEN
                    )
                    try:
                        qa_pairs_per_role = parse_diversified_qa_response(diversified_qa_response)
                    except:
                        logging.warning("Q&A augmentation failed for role, skipping this role now...")
                    facts.extend([fact] * len(qa_pairs_per_role))
                    qa_pairs_all_chunks.extend(qa_pairs_per_role)
            else:
                logging.warning(f"The fact {fact} is too broad or ambiguous to generate any Q&A pairs. Skipping.")
                continue

    qa_pairs_json = []
    for idx, (qa, fact) in enumerate(zip(qa_pairs_all_chunks, facts)):
        try:
            q, a = qa.split('\n')
        except ValueError:
            logging.warning(f"Q&A pair {qa} will be omitted due to splitting failure.")
            continue
        q = q.replace('Q:', '').strip()
        a = a.replace('A:', '').strip()
        qa_pairs_json.append({
            "id": f"identity_{idx}",
            "conversations": [
                {"from": "user", "value": q},
                {"from": "assistant", "value": a}
            ],
            "fact": fact,
            "file_name": data_name
        })
    logging.info(f"Processing for the file {data_name} finished.")
    return qa_pairs_json


def generate_qa_pairs_from_folder(
    data_path: str, 
    qa_amount_per_fact: int, 
    role_amount_per_fact: int, 
    chunk_size_by_token: int, 
    save_dir: str, 
    model: AutoModelForCausalLM, 
    tokenizer: AutoTokenizer,
    read_file_function: Callable[[str], str],
    generate_function: Callable[[str], str],
    ):
    
    files = os.listdir(data_path)
    for file in tqdm(files, desc="Processing files..."):
        data_list = generate_qa_pairs_from_one_file(
            data_path, 
            file,
            qa_amount_per_fact, 
            role_amount_per_fact,
            chunk_size_by_token,
            model,
            tokenizer,
            read_file_function,
            generate_function
        )
        jsonl_append(save_dir, data_list)
        logging.info(f"Data for {file} appended to {save_dir}")


if __name__ == "__main__":
    import time
    start_time = time.time()
    print(time.strftime("%Y-%m-%d %H:%M:%S"))
    setup_logging()
    args = parse_args()
    
    model, tokenizer = initialize_model_and_tokenizer(
        args.model_name_or_path,
        args.model_type,
        )
    
    generate_qa_pairs_from_folder(
        data_path=args.data_path,
        qa_amount_per_fact=args.qa_amount_per_fact, 
        role_amount_per_fact=args.role_amount_per_fact,
        chunk_size_by_token=args.chunk_size_by_token,
        save_dir=args.save_dir,
        model=model,
        tokenizer=tokenizer,
        read_file_function=get_file_reading_function(args.file_type),
        generate_function=get_generate_function(args.model_type),
    )
    end_time = time.time()
    print(time.strftime("%Y-%m-%d %H:%M:%S"))
    # Calculate and print the total running time
    elapsed_time = end_time - start_time
    elapsed_time_hours = elapsed_time / 3600
    # Print the total running time in hours
    print(f"Total running time: {elapsed_time_hours:.2f} hours")
