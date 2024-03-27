import logging
import os
import sys

sys.path.append(os.getcwd())

from dotenv import load_dotenv

load_dotenv(".env")

import pickle

import hydra
import pandas as pd
import torch
import transformers
from datasets import Dataset, load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer

from mediqa.configs import TrainingConfigs, register_base_configs
from mediqa.dataset import MEDIQADataset


def preprocessing(dataset):
    samples = []
    for text, error_flag in zip(dataset["Text"], dataset["Error Flag"]):
        if int(error_flag) == 0:
            label = "No Error"
        elif int(error_flag) == 1:
            label = "Error"
        sample = f"Clinical Note: {text} ### Label: {label}"
        samples.append(sample)
    return samples


def main() -> None:
    train_dataset = Dataset.from_pandas(
        pd.read_csv("data/MEDIQA-CORR-2024-MS+UW-Train.csv", encoding="unicode_escape")
    )
    valid_dataset = Dataset.from_pandas(
        pd.read_csv(
            "data/MEDIQA-CORR-2024-MS-ValidationSet-1-Full.csv",
            encoding="unicode_escape",
        )
    )
    test_dataset = Dataset.from_pandas(
        pd.read_csv(
            "data/March-26-2024-MEDIQA-CORR-Official-Test-Set.csv",
            encoding="unicode_escape",
        )
    )

    concat_sentences_train = preprocessing(train_dataset)
    concat_sentences_valid = preprocessing(valid_dataset)

    train_dataset = train_dataset.add_column("concat_sentences", concat_sentences_train)
    valid_dataset = valid_dataset.add_column("concat_sentences", concat_sentences_valid)

    epochs = 5
    model_name_or_path = "mistralai/Mistral-7B-v0.1"
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    def tokenize_dataset(ds):
        result = tokenizer(ds["concat_sentences"], truncation=True, max_length=1024)
        return result

    train_tokenised = train_dataset.map(tokenize_dataset)
    valid_tokenised = valid_dataset.map(tokenize_dataset)

    print(train_tokenised[:5])

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=[
            "q_proj",
            "o_proj",
            "k_proj",
            "v_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        task_type="CAUSAL_LM",
    )

    args = TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_train_epochs=epochs,
        learning_rate=2e-4,
        bf16=True,
        logging_steps=1,
        output_dir="outputs",
        optim="adamw_hf",
        push_to_hub=True,
        hub_model_id=f"{os.getenv('HF_USERNAME')}/mediqa_binary_classifier",
        hub_token=os.getenv("HF_UPLOAD_TOKEN"),
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_tokenised,
        eval_dataset=valid_tokenised,
        dataset_text_field="concat_sentences",
        args=args,
        peft_config=lora_config,
    )
    trainer.train()


if __name__ == "__main__":
    register_base_configs()
    main()
