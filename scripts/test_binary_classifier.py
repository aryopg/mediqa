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
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
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
    for text in dataset["Text"]:
        sample = f"Clinical Note: {text} ### Label: "
        samples.append(sample)
    return samples


def main() -> None:
    test_dataset = Dataset.from_pandas(
        pd.read_csv(
            "data/March-26-2024-MEDIQA-CORR-Official-Test-Set.csv",
            encoding="unicode_escape",
        )
    )
    concat_sentences_test = preprocessing(test_dataset)
    test_dataset = test_dataset.add_column("concat_sentences", concat_sentences_test)

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

    test_tokenised = test_dataset.map(tokenize_dataset)

    peft_model_id = f"aryopg/mediqa_binary_classifier"
    model = PeftModel.from_pretrained(model, peft_model_id, device_map="auto")

    for test_sample_tokenised in test_tokenised:
        outputs = model.generate(**test_sample_tokenised, max_new_tokens=8)
        print(tokenizer.decode(outputs[0], skip_special_tokens=True))


if __name__ == "__main__":
    register_base_configs()
    main()
