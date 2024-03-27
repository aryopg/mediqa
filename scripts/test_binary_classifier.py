import logging
import os
import sys

sys.path.append(os.getcwd())

from dotenv import load_dotenv

load_dotenv(".env")

import pickle

import huggingface_hub
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
        sample = f"Clinical Note: {text}\n\nQuestion: Does this clinical note contain a clinical error?\nAnswer: "
        samples.append(sample)
    return samples


def main() -> None:
    huggingface_hub.login(token=os.getenv("HF_DOWNLOAD_TOKEN", ""))

    test_dataset = Dataset.from_pandas(
        pd.read_csv(
            "data/March-26-2024-MEDIQA-CORR-Official-Test-Set.csv",
            # "data/MEDIQA-CORR-2024-MS-ValidationSet-1-Full.csv",
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
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    def tokenize_dataset(ds):
        result = tokenizer(ds["concat_sentences"], truncation=True, max_length=1024)
        return result

    test_tokenised = test_dataset.map(tokenize_dataset)

    peft_model_id = f"aryopg/mediqa_binary_classifier"
    model = PeftModel.from_pretrained(model, peft_model_id, device_map="auto")

    preds = {}
    for test_sample_tokenised in test_tokenised:
        input_ids = torch.tensor([test_sample_tokenised["input_ids"]])
        attention_mask = torch.tensor([test_sample_tokenised["attention_mask"]])
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=1,
        )
        pred = tokenizer.decode(
            outputs[0, input_ids.size(1) :], skip_special_tokens=True
        )
        print(f"{test_sample_tokenised['Text ID']} {pred}")
        if "Yes" in pred:
            pred = 1
        elif "No" in pred:
            pred = 0
        else:
            print(f">>>> {pred} is not recognised. Reverting to 0")
            pred = 0
        preds[test_sample_tokenised["Text ID"]] = pred

    if "Error Flag" in test_dataset.column_names:
        reference_flags = {
            text_id: flag
            for text_id, flag in zip(
                test_dataset["Text ID"], test_dataset["Error Flag"]
            )
        }
        for text_id in reference_flags:
            if (
                text_id in predicted_flags
                and reference_flags[text_id] == predicted_flags[text_id]
            ):
                matching_flags_nb += 1

        flags_accuracy = matching_flags_nb / len(reference_flags)
        print(flags_accuracy)


if __name__ == "__main__":
    register_base_configs()
    main()
