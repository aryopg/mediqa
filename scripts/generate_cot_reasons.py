import argparse
import json
import logging
import os
import sys

sys.path.append(os.getcwd())

from dotenv import load_dotenv

load_dotenv(".env")
from enum import Enum
from typing import List

import hydra
import pandas as pd
from omegaconf import OmegaConf
from openai import OpenAI
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from mediqa.configs import (
    DataConfigs,
    PromptConfigs,
    TrainerConfigs,
    TrainingConfigs,
    register_base_configs,
)
from mediqa.dataset import MEDIQADataset

logging.getLogger("httpx").setLevel(logging.WARNING)


class ReasonInstructions(Enum):
    brief = (
        "Please present a brief reasoning that leads to the groundtruth JSON answer provided. "
        "Return the reason in a JSON format with key 'reason'\n"
        "Reason: "
    )
    long = (
        "Please present a step-by-step reasoning that leads to the groundtruth JSON answer provided. "
        "Include any considerations, medical standards, or guidelines that you deem as helpful in your assessment. "
        "Return the reason in a JSON format with key 'reason'\n"
        "Reason: "
    )
    soap = (
        "Please present a step-by-step reasoning that leads to the groundtruth JSON answer provided. Include any considerations, medical standards, or guidelines that you deem as helpful in your assessment.\n"
        "You may follow this reasoning steps:\n"
        "First, organise the clinical note into a structured SOAP format (Subjective, Objective, Assessment, Plan).\n"
        "Second, identify an inconsistency in the collection of clinical facts, if any.\n"
        "Lastly, present the reasoning that leads to the groundtruth JSON answer provided given the SOAP-structured clinical facts\n"
        "Return the reason as contiguous text in a JSON format with key 'reason'\n"
        "Reason: "
    )


def update_json_file(file_path, key, value):
    """Update the JSON file with the new key-value pair."""
    try:
        # Attempt to read the existing data
        with open(file_path, "r") as file:
            data = json.load(file)
    except FileNotFoundError:
        # If the file does not exist, start with an empty dictionary
        data = {}

    # Update the data with the new key-value pair
    data[key] = value

    # Write the updated data back to the file
    with open(file_path, "w") as file:
        json.dump(data, file)


def generate_cot_reasons(
    client,
    model_name,
    text,
    label_flag,
    label_sentence,
    label_sentence_id,
    trainer_name,
    generation_configs,
):
    system_prompt = (
        "You are a clinician tasked with reviewing clinical texts, each of which may contain either one incorrect sentence due to clinical or factual inaccuracies, or no errors at all."
        "You are also presented with a JSON object containing the index of the inaccurate sentence (if any) and the correction with the following structure:\n\n"
        "- 'incorrect_sentence_id': If there is an incorrect sentence, its ID is here. If all sentences are correct, it will be -1.\n"
        "- 'correction': If an incorrect sentence is identified, a corrected sentence is mentioned here. If all sentences are correct, 'NA' is mentioned here instead.\n\n"
        "When evaluating the text, focus specifically on clinical or factual inaccuracies. "
        "This could include incorrect medical information, factual errors related to patient care, or erroneous data interpretations. "
    )
    if trainer_name.startswith("cot_soap"):
        reason_instruction = ReasonInstructions.soap.value
    elif trainer_name.startswith("cot_long"):
        reason_instruction = ReasonInstructions.long.value
    elif trainer_name.startswith("cot_brief"):
        reason_instruction = ReasonInstructions.brief.value
    else:
        raise NotImplementedError

    if label_flag == 1:
        label = json.dumps(
            {"incorrect_sentence_id": label_sentence_id, "correction": label_sentence}
        )
    else:
        label = json.dumps({"incorrect_sentence_id": -1, "correction": "NA"})

    prompts = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": f"{text} {label}\n{reason_instruction}",
        },
    ]

    response = client.chat.completions.create(
        model=model_name, messages=prompts, **generation_configs
    )

    cot_reason = response.choices[0].message.content

    return cot_reason


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(configs: TrainingConfigs) -> None:
    missing_keys: set[str] = OmegaConf.missing_keys(configs)
    if missing_keys:
        raise RuntimeError(f"Got missing keys in config:\n{missing_keys}")

    cot_file_path = configs.trainer.configs.cot_reasons_filepath

    # OpenAI client
    model_name = configs.model.configs.model_name_or_path
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY", ""),
    )

    generation_configs = {
        "temperature": configs.model.configs.temperature,
        "top_p": configs.model.configs.top_p,
        "frequency_penalty": configs.model.configs.frequency_penalty,
        "presence_penalty": configs.model.configs.presence_penalty,
        "max_tokens": 512,
    }

    dataset = MEDIQADataset(
        configs.data,
        configs.prompt,
        configs.trainer,
        None,
        split="train",
    )
    dataloader = DataLoader(
        dataset,
        shuffle=True,
        **configs.data.data_loader_configs,
    )

    try:
        # Attempt to read the existing data
        with open(cot_file_path, "r") as file:
            data = json.load(file)
        train_samples_w_cot = set([key for key in list(data.keys())])
    except FileNotFoundError:
        train_samples_w_cot = set()

    print(f"Existing reasons found: {len(train_samples_w_cot)}/{len(dataloader)}")

    for batch in tqdm(dataloader):
        # Skip if already processed
        if batch["id"][0] in train_samples_w_cot:
            continue
        try:
            cot_reason = generate_cot_reasons(
                client,
                model_name,
                batch["prompted_text"][0],
                batch["label_flags"][0],
                batch["label_sentences"][0],
                batch["label_sentence_ids"][0],
                configs.trainer.name,
                generation_configs,
            )
            update_json_file(cot_file_path, batch["id"][0], cot_reason)
        except Exception as e:
            print(f"Exception: {e}")
            print(f"Failed to generate reason: {batch['id'][0]}")


if __name__ == "__main__":
    register_base_configs()
    main()
