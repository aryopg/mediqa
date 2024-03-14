import argparse
import json
import os
import sys

sys.path.append(os.getcwd())

from dotenv import load_dotenv

load_dotenv(".env")
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
    generation_configs,
):
    system_prompt = (
        "You are a clinician reviewing clinical texts that may or may not contain 1 incorrect sentence. "
        "If there is an incorrect sentence, it will be noted in a JSON object with keys 'incorrect_sentence_id' "
        "and 'correction'. If all correct, null JSON object is mentioned. Now you need to provide a reason to your answer."
    )
    json_instruction = "Return the reason in JSON with key 'reason'"
    pos_reason_instruction = (
        "Reason briefly why this correction is more plausible. " + json_instruction
    )
    neg_reason_instruction = (
        "Reason briefly why this clinical note is correct. " + json_instruction
    )

    if label_flag == 1:
        label = json.dumps(
            {"incorrect_sentence_id": label_sentence_id, "correction": label_sentence}
        )
        user_content = f"{text} {label}\n{pos_reason_instruction}"
    else:
        label = json.dumps(None)
        user_content = f"{text} {label}\n{neg_reason_instruction}"

    prompts = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": user_content,
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

    cot_file_path = "data/cot_reasons.json"

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
        "max_tokens": configs.model.configs.max_tokens,
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
                generation_configs,
            )
            update_json_file(cot_file_path, batch["id"][0], cot_reason)
        except Exception as e:
            print(f"Exception: {e}")
            print(f"Failed to generate reason: {batch['text_id'][0]}")


if __name__ == "__main__":
    register_base_configs()
    main()
