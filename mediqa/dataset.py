import json
import math
import os
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from transformers import PreTrainedTokenizer

from mediqa.configs import DataConfigs, PromptConfigs, TrainerConfigs

TASK2ID = {
    "Results": 0,
    "Intervention": 1,
    "Eligibility": 2,
    "Adverse Events": 3,
}


class MEDIQADataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_configs: DataConfigs,
        prompt_configs: PromptConfigs,
        trainer_configs: TrainerConfigs,
        split: str,
        **kwargs,
    ):
        self.data_configs = data_configs
        self.prompt_configs = prompt_configs
        self.trainer_configs = trainer_configs

        self.split = split

        self.data_dir = data_configs.data_dir
        self.data_filename = os.path.join(
            self.data_dir, getattr(data_configs, f"{split}_data_filename")
        )

        # Prepare data
        self.data = self.parse_data()

        self.prompt_template = self.prompt_configs.prompt_template

    def parse_data(self) -> dict:
        """Parsing reference file path."""
        df = pd.read_csv(self.data_filename)

        text_ids = df["Text ID"].tolist()

        original_texts = []
        label_flags = []
        label_sentences = []
        label_sentence_ids = []

        # Handle UW and MS datasets difference
        if "Error Flag" in df.columns:
            error_flag_column = "Error Flag"
        else:
            error_flag_column = "Error_flag"

        for _, row in df.iterrows():
            original_texts += [str(row["Sentences"])]
            corrected_sentence = row["Corrected Sentence"]

            if not isinstance(corrected_sentence, str):
                if math.isnan(corrected_sentence):
                    corrected_sentence = "NA"
                else:
                    corrected_sentence = str(corrected_sentence)
                    corrected_sentence = (
                        corrected_sentence.replace("\n", " ").replace("\r", " ").strip()
                    )

            label_flags += [str(row[error_flag_column])]
            label_sentences += [corrected_sentence]
            label_sentence_ids += [str(row["Error Sentence ID"])]

        return {
            "text_ids": text_ids,
            "original_texts": original_texts,
            "label_flags": label_flags,
            "label_sentences": label_sentences,
            "label_sentence_ids": label_sentence_ids,
        }

    def __getitem__(self, idx):
        return {
            "text_id": self.data["text_ids"][idx],
            "original_text": self.data["original_texts"][idx],
            "prompted_text": self.prompt_template.format(
                icl_example="",
                clinical_sentences=self.data["original_texts"][idx],
                cot_prompt="",
            ),
            "label_flags": self.data["label_flags"][idx],
            "label_sentences": self.data["label_sentences"][idx],
            "label_sentence_ids": self.data["label_sentence_ids"][idx],
        }

    def __len__(self):
        return len(self.data["text_ids"])
