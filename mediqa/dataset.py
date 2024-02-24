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
        icl_texts = []
        icl_labels = []
        if self.trainer_configs.configs.num_in_context_examples > 0:
            if self.trainer_configs.configs.num_in_context_examples % 2 == 0:
                # If number of ICL examples is divisible by 2, provide half positive and half negative examples
                # (this can be optimised, but future aryo's problem :p)
                pos_icl_candidates = []
                neg_icl_candidates = []
                for i, label_flag in enumerate(self.data["label_flags"]):
                    if i != idx:
                        if int(label_flag) == 1:
                            pos_icl_candidates += [i]
                        elif int(label_flag) == 0:
                            neg_icl_candidates += [i]
                pos_icl_example_ids = np.random.choice(
                    pos_icl_candidates,
                    self.trainer_configs.configs.num_in_context_examples // 2,
                )
                neg_icl_example_ids = np.random.choice(
                    neg_icl_candidates,
                    self.trainer_configs.configs.num_in_context_examples // 2,
                )

                for icl_example_id in pos_icl_example_ids:
                    icl_texts += [
                        self.prompt_template.format(
                            clinical_sentences=self.data["original_texts"][
                                icl_example_id
                            ],
                            cot_prompt="",
                        )
                    ]

                    icl_labels += [
                        {
                            "label_flags": int(
                                self.data["label_flags"][icl_example_id]
                            ),
                            "label_sentences": self.data["label_sentences"][
                                icl_example_id
                            ],
                            "label_sentence_ids": self.data["label_sentence_ids"][
                                icl_example_id
                            ],
                        }
                    ]

                for icl_example_id in neg_icl_example_ids:
                    icl_texts += [
                        self.prompt_template.format(
                            clinical_sentences=self.data["original_texts"][
                                icl_example_id
                            ],
                            cot_prompt="",
                        )
                    ]
                    icl_labels += [
                        {
                            "label_flags": int(
                                self.data["label_flags"][icl_example_id]
                            ),
                            "label_sentences": self.data["label_sentences"][
                                icl_example_id
                            ],
                            "label_sentence_ids": self.data["label_sentence_ids"][
                                icl_example_id
                            ],
                        }
                    ]
            else:
                # Randomly sample in-context examples whose id is different from the current one
                icl_candidates = [
                    i for i in range(len(self.data["text_ids"])) if i != idx
                ]
                icl_example_ids = np.random.choice(
                    icl_candidates, self.trainer_configs.configs.num_in_context_examples
                )

                for icl_example_id in icl_example_ids:
                    icl_texts += [
                        self.prompt_template.format(
                            clinical_sentences=self.data["original_texts"][
                                icl_example_id
                            ],
                            cot_prompt="",
                        )
                    ]
                    icl_labels += [
                        {
                            "label_flags": int(
                                self.data["label_flags"][icl_example_id]
                            ),
                            "label_sentences": self.data["label_sentences"][
                                icl_example_id
                            ],
                            "label_sentence_ids": self.data["label_sentence_ids"][
                                icl_example_id
                            ],
                        }
                    ]

        return {
            "text_id": self.data["text_ids"][idx],
            "original_text": self.data["original_texts"][idx],
            "icl_texts": icl_texts,
            "icl_labels": icl_labels,
            "prompted_text": self.prompt_template.format(
                clinical_sentences=self.data["original_texts"][idx],
                cot_prompt="",
            ),
            "label_flags": int(self.data["label_flags"][idx]),
            "label_sentences": self.data["label_sentences"][idx],
            "label_sentence_ids": self.data["label_sentence_ids"][idx],
        }

    def __len__(self):
        return len(self.data["text_ids"])
