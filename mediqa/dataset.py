import json
import math
import os
import re
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from transformers import PreTrainedTokenizer

from mediqa.configs import DataConfigs, PromptConfigs, RetrieverConfigs, TrainerConfigs

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
        retriever_configs: RetrieverConfigs,
        split: str,
        **kwargs,
    ):
        self.data_configs = data_configs
        self.prompt_configs = prompt_configs
        self.trainer_configs = trainer_configs
        self.retriever_configs = retriever_configs

        self.split = split

        self.data_dir = data_configs.data_dir
        self.data_filename = os.path.join(
            self.data_dir, getattr(data_configs, f"{split}_data_filename")
        )

        # Prepare data
        ## If ICL run, set ICL examples
        self.num_in_context_examples = (
            self.trainer_configs.configs.num_in_context_examples
        )
        self.icl_corpus, self.icl_examples = self._set_icl_examples()

        ## If CoT run, set CoT labels
        self.cot_prompt = self.trainer_configs.configs.cot_prompt
        self.cot_reasons = self._set_cot_reasons()

        ## Parse data
        self.data = self.parse_data()

        self.prompt_template = self.prompt_configs.prompt_template

    def _set_icl_examples(self):
        if self.retriever_configs is None or self.num_in_context_examples == 0:
            return None, None

        icl_corpus = pd.read_csv(
            os.path.join(self.data_dir, self.retriever_configs.knowledge_corpus)
        )
        icl_examples = pd.read_json(
            os.path.join(
                self.retriever_configs.icl_examples_dir,
                self.split + ".json",
            )
        )

        return icl_corpus, icl_examples

    def _set_cot_reasons(self):
        if not self.cot_prompt:
            return {}

        if os.path.isfile(self.trainer_configs.configs.cot_reasons_filepath):
            with open(self.trainer_configs.configs.cot_reasons_filepath, "r") as f:
                return json.load(f)
        else:
            print(
                f"No CoT reasons file found: {self.trainer_configs.configs.cot_reasons_filepath}."
            )
            return {}

    def get_icl_examples_by_id(self, text_id: str):
        # Get the ICL examples depending on the number of ICL examples allowed
        sample_icl_examples = self.icl_examples[text_id]

        selected_icl_examples = []

        # Choose the top-k/2 of both the positive and negative examples
        top_k_positives = [
            (example["id"], example["score"])
            for example in sample_icl_examples["pos"][
                : int(self.num_in_context_examples / 2)
            ]
        ]
        top_k_negatives = [
            (example["id"], example["score"])
            for example in sample_icl_examples["neg"][
                : int(self.num_in_context_examples / 2)
            ]
        ]

        selected_icl_examples += top_k_positives + top_k_negatives

        # sorted by their scores, highest to lowest. Take only the ids
        selected_icl_examples = [
            example[0]
            for example in sorted(
                selected_icl_examples, key=lambda x: x[1], reverse=True
            )
        ]

        return selected_icl_examples

    @staticmethod
    def _preprocess_sentence(sentence: str) -> str:
        if not isinstance(sentence, str):
            if type(sentence) == float:
                sentence = "NA"
            else:
                sentence = str(sentence)
                sentence = sentence.replace("\n", " ").replace("\r", " ").strip()
        return sentence

    @staticmethod
    def split_sentences(sentences: str) -> List[str]:
        # Split the sentences into lines
        lines = sentences.strip().split("\n")

        # Placeholder for the processed list
        processed_list = []

        # Temporary storage for building up sentences or data points
        temp_sentence = ""

        # Iterate over each line
        for line in lines:
            # Check if the line starts with an index (indicating a new sentence or data point)
            if re.match(r"^\d+", line):
                if temp_sentence:  # If there's a sentence built up, add it to the list
                    processed_list.append(temp_sentence.strip())
                temp_sentence = line  # Start a new sentence/data point
            else:
                temp_sentence += (
                    " " + line
                )  # Continue building up the current sentence/data point

        # Add the last built-up sentence/data point to the list, if any
        if temp_sentence:
            processed_list.append(temp_sentence.strip())

        return processed_list

    def parse_data(self) -> dict:
        """Parsing reference file path."""
        df = pd.read_csv(self.data_filename)

        text_ids = df["Text ID"].tolist()

        texts = []
        sentences = []
        split_sentences = []
        label_flags = []
        label_sentences = []
        label_sentence_ids = []

        # Handle UW and MS datasets difference
        if "Error Flag" in df.columns:
            error_flag_column = "Error Flag"
        else:
            error_flag_column = "Error_flag"

        icl_examples = []
        for _, row in df.iterrows():
            texts += [str(row["Text"])]
            sentences += [str(row["Sentences"])]
            split_sentences += [self.split_sentences(str(row["Sentences"]))]
            corrected_sentence = self._preprocess_sentence(row["Corrected Sentence"])

            label_flags += [str(row[error_flag_column])]
            label_sentences += [corrected_sentence]
            label_sentence_ids += [str(row["Error Sentence ID"])]

            if self.icl_corpus is not None and self.icl_examples is not None:
                sample_icl_example_ids = self.get_icl_examples_by_id(row["Text ID"])
                sample_icl_examples = []
                for sample_icl_example_id in sample_icl_example_ids:
                    sample_icl_example = self.icl_corpus.loc[
                        self.icl_corpus["Text ID"] == sample_icl_example_id
                    ]
                    sample_icl_example = {
                        "sentences": sample_icl_example["Sentences"].tolist()[0],
                        "label_flags": str(
                            sample_icl_example[error_flag_column].tolist()[0]
                        ),
                        "label_sentences": self._preprocess_sentence(
                            sample_icl_example["Corrected Sentence"].tolist()[0]
                        ),
                        "label_sentence_ids": str(
                            sample_icl_example["Error Sentence ID"].tolist()[0]
                        ),
                        "label_reason": "",
                    }
                    if self.cot_prompt and self.cot_reasons:
                        sample_icl_example["label_reason"] = self.cot_reasons[
                            sample_icl_example_id
                        ]["reason"]
                    sample_icl_examples += [sample_icl_example]
            else:
                sample_icl_examples = []

            icl_examples += [sample_icl_examples]

        return {
            "ids": text_ids,
            "icl_examples": icl_examples,
            "texts": texts,
            "sentences": sentences,
            "split_sentences": split_sentences,
            "label_flags": label_flags,
            "label_sentences": label_sentences,
            "label_sentence_ids": label_sentence_ids,
        }

    def __getitem__(self, idx):
        icl_texts = []
        icl_labels = []
        if self.num_in_context_examples > 0:
            for icl_example in self.data["icl_examples"][idx]:
                # print(icl_example)
                icl_texts += [
                    self.prompt_template.format(
                        clinical_sentences=icl_example["sentences"],
                        cot_prompt=self.cot_prompt,
                    )
                ]

                icl_labels += [
                    {
                        "label_flags": int(icl_example["label_flags"]),
                        "label_sentences": icl_example["label_sentences"],
                        "label_sentence_ids": icl_example["label_sentence_ids"],
                        "label_reason": icl_example["label_reason"],
                    }
                ]

        return {
            "id": self.data["ids"][idx],
            "texts": self.data["texts"][idx],
            "sentences": self.data["sentences"][idx],
            "split_sentences": self.data["split_sentences"][idx],
            "icl_texts": icl_texts,
            "icl_labels": icl_labels,
            "prompted_text": self.prompt_template.format(
                clinical_sentences=self.data["sentences"][idx],
                cot_prompt=self.cot_prompt,
            ),
            "label_flags": int(self.data["label_flags"][idx]),
            "label_sentences": self.data["label_sentences"][idx],
            "label_sentence_ids": self.data["label_sentence_ids"][idx],
        }

    def __len__(self):
        return len(self.data["ids"])
