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


def clean_string(s):
    if isinstance(s, str):
        return s.replace("\n", " ").replace("\r", " ").strip()
    else:
        return s
    
def clean_string_col(col):
    if col.dtype == 'object':  # Check if the column datatype is 'object' (usually indicates strings)
        return col.str.replace("\n", " ").str.replace("\r", " ").str.strip()
    else:
        return col
    

# Index(['Unnamed: 0', 'Text ID', 'Text', 'Sentences', 'Error Flag',
#     'Error Sentence ID', 'Error Sentence', 'Corrected Sentence',
#     'Corrected Text'],
#     dtype='object')
    


class AnnotatedDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_configs: DataConfigs,
        # prompt_configs: PromptConfigs,
        # trainer_configs: TrainerConfigs,
        split: str,
        **kwargs,
    ):
        
        self.split = split

        self.data_dir = data_configs.data_dir
        self.data_filename = os.path.join(
            self.data_dir, getattr(data_configs, f"{split}_data_filename")
        )

        # Prepare data
        self.df = self.parse_data()

    def parse_data(self) -> dict:

        df = pd.read_csv(self.data_filename)

        # df.fillna("NA", inplace=True)
        str_cols = ['Text', 'Sentences', 'Error Sentence', 'Corrected Sentence', 'Corrected Text']
        df[str_cols].fillna("NA")
        df[str_cols].apply(clean_string_col)

        for col in ['treatments', 'diagnoses', 'exam_interpretation', 'exam_results history']:
            df[col] = df[col].apply(lambda x: [x] if isinstance(x, str) else ["NA"] if pd.isna(x) else x)

        return df
    
    def __getitem__(self, idx):

        return self.df.iloc[idx].to_dict()



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
        self.trainer_configs = trainer_configs.configs

        self.split = split

        filename = f"{split}_data_filename" if self.trainer_configs=='annotate' else f"{split}_annotated_data_filename"
        self.data_filename = getattr(data_configs, filename) 
        print(f"self.data_filename in MEDIQADataset: {self.data_filename}")

        # Prepare data
        self.df = self.parse_data()

        self.prompt_template = self.prompt_configs.prompt_template

    def parse_data(self) -> dict:

        print(f"self.data_filename in MEDIQADataset parse_data: {self.data_filename}")
        df = pd.read_csv(self.data_filename)

        str_cols = ['Text', 'Sentences', 'Error Sentence', 'Corrected Sentence', 'Corrected Text']
        df[str_cols].fillna("NA")
        df[str_cols].apply(clean_string_col)

        return df

    def __getitem__(self, idx):

        return self.df.iloc[idx].to_dict()


    def __len__(self):
        return self.df.shape[0]
