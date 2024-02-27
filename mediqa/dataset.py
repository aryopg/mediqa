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

        self.mode = self.trainer_configs.mode
        self.split = split

        filename = f"{self.mode}_data_{self.split}"

        self.data_filename = getattr(data_configs, filename) 

        self.df = self.parse_data()

    def parse_data(self) -> dict:

        df = pd.read_csv(self.data_filename)

        str_cols = ['Text', 'Sentences', 'Error Sentence', 'Corrected Sentence', 'Corrected Text']
        df[str_cols].fillna("NA")
        df[str_cols].apply(clean_string_col)

        return df

    def __getitem__(self, idx):

        return self.df.iloc[idx].to_dict()


    def __len__(self):
        return self.df.shape[0]
