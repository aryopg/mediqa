
from typing import List, Tuple

import torch

# import numpy as np
import pandas as pd

# import torch

class MEDIQADataset(torch.utils.data.Dataset):
    def __init__(
        self,
    ):
        input_file_name='/home/co-chae/mediqa/span_prediction/data/for_gpt_correction/test_set_for_gpt_correction.csv'
        self.df = pd.read_csv(input_file_name, index_col=False) #, encoding='MacRoman')
        # import pdb; pdb.set_trace()
        

    def __getitem__(self, idx):
        # import pdb; pdb.set_trace()
        return self.df.iloc[idx].to_dict()

    def __len__(self):
        return self.df.shape[0]