import torch
import numpy as np
import pandas as pd

class MEDIQADataset(torch.utils.data.Dataset):
    def __init__(
        self,
        input_file
    ):
        input_file_name=input_file 
        df = pd.read_csv(input_file_name, index_col=False) #, encoding='MacRoman')
        self.df = df.head(5)
        
        

    def __getitem__(self, idx):
        return self.df.iloc[idx].to_dict()

    def __len__(self):
        return self.df.shape[0]