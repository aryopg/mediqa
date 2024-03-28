
from typing import List, Tuple

import torch
import json

# import numpy as np
import pandas as pd

# import torch

class MEDIQADataset(torch.utils.data.Dataset):
    def __init__(
        self,
    ):
        # input_file_name='/home/co-chae/mediqa/span_prediction/data/for_gpt_correction/test_set_for_gpt_correction.csv'
        input_file_name='/home/co-chae/mediqa/span_prediction/data/for_gpt_correction/test_set_for_gpt_correction_ms.csv'
        self.df = pd.read_csv(input_file_name, index_col=False) #, encoding='MacRoman')
        
        # self.df = pd.read_csv(input_file_name, index_col=False, nrows=20)
        # import pdb; pdb.set_trace()
        
        with open('/home/co-chae/mediqa/correction/test_pairs_no_pairs.json', encoding='utf-8') as f:
            pairs = json.load(f)
            
        # import pdb; pdb.set_trace()
        
        self.pairs = pairs["pairs"]
        

    def __getitem__(self, idx):
        # import pdb; pdb.set_trace()
        # return self.df.iloc[idx].to_dict()
        try:
            pair = self.pairs[idx]
        except:
            print("%*(#$&$(#%*$(#&%*($@*$)#@*)$*#@)$*@#)(*$%(&*%($#*%)$*#@$)@#($)#*@)*$%()")
            pair = self.pairs[0]
        
        print(pair)
        
        row1 = self.df[self.df['Text ID'] == pair[0]]
        row2 = self.df[self.df['Text ID'] == pair[1]]
        
        if not row1.empty:
            res1 = row1.iloc[0].to_dict()
        else:
            res1 = {}  # Or some default value

        if not row2.empty:
            res2 = row2.iloc[0].to_dict()
        else:
            res2 = {}
            
        if res1 == {} or res2 == {}:
            import pdb; pdb.set_trace()
            
        
        
        
        # res1, res2 = row1.iloc[0].to_dict(), row2.iloc[0].to_dict()
        
        # import pdb; pdb.set_trace()
        
        # self.df.iloc[idx].to_dict()
        return (res1, res2)

    def __len__(self):
        # return int(self.df.shape[0]/2)
        return len(self.pairs)
    
    
# python correction/api_main_ms_opts.py