import ast
import json
import math
import os
import pdb
import time

import huggingface_hub
import hydra
import pandas as pd
import torch
import yaml
from accelerate import Accelerator
from accelerate.tracking import WandBTracker
from huggingface_hub import HfApi
from omegaconf import OmegaConf
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb

from .api_pipeline import APIPipeline
from .configs import TrainingConfigs
from .dataset import MEDIQADataset
from .metrics import NLGMetrics
from .pipeline import ModelPipeline


def dicstr_to_dic(dicstr):
    return ast.literal_eval(dicstr.replace("\n", ""))


def dic_of_list_as_single_value(dic):
    for key in dic:
        dic[key] = [' $ '.join(dic[key])] if isinstance(dic[key], list) else [dic[key]]
    return dic

class APITrainer: 
    def __init__(self, configs: TrainingConfigs):

        self.configs = configs
        self.hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
        self.output_dir = self.hydra_cfg["runtime"]["output_dir"]
       
        print(f"output_dir = {self.output_dir}")

        self.api_key = self._get_api_key()

        self.dataloaders = self._load_dataloaders()
        self.pipeline = self._load_chat_pipeline()

    def _get_api_key(self) -> str:

        key_path = self.configs.api_key_path

        with open(key_path, 'r') as file:
            key_dict = yaml.safe_load(file)
            api_key = key_dict['openai_api_key']

        return api_key
    


    def _load_dataloaders(self) -> dict:
        dataloaders = {}

        # Convert data into datasets
        for split in ["train"]:#, "valid", "test"]:
            print(f"Setup {split} data loader")
            dataset = MEDIQADataset(
                self.configs.data,
                self.configs.prompt,
                self.configs.trainer,
                split=split,
            )
            dataloaders[split] = DataLoader(
                dataset,
                shuffle=True,
                **self.configs.data.data_loader_configs,
            )

        return dataloaders

    def _load_chat_pipeline(self): 
        return APIPipeline(self.configs.model, self.api_key)

    @staticmethod
    def compute_metrics(prediction, batch):
        """
        TO DO!

        prediction: API output converted to JSON object. 
        """

        # prediction as of now is 

        return {}

    def train(self):
        pass

    def test(self, split: str, log_metrics: bool = True, mode='annotate'):

        print(f"Testing on {split}")

        prev_df = pd.DataFrame()


        for step, batch in enumerate(self.dataloaders[split]): 

            predictions, prompts = self.pipeline.chat(batch, apply_template=True, mode=mode)

            batch_df = pd.DataFrame(batch)

            if mode=='annotate':

                predictions = dicstr_to_dic(predictions[0])

                res_dict = {
                    'prompt': [prompts[0][0]], # ValueError: If using all scalar values, you must pass an index
                    'sys_prompt': [prompts[0][1]],
                }

                predictions, res_dict = dic_of_list_as_single_value(predictions), res_dict
                res_df = pd.concat([pd.DataFrame(res_dict), pd.DataFrame(predictions)], axis=1)#, ignore_index=True)
           
            elif mode=='detect':
                
                predictions = [dicstr_to_dic(prediction) for prediction in predictions]

                res_dict = {}
                for pred_idx, pred_dic in enumerate(predictions):

                    res_dict[f'Prompt for {prompts[pred_idx][0]}'] = [prompts[pred_idx][1]]
                    res_dict[f'Sys Prompt for {prompts[pred_idx][0]}'] = [prompts[pred_idx][2]]

                    for key, val in pred_dic.items(): # sts reasoning final answer
                        colname = f"{key} for {prompts[pred_idx][0]}" # for treatment 1
                        res_dict[colname] = [val] if isinstance(val, str) else [" ".join(val)]

                res_df = pd.DataFrame(res_dict)

            res_df = pd.concat([batch_df, res_df], axis=1)#, ignore_index=True) 
            res_df = pd.concat([prev_df, res_df], ignore_index=True)
            prev_df = res_df
       
            res_df.to_csv(os.path.join(self.output_dir, f"{mode}_output_{split}.csv"), index=False)

            print(f"CSV file saved after step!")

            if step % 20 == 0:
                
                print(f"step = {step}")
                print("Sleeping for 30 seconds after 20 generations...")
                print()
                time.sleep(30)  # Sleep for 30 seconds


            num_test_steps = 1000
      
            if step >= num_test_steps:
                print(f"break statement in Training loop after {step} steps!")
                break
