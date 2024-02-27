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

from ._metrics import *
from .api_pipeline import APIPipeline
from .configs import TrainingConfigs
from .dataset import MEDIQADataset
from .metrics import NLGMetrics
from .pipeline import ModelPipeline
from .utils.wandb_logger import WandbLogger

# add error-handling topk_dicts = [json.loads(json_str) for json_str in resp] 
# do i need to add keys? luckily i don't have to

# import pandas as pd

# dic1 = {'key1': ['dic1_1'], 'key2': ['2'], 'key3': ['3'], 'key4': ['4'], 'key5': ['5']}
# dic2 = {'key2': ['dic2_2'], 'key3': ['dic2_3']}

# concat_df = pd.concat([pd.DataFrame(dic1), pd.DataFrame(dic2)])

# print(concat_df)

#      key1    key2    key3 key4 key5
# 0  dic1_1       2       3    4    5
# 0     NaN  dic2_2  dic2_3  NaN  NaN
# okay but if dic2 is completely empty ({}) a new row isn't appended. is this a problem? maybe not cos there will always be the prompt keys. but make sure you handle NaNs accordingly. 
def dicstr_to_dic(dicstr):
    try:
        return ast.literal_eval(dicstr.replace("\n", "")) # json.loads(json_str)
    except:
        return {}


def stringify_dict_values(dic):
    for key in dic:
        dic[key] = [' $ '.join(dic[key])] if isinstance(dic[key], list) else [dic[key]]
    return dic

class APITrainer: 
    def __init__(self, configs: TrainingConfigs):

        self.configs = configs
        self.hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
        self.output_dir = self.hydra_cfg["runtime"]["output_dir"]
        self.output_filename = self.configs.data.save_args.output_filename # "{split}_output_{mode}.csv"
       
        print(f"output_dir = {self.output_dir}")

        self.api_key = self._get_api_key()
        self.mode = self.configs.trainer.configs.mode

        self.dataloaders = self._load_dataloaders()
        self.pipeline = self._load_chat_pipeline()

        # self.mode = self.configs.trainer.configs.mode

        if not configs.debug:
            self._setup_logger()

    def _setup_logger(self, logger='wandb'):

        # wandb_dict_step = {'eval/step': batch_idx, 'eval/loss/step': batch_loss}
        # self.logger.info(wandb_dict_step)

        if logger=='wandb':
            self.logger = WandbLogger(self.configs.trainer.wanb_args)

    def _get_api_key(self) -> str:

        key_path = self.configs.api_key_path

        with open(key_path, 'r') as file:
            key_dict = yaml.safe_load(file)
            api_key = key_dict['openai_api_key']

        return api_key
    


    def _load_dataloaders(self) -> dict:
        dataloaders = {}

        # Convert data into datasets
        for split in ["valid"]:#, "valid", "test"]:
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
        return APIPipeline(self.configs.model, self.configs.prompt, self.api_key, mode=self.mode)

    @staticmethod
    def compute_metrics(prediction, batch):
        """
        TO DO!

        prediction: API output converted to JSON object. 
        """

        # prediction as of now is 

        return {}
    
    def log_eval_metrics(self, mode=None): # mode as in evaluate the output csv file from mode ~. 

        if not mode:
            mode = self.mode

        # if self.mode=="prompt_identify_n_error_sentences":
        #     metric_name = ... # log to wandb
        #     eval_result = evaluate_error_detection(pd.read_csv(self.configs.data.eval_args.ismodeisrelevant)) # this path also has to come from data yaml

        # if mode=="binary":
        #     eval_result = binary_classification_acc()
        # elif "identify":
        #     error_sent_identification_acc

        wandb_dict = {}
        wandb_dict['eval/epoch'] = 1 # x-axis
        # run.define_metric('eval/step') # x-axis
        wandb_dict['eval/metrics/epoch'] = eval_result

        self.logger.info(wandb_dict)



    def train(self):
        pass

    def test(self, split: str, log_metrics: bool = True):

        print(f"Testing on {split}")

        prev_df = pd.DataFrame()




        for step, batch in enumerate(tqdm(self.dataloaders[split])): 

            predictions, prompts = self.pipeline.chat(batch)

            batch_df = pd.DataFrame(batch)

            ## 'identify_n_error_sentences': # originally 'detect':
            if self.mode=='identify_n_error_sentences': # has multiple predictions corresponding to each categories and sub-sentence. one pred is one sub-phrase
                
                predictions = [dicstr_to_dic(pred) for pred in predictions]

                save_dict = {}
                for pred_idx, pred_dic in enumerate(predictions): # each pred_dic is 

                    # prompts.append((key, prompt, sys_prompt))
                    key, prompt, sys_prompt = prompts[pred_idx][0], prompts[pred_idx][1], prompts[pred_idx][2]

                    save_dict[f'Prompt for {key}'] = [prompt] ### 1 
                    save_dict[f'Sys Prompt for {key}'] = [sys_prompt] ### 2

                    for reason_or_answer, val in pred_dic.items(): # sts reasoning final answer
                 
                        colname = f"{reason_or_answer} for {key}" # for treatment 1 ### 0
                        save_dict[colname] = [val] if isinstance(val, str) else [" ".join(val)]

                result_df = pd.DataFrame(save_dict)

            else: # for all except detect!

                ## dict to be turned into df
                ## why the weird fucking indexing? ah in the list of tuples of (prompt, sys_prompt) but has single element for all that's not detect
                prompts = {
                    'Prompt for Error Sentence Detection': [prompts[0][0]], # ValueError: If using all scalar values, you must pass an index
                    'Sys Prompt for Error Sentence Detection': [prompts[0][1]],
                }

                # ah this is where the dollar sign happens. is this necessary? i want to s
                predictions = stringify_dict_values(dicstr_to_dic(predictions[0]))

                # result_dict = {**prompts, **predictions}
                result_df = pd.concat([pd.DataFrame(prompts), pd.DataFrame(predictions)], axis=1) # horizontal
                # result_df = pd.DataFrame(result_dict)

            # prev_df, result_df = prev_df.reset_index(drop=True), result_df.reset_index(drop=True)
            result_df = pd.concat([batch_df, result_df], axis=1)#, ignore_index=True)
            result_df = pd.concat([prev_df, result_df], ignore_index=True)
            prev_df = result_df
       
            result_df.to_csv(os.path.join(self.output_dir, self.output_filename.format(split=split, mode=self.mode)), index=False)

            rest_for, after = 5, 50
            if step % 20 == 0:
                
                print(f"step = {step}")
                print(f"Sleeping for {rest_for} seconds after {after} generations...")
                time.sleep(rest_for) 

            # num_test_steps = 1000
      
            # if step >= num_test_steps:
            #     print(f"break statement in Training loop after {step} steps!")
            #     break
                
        self.log_eval_metrics()


    ## can i add compute_metrics here. 
            
