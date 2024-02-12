import json
import math
import os

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

    def _load_chat_pipeline(self): #-> ModelPipeline:
        # return ModelPipeline(self.configs.model)
        return APIPipeline(self.configs.model, self.api_key)

    @staticmethod
    def compute_metrics(prediction, batch):
        """
        TO DO!

        prediction: API output converted to JSON object. 
        """

        return {}

    def train(self):
        pass

    def test(self, split: str, log_metrics: bool = True):

        print(f"Testing on {split}")

        predictions_df = pd.DataFrame(
            columns=[
                "text_id",
                "original_text",
                "prompted_text",
                "label_flags",
                "label_sentences",
                "label_sentence_ids",
                "predicted_flags",
                "predicted_sentences",
                "predicted_sentence_ids",
                "original_prediction",
            ]
        )

        predictions_df = pd.DataFrame()

        for step, batch in enumerate(self.dataloaders[split]):

            # Predict
            prediction, prompt = self.pipeline.chat(batch, apply_template = True)
            # print(f"pred_res = {pred_res}")
            print(f"Prediction: {prediction}")

            # Evaluate
            metrics = self.compute_metrics(prediction, batch)
            # Log to console
            print(f" > Step: {step}; Metrics: {metrics}")

            batch_df = pd.DataFrame({
                "text_id": batch["text_id"],
                "original_text": batch["original_text"],
                "prompted_text": prompt, # batch["prompted_text"],
                "label_flags": batch["label_flags"],
                "label_sentences": batch["label_sentences"],
                "label_sentence_ids": batch["label_sentence_ids"],

                # "correction": batch["correction"],
                'error_span': batch["error_span"],
                # Update these lines to correctly reflect your prediction structure
                # "predicted_flags": prediction["predicted_flags"],
                # "predicted_sentences": prediction["predicted_sentences"],
                # "predicted_sentence_ids": prediction["predicted_sentence_ids"],
                # "original_prediction": prediction["original_prediction"],
                # # Example of adding a new column for a single prediction, adjust based on actual prediction structure
                "prediction": prediction
            })

            # Append the batch DataFrame to the overall predictions DataFrame
            predictions_df = pd.concat([predictions_df, batch_df], ignore_index=True)

            # Save the updated DataFrame to a CSV file after each batch
            predictions_df.to_csv(os.path.join(self.output_dir, f"predictions_{split}.csv"), index=False)

            # If you want to stop after the first batch (for testing), remove this line for the actual run

            num_test_steps = 100
            if step >= num_test_steps:
                break
