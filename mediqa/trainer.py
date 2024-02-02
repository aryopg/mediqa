import json
import math
import os

import huggingface_hub
import hydra
import pandas as pd
import torch
import wandb
from accelerate import Accelerator
from accelerate.tracking import WandBTracker
from huggingface_hub import HfApi
from omegaconf import OmegaConf
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .configs import TrainingConfigs
from .dataset import MEDIQADataset


class Trainer:
    def __init__(self, configs: TrainingConfigs):
        self.configs = configs

        self.hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
        self.output_dir = self.hydra_cfg["runtime"]["output_dir"]

        self.dataloaders = self._load_dataloaders()

        if not configs.debug:
            self._setup_run()

    def _load_dataloaders(self) -> dict:
        dataloaders = {}

        # Convert data into datasets
        for split in ["train", "valid", "test"]:
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

    def _setup_run(self):
        ## Set group name by trainer name (i.e. zero_shot, fine_tune)
        self.wandb_group_name = self.configs.trainer.name

        # Naming by model name
        self.wandb_run_name = self.configs.model.name

        self.wandb_tracker = None
        if self.accelerator.is_main_process:
            self.accelerator.init_trackers(
                project_name=self.configs.wandb_project,
                init_kwargs={
                    "wandb": {
                        "entity": self.configs.wandb_entity,
                        "name": self.wandb_run_name,
                        "group": self.wandb_group_name,
                    }
                },
            )
            self.wandb_tracker: WandBTracker = self.accelerator.get_tracker("wandb")
        self.accelerator.wait_for_everyone()

    def train(self):
        pass

    def test(self, split: str, log_metrics: bool = True):
        print(f"Testing on {split}")
        for step, batch in enumerate(self.dataloaders[split]):
            print(batch)
            print(f" > Step: {step}; Loss: ")
            break
        pass
