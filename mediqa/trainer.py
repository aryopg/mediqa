import json
import math
import os

import huggingface_hub
import hydra
import pandas as pd
import torch
from accelerate import Accelerator
from accelerate.tracking import WandBTracker
from huggingface_hub import HfApi
from omegaconf import OmegaConf
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb

from .configs import TrainingConfigs
from .dataset import MEDIQADataset
from .metrics import NLGMetrics, compute_accuracy, get_nlg_eval_data
from .pipelines import APIPipeline, BasePipeline, HFPipeline


class Trainer:
    def __init__(self, configs: TrainingConfigs):
        self.configs = configs

        self.hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
        self.output_dir = self.hydra_cfg["runtime"]["output_dir"]

        self._load_dataloaders()
        self._load_pipeline()
        self._load_accelerator()

        if not configs.debug:
            self._setup_run()

    def _load_dataloaders(self) -> dict:
        self.dataloaders = {}

        # Convert data into datasets
        for split in ["train", "valid", "test"]:
            print(f"Setup {split} data loader")
            dataset = MEDIQADataset(
                self.configs.data,
                self.configs.prompt,
                self.configs.trainer,
                split=split,
            )
            self.dataloaders[split] = DataLoader(
                dataset,
                shuffle=True,
                **self.configs.data.data_loader_configs,
            )

    def _load_pipeline(self) -> BasePipeline:
        if self.configs.model.configs.model_type == "hf":
            self.pipeline = HFPipeline(self.configs.model)
        elif self.configs.model.configs.model_type == "api":
            self.pipeline = APIPipeline(self.configs.model)

    def _load_accelerator(self) -> Accelerator:
        if self.configs.model.configs.model_type == "hf":
            self.accelerator = Accelerator(log_with="wandb")
            (
                self.pipeline.model,
                self.dataloaders["train"],
                self.dataloaders["valid"],
                self.dataloaders["test"],
            ) = self.accelerator.prepare(
                self.pipeline.model,
                self.dataloaders["train"],
                self.dataloaders["valid"],
                self.dataloaders["test"],
            )
        else:
            self.accelerator = None

    @staticmethod
    def compute_metrics(predictions: pd.DataFrame):
        """
        TO DO!
        """

        def create_dict_by_text_id(column):
            return {
                text_id: value
                for text_id, value in zip(predictions["text_id"], predictions[column])
            }

        # Make a dictionary out of the DataFrame, with text_id as keys
        candidate_flags = create_dict_by_text_id("predicted_error_flags")
        candidate_sent_id = create_dict_by_text_id("predicted_error_sentence_id")
        candidate_corrections = create_dict_by_text_id("predicted_corrected_sentence")

        reference_flags = create_dict_by_text_id("label_flags")
        reference_sent_id = create_dict_by_text_id("label_sentence_ids")
        reference_corrections = create_dict_by_text_id("label_sentences")

        accuracy = compute_accuracy(
            reference_flags, reference_sent_id, candidate_flags, candidate_sent_id
        )

        # NLG Eval for corrections
        references, predictions, counters = get_nlg_eval_data(
            reference_corrections, candidate_corrections
        )
        metrics = NLGMetrics()
        nlg_eval_results = metrics.compute(references, predictions, counters)

        return {
            "accuracy": accuracy,
            "R1F_subset_check": nlg_eval_results["R1F_subset_check"],
            "R2F_subset_check": nlg_eval_results["R2F_subset_check"],
            "RLF_subset_check": nlg_eval_results["RLF_subset_check"],
            "R1FC": nlg_eval_results["R1FC"],
            "R2FC": nlg_eval_results["R2FC"],
            "RLFC": nlg_eval_results["RLFC"],
        }

    def _setup_run(self):
        ## Set group name by trainer name (i.e. zero_shot, fine_tune)
        self.wandb_group_name = self.configs.trainer.name

        # Naming by model name
        self.wandb_run_name = self.configs.model.name

        self.wandb_tracker = None
        if self.accelerator:
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
        else:
            wandb.init(
                project=self.configs.wandb_project,
                entity=self.configs.wandb_entity,
                name=self.wandb_run_name,
                group=self.wandb_group_name,
            )

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
                "predicted_error_flags",
                "predicted_error_sentence_id",
                "predicted_corrected_sentence",
                "original_prediction",
            ]
        )
        for step, batch in enumerate(tqdm(self.dataloaders[split])):
            # Predict
            prediction = self.pipeline.generate(batch)

            batch_df = pd.DataFrame(
                {
                    "text_id": batch["text_id"],
                    "original_text": batch["original_text"],
                    "prompted_text": batch["prompted_text"],
                    "label_flags": batch["label_flags"],
                    "label_sentences": batch["label_sentences"],
                    "label_sentence_ids": batch["label_sentence_ids"],
                    "predicted_error_flags": prediction["predicted_error_flags"],
                    "predicted_error_sentence_id": prediction[
                        "predicted_error_sentence_id"
                    ],
                    "predicted_corrected_sentence": prediction[
                        "predicted_corrected_sentence"
                    ],
                    "postprocess_success": prediction["postprocess_success"],
                    "original_prediction": prediction["original_prediction"],
                }
            )

            # Append the batch DataFrame to the overall predictions DataFrame
            predictions_df = pd.concat([predictions_df, batch_df], ignore_index=True)

            # Save the updated DataFrame to a CSV file after each batch
            predictions_df.to_csv(
                os.path.join(self.output_dir, f"predictions_{split}.csv"), index=False
            )

        # Evaluate
        metrics = self.compute_metrics(predictions_df)

        # Log
        print(metrics)
        if self.accelerator:
            self.accelerator.log(
                metrics
                | {f"{split}_prediction_df": wandb.Table(dataframe=predictions_df)}
            )
        else:
            wandb.log(
                metrics
                | {f"{split}_prediction_df": wandb.Table(dataframe=predictions_df)}
            )
