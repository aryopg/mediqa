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
from .metrics import NLGMetrics
from .pipeline import ModelPipeline
from .utils_chaeeun import modify_path_with_model_name


class Trainer:
    def __init__(self, configs: TrainingConfigs):
        self.configs = configs

        self.hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
        self.output_dir = self.hydra_cfg["runtime"]["output_dir"]
        # self.output_dir = modify_path_with_model_name(self.output_dir, configs['model']['name'])
        print(f"output_dir = {self.output_dir}")

        self.dataloaders = self._load_dataloaders()
        self.pipeline = self._load_pipeline()
        # self.accelerator = self._load_accelerator()
        self._load_accelerator()

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

    def _load_pipeline(self) -> ModelPipeline:
        return ModelPipeline(self.configs.model)

    def _load_accelerator(self) -> Accelerator:
        self.accelerator = Accelerator(log_with="wandb")
        print(f"self.accelerator = {self.accelerator}")
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

    @staticmethod
    def compute_metrics(prediction, batch):
        """
        TO DO!
        """
        # predicted_flags = prediction["predicted_flags"]
        # predicted_sentences = prediction["predicted_sentences"]
        # predicted_sentence_ids = prediction["predicted_sentence_ids"]

        # label_flags = batch["label_flags"]
        # label_sentences = batch["label_sentences"]
        # label_sentence_ids = batch["label_sentence_ids"]
        return {}
        # accuracy = compute_accuracy(
        #     reference_flags, reference_sent_id, candidate_flags, candidate_sent_id
        # )

        # # NLG Eval for corrections
        # metrics = NLGMetrics()
        # nlg_eval_results = metrics.compute(references, predictions, counters)

        # return {
        #     "accuracy": accuracy,
        #     "R1F_subset_check": nlg_eval_results["R1F_subset_check"],
        #     "R2F_subset_check": nlg_eval_results["R2F_subset_check"],
        #     "RLF_subset_check": nlg_eval_results["RLF_subset_check"],
        #     "R1FC": nlg_eval_results["R1FC"],
        #     "R2FC": nlg_eval_results["R2FC"],
        #     "RLFC": nlg_eval_results["RLFC"],
        # }

    def _setup_run(self):

        print(f"self.accelerator in _setup_run(self)= {self.accelerator}")

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
        self.pipeline.model.eval()
        # import pandas as pd
        # import os

        # Initialize an empty DataFrame for all predictions
        predictions_df = pd.DataFrame()

        for step, batch in enumerate(self.dataloaders[split]):


            print(f"batch = {batch}")
            print(f"type(batch) = {type(batch)}")

            # Predict
            prediction = self.pipeline.generate(batch)
            print(f"Prediction: {prediction}")

            # Evaluate
            metrics = self.compute_metrics(prediction, batch)
            # Log to console
            print(f" > Step: {step}; Metrics: {metrics}")
            # Log to wandb (assuming you have a way to convert predictions and batch to a dataframe)
            # self.accelerator.log(metrics | {f"{split}_prediction_df": wandb.Table(dataframe=predictions_df)})

            # Assuming 'prediction' is a list of predicted values, you need to match its structure to your batch data
            batch_df = pd.DataFrame({
                "text_id": batch["text_id"],
                "original_text": batch["original_text"],
                "prompted_text": batch["prompted_text"],
                "label_flags": batch["label_flags"],
                "label_sentences": batch["label_sentences"],
                "label_sentence_ids": batch["label_sentence_ids"],
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

            # # Append the current batch DataFrame directly to the CSV file
            # batch_df.to_csv(output_csv_path, mode='a', header=False, index=False)


            # If you want to stop after the first batch (for testing), remove this line for the actual run

            num_test_steps = 5
            if step >= num_test_steps:
                break
