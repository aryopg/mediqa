import logging
import os
import sys

from dotenv import load_dotenv

load_dotenv(".env")

sys.path.append(os.getcwd())


import hydra
from omegaconf import OmegaConf

# from mediqa.trainer import Trainer
from mediqa.api_trainer import APITrainer
from mediqa.configs import TrainingConfigs, register_base_configs
from mediqa.utils import common_utils


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(configs: TrainingConfigs) -> None:
    missing_keys: set[str] = OmegaConf.missing_keys(configs)
    # if missing_keys:
    #     raise RuntimeError(f"Got missing keys in config:\n{missing_keys}")

    common_utils.setup_random_seed(configs.random_seed)

    trainer = APITrainer(configs)
    mode = configs.trainer.configs.mode
    print(f"MODE = {mode}")
    
    # _ = trainer.test("train", log_metrics=True, mode=mode)
    _ = trainer.test("valid", log_metrics=True)
    # _ = trainer.test("test", log_metrics=True)


if __name__ == "__main__":
    register_base_configs()
    main()
