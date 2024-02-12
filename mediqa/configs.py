from dataclasses import dataclass
from typing import List

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING


@dataclass
class DataConfigs:
    data_dir: str = MISSING
    train_data_filename: str = MISSING
    valid_data_filename: str = MISSING
    test_data_filename: str = MISSING
    data_loader_configs: dict = MISSING


@dataclass
class ModelConfigs:
    name: str = MISSING
    configs: dict = MISSING


@dataclass
class TrainerConfigs:
    name: str = MISSING
    configs: dict = MISSING


@dataclass
class PromptConfigs:
    prompt_template: str = MISSING


@dataclass
class TrainingConfigs:
    data: DataConfigs = MISSING
    trainer: TrainerConfigs = MISSING
    model: ModelConfigs = MISSING
    prompt: PromptConfigs = MISSING
    wandb_project: str = MISSING
    wandb_entity: str = MISSING
    debug: bool = False
    random_seed: int = 1234
    api_key_path: str = MISSING


def register_base_configs() -> None:
    configs_store = ConfigStore.instance()
    configs_store.store(name="base_config", node=TrainingConfigs)
    configs_store.store(group="data", name="base_data_config", node=DataConfigs)
    configs_store.store(group="model", name="base_model_config", node=ModelConfigs)
    configs_store.store(group="prompt", name="base_prompt_config", node=PromptConfigs)
    configs_store.store(
        group="trainer", name="base_trainer_config", node=TrainerConfigs
    )
