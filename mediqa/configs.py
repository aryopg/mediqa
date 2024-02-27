from dataclasses import dataclass
from typing import List

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING


@dataclass
class DataConfigs:
    data_dir: str = MISSING

    binary_data_train: str = MISSING
    annotate_data_train: str = MISSING
    sentence_correction_without_span_data_train: str = MISSING
    identify_one_error_sentence_data_train: str = MISSING
    identify_n_error_sentences_data_train: str = MISSING

    binary_data_valid: str = MISSING
    annotate_data_valid: str = MISSING
    sentence_correction_without_span_data_valid: str = MISSING
    identify_one_error_sentence_data_valid: str = MISSING
    identify_n_error_sentences_data_valid: str = MISSING

    binary_data_test: str = MISSING
    annotate_data_test: str = MISSING
    sentence_correction_without_span_data_test: str = MISSING
    identify_one_error_sentence_data_test: str = MISSING
    identify_n_error_sentences_data_test: str = MISSING

    eval_paths: dict = MISSING
    data_loader_configs: dict = MISSING

    save_args: dict = MISSING


@dataclass
class ModelConfigs:
    name: str = MISSING
    configs: dict = MISSING


@dataclass
class TrainerConfigs:
    name: str = MISSING
    configs: dict = MISSING
    wandb_args: dict = MISSING


@dataclass
class PromptConfigs:
    # prompt_template: str = MISSING
    prompt_base: str = MISSING
    prompt_data: str = MISSING
    prompt_format: str = MISSING
    sys_prompt_base: str = MISSING
    sys_prompt_format: str = MISSING

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
    configs_store.store(group="trainer", name="base_trainer_config", node=TrainerConfigs)
