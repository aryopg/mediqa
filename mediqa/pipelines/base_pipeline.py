from abc import ABC, abstractmethod
from itertools import combinations, product
from typing import List, Optional, Tuple

import torch
from omegaconf import OmegaConf

from mediqa.configs import ModelConfigs, PromptConfigs


class BasePipeline(ABC):
    def __init__(
        self,
        model_configs: ModelConfigs,
        prompt_configs: PromptConfigs,
    ):
        self.model_configs = model_configs

        self.system_prompt = prompt_configs.system_prompt

        self.max_seq_len = model_configs.configs.max_seq_len

    @abstractmethod
    def generate(self):
        pass

    @staticmethod
    @abstractmethod
    def postprocess_prediction(answer):
        pass
