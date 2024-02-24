from abc import ABC, abstractmethod
from itertools import combinations, product
from typing import List, Optional, Tuple

import torch
from omegaconf import OmegaConf

from mediqa.configs import ModelConfigs


class BasePipeline(ABC):
    def __init__(
        self,
        model_configs: ModelConfigs,
    ):
        self.model_configs = model_configs

        # Only LLaMA and GPT that can use system prompt
        self.system_prompt = None
        if (
            "llama" in self.model_configs.configs.model_name_or_path.lower()
            or "gpt" in self.model_configs.configs.model_name_or_path.lower()
        ):
            self.system_prompt = model_configs.configs.system_prompt

        self.max_seq_len = model_configs.configs.max_seq_len

    @abstractmethod
    def generate(self):
        pass

    @staticmethod
    @abstractmethod
    def postprocess_prediction(answer):
        pass
