from itertools import combinations, product
from typing import List, Optional, Tuple

import torch
from omegaconf import OmegaConf
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer

from mediqa.configs import ModelConfigs


class ModelPipeline:
    def __init__(
        self,
        model_configs: ModelConfigs,
    ):
        self.model_configs = model_configs
        self.model = AutoModelForCausalLM.from_pretrained(
            model_configs.configs.model_name_or_path,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_configs.configs.model_name_or_path
        )

        # Only LLaMA that can use system prompt
        self.system_prompt = None
        if "llama" in self.model_configs.configs.model_name_or_path.lower():
            self.system_prompt = model_configs.configs.system_prompt

        self.max_seq_len = model_configs.configs.max_seq_len

    def _tokenize_input(self, inputs):
        prompt = []
        if self.system_prompt:
            prompt += [{"role": "system", "content": self.system_prompt}]
        prompt += [{"role": "user", "content": inputs["prompted_text"][0]}]

        model_input = self.tokenizer.apply_chat_template(
            prompt, return_tensors="pt", max_length=self.max_seq_len
        )

        return [model_input]

    def generate(
        self,
        inputs,
        use_cot=False,
    ) -> Tuple[List[List[int]], Optional[List[List[float]]]]:
        # Most likely we're just gonna do batch size = 1
        model_input = self._tokenize_input(inputs)[0]

        # Limit generation length
        max_new_tokens = min(
            128, self.max_seq_len - model_input.size(1)
        )  # TODO: Move 128 to config

        # Predict
        with torch.inference_mode():
            model_input = model_input.to(self.model.device)

            model_inputs = {
                "input_ids": model_input,
                "max_new_tokens": max_new_tokens,
                "do_sample": False,
                "pad_token_id": self.tokenizer.eos_token_id,
            }

            # Forward pass
            output = self.model.generate(**model_inputs)
            decoded_text = self.tokenizer.decode(
                output[0, model_input.size(1) :],
                skip_special_tokens=True,
            )

        # TODO: postprocess the prediction
        prediction = self.postprocess_prediction(decoded_text)

        return {
            "input_length": model_input.size(1),
            "max_new_tokens": max_new_tokens,
            "predicted_flags": prediction["predicted_flags"],
            "predicted_sentences": prediction["predicted_sentences"],
            "predicted_sentence_ids": prediction["predicted_sentence_ids"],
            "original_prediction": decoded_text,
        }

    @staticmethod
    def postprocess_prediction(answer):
        """
        TO DO
        """
        # TODO!
        return {
            "predicted_flags": "",
            "predicted_sentences": "",
            "predicted_sentence_ids": "",
        }
