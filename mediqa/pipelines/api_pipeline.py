import json
import os
import time
from itertools import combinations, product
from typing import List, Optional, Tuple

from omegaconf import OmegaConf
from openai import OpenAI

from ..configs import ModelConfigs
from .base_pipeline import BasePipeline


class APIPipeline(BasePipeline):
    def __init__(
        self,
        model_configs: ModelConfigs,
    ):
        super().__init__(model_configs)

        self.model_name = self.model_configs.configs.model_name_or_path
        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY", ""),
        )

        self.generation_configs = {
            "temperature": self.model_configs.configs.temperature,
            "top_p": self.model_configs.configs.top_p,
            "frequency_penalty": self.model_configs.configs.frequency_penalty,
            "presence_penalty": self.model_configs.configs.presence_penalty,
            "max_tokens": self.model_configs.configs.max_tokens,
        }

    @staticmethod
    def _label_template(label):
        if label["label_flags"] == 0:
            expected_label = None
        elif label["label_flags"] == 1:
            expected_label = {
                "incorrect_sentence_id": label["label_sentence_ids"][0],
                "correction": label["label_sentences"][0],
            }
        return json.dumps(expected_label)

    def _create_prompt(self, inputs):
        prompt = []
        # System prompt
        if self.system_prompt:
            prompt += [{"role": "system", "content": self.system_prompt}]

        # ICL examples
        for icl_text, icl_label in zip(inputs["icl_texts"], inputs["icl_labels"]):
            prompt += [{"role": "user", "content": icl_text[0]}]
            prompt += [
                {"role": "assistant", "content": self._label_template(icl_label)}
            ]

        # Actual prompt
        prompt += [{"role": "user", "content": inputs["prompted_text"][0]}]

        return prompt

    def generate(
        self,
        inputs,
        use_cot=False,
    ) -> Tuple[List[List[int]], Optional[List[List[float]]]]:
        prompt = self._create_prompt(inputs)

        response = self.client.chat.completions.create(
            model=self.model_name, messages=prompt, **self.generation_configs
        )
        if response and response.choices:
            generated_text = response.choices[0].message.content
        else:
            generated_text = ""

        # Postprocess the prediction
        prediction = self.postprocess_prediction(generated_text)

        return {
            "original_prediction": generated_text,
            "predicted_error_flags": prediction["predicted_error_flags"],
            "predicted_error_sentence_id": prediction["predicted_error_sentence_id"],
            "predicted_corrected_sentence": prediction["predicted_corrected_sentence"],
            "postprocess_success": prediction["postprocess_success"],
        }

    @staticmethod
    def postprocess_prediction(generated_text):
        """
        TO DO
        """
        try:
            jsonified_text = json.loads(generated_text)

            if jsonified_text is None:
                predicted_error_flags = 0
                predicted_error_sentence_id = -1
                predicted_corrected_sentence = "NA"
            else:
                predicted_error_flags = 1
                predicted_error_sentence_id = int(
                    jsonified_text["incorrect_sentence_id"]
                )
                predicted_corrected_sentence = jsonified_text["correction"]

            success = True
        except json.decoder.JSONDecodeError as e:
            predicted_error_flags = 0
            predicted_error_sentence_id = -1
            predicted_corrected_sentence = "NA"
            success = False

        return {
            "predicted_error_flags": predicted_error_flags,
            "predicted_error_sentence_id": predicted_error_sentence_id,
            "predicted_corrected_sentence": predicted_corrected_sentence,
            "postprocess_success": success,
        }
