import json
import logging
import os
import time
from itertools import combinations, product
from typing import List, Optional, Tuple

import json_repair
import pandas as pd
from bert_score import score
from omegaconf import OmegaConf
from openai import OpenAI

from ..configs import ModelConfigs, PromptConfigs
from .base_pipeline import BasePipeline

logging.getLogger("httpx").setLevel(logging.WARNING)


class APIPipeline(BasePipeline):
    def __init__(self, model_configs: ModelConfigs, prompt_configs: PromptConfigs):
        super().__init__(model_configs, prompt_configs)

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
        expected_label = {}
        if len(label["label_reason"]):
            if type(label["label_reason"]) == list:
                if label["label_reason"][0]:
                    expected_label["reason"] = label["label_reason"][0]
            elif type(label["label_reason"]) == str:
                if label["label_reason"]:
                    expected_label["reason"] = label["label_reason"]
        if label["label_flags"] == 0:
            expected_label["incorrect_sentence_id"] = -1
            expected_label["correction"] = "NA"
        elif label["label_flags"] == 1:
            expected_label["incorrect_sentence_id"] = label["label_sentence_ids"][0]
            expected_label["correction"] = label["label_sentences"][0]
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
            "predicted_error_flag": prediction["predicted_error_flag"],
            "predicted_error_sentence_id": prediction["predicted_error_sentence_id"],
            "predicted_corrected_sentence": prediction["predicted_corrected_sentence"],
            "postprocess_success": prediction["postprocess_success"],
        }

    @staticmethod
    def postprocess_prediction(generated_text):
        if "{" in generated_text and "}" in generated_text:
            json_string = generated_text[
                generated_text.find("{") : generated_text.rfind("}") + 1
            ]
        else:
            # Most likely will fail
            json_string = generated_text
        try:
            jsonified_text = json.loads(json_string)

            if jsonified_text is None:
                predicted_error_flag = 0
                predicted_error_sentence_id = -1
                predicted_corrected_sentence = "NA"
                success = True
            else:
                try:
                    if jsonified_text["correction"] == "NA":
                        predicted_error_flag = 0
                        predicted_error_sentence_id = -1
                        predicted_corrected_sentence = "NA"
                    else:
                        predicted_error_flag = 1
                        predicted_error_sentence_id = int(
                            jsonified_text["incorrect_sentence_id"]
                        )
                        predicted_corrected_sentence = jsonified_text["correction"]
                    success = True
                except KeyError as e:
                    predicted_error_flag = 0
                    predicted_error_sentence_id = -1
                    predicted_corrected_sentence = "NA"
                    success = False
        except json.decoder.JSONDecodeError as e:
            try:
                print("Desperate fix with json_repair")
                jsonified_text = json_repair.loads(json_string)
                if jsonified_text["correction"] == "NA":
                    predicted_error_flag = 0
                    predicted_error_sentence_id = -1
                    predicted_corrected_sentence = "NA"
                else:
                    predicted_error_flag = 1
                    predicted_error_sentence_id = int(
                        jsonified_text["incorrect_sentence_id"]
                    )
                    predicted_corrected_sentence = jsonified_text["correction"]
            except Exception as e:
                print("json_repair failed too")
                print(e)
                predicted_error_flag = 0
                predicted_error_sentence_id = -1
                predicted_corrected_sentence = "NA"
            success = False
        except TypeError as e:
            print("Failed postprocessing")
            predicted_error_flag = 0
            predicted_error_sentence_id = -1
            predicted_corrected_sentence = "NA"
            success = False

        return {
            "predicted_error_flag": predicted_error_flag,
            "predicted_error_sentence_id": predicted_error_sentence_id,
            "predicted_corrected_sentence": predicted_corrected_sentence,
            "postprocess_success": success,
        }

    @staticmethod
    def bertscore_filter(predictions_df: pd.DataFrame, threshold: float = 0.85):
        """
        Comparing the predicted correction with the original sentence (the suspected mistake)
        If the BERTscore F1 is higher than the threshold, the prediction is kept, otherwise it is replaced with "NA"

        Args:
            predictions_df (pd.DataFrame): _description_
            threshold (float, optional): _description_. Defaults to 0.85.

        Returns:
            _type_: _description_
        """
        original_predicted_error_flags = predictions_df["predicted_error_flag"].tolist()
        original_predicted_error_sentence_ids = predictions_df[
            "predicted_error_sentence_id"
        ].tolist()
        original_predicted_corrected_sentences = predictions_df[
            "predicted_corrected_sentence"
        ].tolist()
        # Reference is the original sentence
        references = [
            (
                split_sentences[int(predicted_error_sentence_id)]
                if predicted_error_sentence_id != -1
                else "NA"
            )
            for split_sentences, predicted_error_sentence_id in zip(
                predictions_df["split_sentences"].tolist(),
                predictions_df["predicted_error_sentence_id"].tolist(),
            )
        ]
        _, _, f1s = score(
            original_predicted_corrected_sentences, references, lang="en", verbose=True
        )

        filtered_predicted_error_flags = []
        filtered_predicted_error_sentence_ids = []
        filtered_predicted_corrected_sentences = []
        for i, f1 in enumerate(f1s):
            if f1 >= threshold:
                filtered_predicted_error_flags += [original_predicted_error_flags[i]]
                filtered_predicted_error_sentence_ids += [
                    original_predicted_error_sentence_ids[i]
                ]
                filtered_predicted_corrected_sentences += [
                    original_predicted_corrected_sentences[i]
                ]
            else:
                filtered_predicted_error_flags += [0]
                filtered_predicted_error_sentence_ids += [-1]
                filtered_predicted_corrected_sentences += ["NA"]

        postprocessed_predictions_df = predictions_df.copy()
        postprocessed_predictions_df["predicted_error_flag"] = (
            filtered_predicted_error_flags
        )
        postprocessed_predictions_df["predicted_error_sentence_id"] = (
            filtered_predicted_error_sentence_ids
        )
        postprocessed_predictions_df["predicted_corrected_sentence"] = (
            filtered_predicted_corrected_sentences
        )

        return postprocessed_predictions_df
