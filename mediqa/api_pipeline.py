import json
import os

import yaml
from openai import OpenAI

from mediqa.configs import ModelConfigs


# "gpt-3.5-turbo"
class APIPipeline:
    def __init__(self, model_configs: ModelConfigs, api_key, prompt_template="{}", model_name="gpt-4", settings=None):
        self.api_key = api_key
        self.model_name = model_name
        self.prompt_template = prompt_template
        self.model_configs = model_configs

        self.client = OpenAI(
            
            api_key=self.api_key,
        )

        self.settings = settings if settings is not None else {
            "temperature": 0.7,
            "max_tokens": 1000,
            "top_p": 1.0,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
        }

    def construct_prompt(self, batch, mode='probe'):

        if mode=='probe':
            self.prompt_template = "{}\n\n-> what should <BLANK> be replaced with if '{}' is incorrect?"
            # print(f"type(batch['correction'][0]) = {type(batch['correction'][0])}")
            prompt = self.prompt_template.format(batch["prompted_text"][0], batch['error_span'][0])
        # elif mode=='cot':
        # elif mode=='nle':

        return prompt

    def chat(self, batch, apply_template=True, post_process=False):

        prompt = self.construct_prompt(batch) if apply_template else batch["prompted_text"]

        print(f"prompt = {prompt}")

        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.model_configs.configs.system_prompt}, # You will be provided with unstructured data, and your task is to parse it into CSV format.
                    {"role": "user", "content": prompt}
                ],
                **self.settings
            )

            print(f"completion = {completion}")

            if completion and completion.choices:
                # generated_text = completion['choices'][0]['message']['content']
                generated_text = completion.choices[0].message.content
                # return generated_text
            else:
                return "No response generated."
        except Exception as e:
            return f"Error: {str(e)}"
        
        pred =  self.post_process(generated_text) if post_process else generated_text

        return pred, prompt
        
    def post_process(self, json_in_str):

        json_data = json.loads(json_in_str)

        return json_data
