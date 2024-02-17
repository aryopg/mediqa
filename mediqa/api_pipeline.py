import json
import os

import yaml
from openai import OpenAI

from mediqa.configs import ModelConfigs

# "gpt-3.5-turbo"

class APIPipeline:
        # "gpt-3.5-turbo"
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

    def construct_prompts(self, batch, mode='annotate'):

        if mode=='detect':
            
            prompts = []
            sys_prompt = "You are a clinician reviewing a clinical note that may or may not contain an error. Output your response in JSON format with keys “Step-by-step Reasoning” and “Final Answer”, where the value for “Final Answer” is either ‘Yes’ or ‘No’"

            categories = {
                'interpretation of exam results': 'interpretation of exam result', 
                'diagnoses': 'diagnosis',
                'treatments': 'treatment'
            }

            self.prompt_template = "Consider the following clinical note:\n\n{}\n\n-> Was '{}' the best possible {} here? Let’s think step-by-step. Output your response in JSON format with keys “Step-by-step Reasoning” and “Final Answer”, where the value for “Final Answer” is either ‘Yes’ or ‘No’"

            for category in categories.keys():
                for sub_phrases in batch[category]:

                    if not isinstance(sub_phrases, str):
                        continue

                    sub_phrases = sub_phrases.split("$")
                    sub_phrases = [sub_phrase.strip() for sub_phrase in sub_phrases]

                    # import pdb; pdb.set_trace()

                    for idx, sub_phrase in enumerate(sub_phrases):

                        key = f"{category} {idx}"

                        prompt = self.prompt_template.format(batch['Text'], sub_phrase, categories[category])
                        
                        prompts.append((key, prompt, sys_prompt))

                    # (Pdb) [prompt[0] for prompt in prompts]
                    # ['interpretation of exam results 0', 'treatments 0', 'treatments 1']

        elif mode=='annotate':

            self.prompt_template = "The following clinical text contains factual history of patient illness, medical exam results, interpretation of exam results, diagnoses, and treatments. I want you to classify each sentence (or each parts of sentences) in the paragraph into whether they describe factual history of present illness, medical exam results, diagnoses, or treatments.\nOutput your response in JSON format with keys “factual history of patient illness”, “medical exam results”, “interpretation of exam results”, “diagnoses”, and “treatments”. Note that “factual history of patient illness”, “medical exam results”, “interpretation of exam results”, “diagnoses”, and “treatments” may not appear in the original text in the said order.\n\n{}"
            prompt = self.prompt_template.format(batch['Text'])
            sys_prompt = "You are a clinician reviewing a clinical note that may or may not contain an error. Output your response in JSON format with keys “factual history of patient illness”, “medical exam results”, “interpretation of exam results”, “diagnoses”, and “treatments”."

            prompts=[(prompt, sys_prompt)]

        return prompts
    
    def chat(self, batch, apply_template=True, post_process=False, mode='annotate'):

        prompt_list = self.construct_prompts(batch, mode=mode)

        preds_list = [] 

        if mode=='annotate':
            for prompt, sys_prompt in prompt_list:

                try:
                    completion = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[
                            {"role": "system", "content": sys_prompt}, # You will be provided with unstructured data, and your task is to parse it into CSV format.
                            {"role": "user", "content": prompt}
                        ],
                        **self.settings
                    )

                    # print(f"completion = {completion}")
                    print("generating...")

                    if completion and completion.choices:
                        # generated_text = completion['choices'][0]['message']['content']
                        generated_text = completion.choices[0].message.content
                        # return generated_text
                    else:
                        generated_text = "No response from API call."
                except Exception as e:
                    generated_text = f"API Call Unsuccessful. Error: {str(e)}"

                preds_list.append(generated_text)

        elif mode=='detect':

            for key, prompt, sys_prompt in prompt_list:
                # import pdb; pdb.set_trace()
                try:
                    completion = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[
                            {"role": "system", "content": sys_prompt}, # You will be provided with unstructured data, and your task is to parse it into CSV format.
                            {"role": "user", "content": prompt}
                        ],
                        **self.settings
                    )

                    # print(f"completion = {completion}")
                    print("generating...")

                    if completion and completion.choices:
                        # generated_text = completion['choices'][0]['message']['content']
                        generated_text = completion.choices[0].message.content
                        print(f"generated type = {type(generated_text)}")
                        # return generated_text
                    else:
                        generated_text = "No response from API call."
                except Exception as e:
                    generated_text = f"API Call Unsuccessful. Error: {str(e)}"

                preds_list.append(generated_text)
        
        preds =  self.post_process(preds_list) if post_process else preds_list
        

        return preds, prompt_list # list of 9 prompt pairs and preds in json format
    
    def post_process(self, list_json_in_str):
        """ input string is  """
        json_dict_list = []
        for json_in_str in list_json_in_str:
            try:
                json_data = json.loads(json_in_str) 
                # handle error here. 
            except:
                json_data = json_in_str

            json_dict_list.append(json_data)


        return json_dict_list