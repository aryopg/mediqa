import json
import os

import yaml
from openai import OpenAI

from mediqa.configs import ModelConfigs, PromptConfigs


class APIPipeline:
        # "gpt-3.5-turbo"
    def __init__(self, model_configs: ModelConfigs, prompt_configs: PromptConfigs, api_key, mode, model_name="gpt-3.5-turbo", settings=None):
        
        self.api_key = api_key
        self.model_name = model_name
        self.model_configs = model_configs

        self.mode = mode

        self.prompt_configs = prompt_configs
        self.prompt_base, self.prompt_data, self.prompt_format = self.prompt_configs.prompt_base, self.prompt_configs.prompt_data, self.prompt_configs.prompt_format
        self.sys_prompt_base, self.sys_prompt_format = self.prompt_configs.sys_prompt_base, self.prompt_configs.sys_prompt_format

        self.client = OpenAI(api_key=self.api_key)

        self.settings = settings if settings is not None else {
            "temperature": 0.7,
            "max_tokens": 1000,
            "top_p": 1.0,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
        }

    def construct_fewshots(self):
        ...

    def get_prompts(self, batch): 

        self.prompt_base, self.prompt_data, self.prompt_format = self.prompt_configs.prompt_base, self.prompt_configs.prompt_data, self.prompt_configs.prompt_format
        self.sys_prompt_base, self.sys_prompt_format = self.prompt_configs.sys_prompt_base, self.prompt_configs.sys_prompt_format

        if self.mode=='sentence_correction_without_span':
            # prompt_base = "In the context of the following clinical text, the sentence '{}' contains an error. Output a correct sentence. " # okay from this how do i do eval. i mean that's it. this is it. 
        
            # prompt_base = "In the provided clinical text, the sentence '{}' is identified to contain a clinical error. Please review this sentence with a focus on its medical content and correct it to be medically accurate. Provide the corrected sentence as your output."
            self.prompt_base = self.prompt_base.format(error_sentence=batch['Error Sentence']) # what here?
            self.prompt_data = self.prompt_data.format(full_text=batch['Text'][0])

            prompt = self.prompt_base + self.prompt_data + self.prompt_format
            sys_prompt = self.sys_prompt_base + self.sys_prompt_format

            prompts = [(prompt, sys_prompt)]

            return prompts

        elif self.mode=='binary':

            self.prompt_data = self.prompt_data.format(full_text=batch['Text'][0])
            
            prompt = self.prompt_base + self.prompt_data + self.prompt_format
            sys_prompt = self.sys_prompt_base + self.sys_prompt_format

            prompts = [(prompt, sys_prompt)]

        elif self.mode=='identify_one_error_sentence': # i want to unify if i can, how to make prompts = []. detect includes key and annotate doesn't, maybe we can unify only detect? nah fuck it. 

            categories = {
                'interpretation of exam results': 'interpretation of exam result', 
                'diagnoses': 'diagnosis',
                'treatments': 'treatment'
            }

            # import pdb; pdb.set_trace()
            ## Getting 'breakdown' string. 
            categories_dict = {}
            for category in categories.keys():
                categories_dict[category] = []

                sub_phrases = batch[category]
                # print(f"sub_phrases = {sub_phrases}")
                if len(sub_phrases) and isinstance(sub_phrases[0], str):
                    # print(f"sub_phrases = {sub_phrases}")
                    sub_phrases = [sub_phrase.strip() for sub_phrase in sub_phrases[0].split("$")]
                else:
                    sub_phrases = []



                # print(f"len(batch[category]) = {len(batch[category])}") # this is probs always 1 and we don't need a nested for loop
                
                # sub_phrases = [sub_phrase.strip() for sub_phrase in sub_phrases[0].split("$")] if len(sub_phrases) else []

                categories_dict[category] = sub_phrases
            breakdown = str(categories_dict)
            prompt_breakdown = "\n\nBreakdown into sections:\n\n{}".format(breakdown)

            self.prompt_data = self.prompt_data.format(full_text=batch['Text'][0])
            
            prompt = self.prompt_base + self.prompt_format + self.prompt_data + prompt_breakdown
            sys_prompt = self.sys_prompt_base + self.sys_prompt_format
            
            prompts = [(prompt, sys_prompt)]
                
        elif self.mode=='identify_n_error_sentences': # originally 'detect':
            
            prompts = []

            sys_prompt = self.sys_prompt_base + self.sys_prompt_format
            # the fuck is the point of this? the keys are supposed to be the output colnames from annotation, and the value are the sigular forms. 
            categories = {
                'interpretation of exam results': 'interpretation of exam result', 
                'diagnoses': 'diagnosis',
                'treatments': 'treatment'
            }

            qualifier = "a_necessary" #, a correct # "an admissable"

            # just running this would output the same thing. i can run and by run imean make 
            # and i can run ensemble. yeah sure. 

            self.prompt_template = "Consider the following clinical note:\n\n{}\n\n-> Was '{}' {} {} here? Let’s think step-by-step. Output your response in JSON format with keys “Step-by-step Reasoning” and “Final Answer”, where the value for “Final Answer” is either ‘Yes’ or ‘No’"

            for category in categories.keys(): # plural form colnames from annotation
                for sub_phrases in batch[category]: # is there ever more than one? i don't think so. 

                    if not isinstance(sub_phrases, str):
                        continue

                    sub_phrases = sub_phrases.split("$")
                    sub_phrases = [sub_phrase.strip() for sub_phrase in sub_phrases]

                    # import pdb; pdb.set_trace()

                    for idx, sub_phrase in enumerate(sub_phrases):

                        key = f"{categories[category]} {idx}" # singular form

                        self.prompt_data = self.prompt_data.format(full_text=batch['Text'][0]) 
                        self.prompt_base = self.prompt_base.format(error_sentence=sub_phrase, qualifier=qualifier, category=categories[category])
                        
                        prompt = self.prompt_data + self.prompt_base + self.prompt_format
                        
                        prompts.append((key, prompt, sys_prompt))

        elif self.mode=='annotate':

            self.prompt_data = self.prompt_data.format(full_text=batch['Text'][0])

            prompt = self.prompt_base + self.prompt_data + self.prompt_format
            sys_prompt = self.sys_prompt_base + self.sys_prompt_format

            prompts=[(prompt, sys_prompt)]

        return prompts
    
    def chat(self, batch, post_process=False): # how do i add self-consistency here? it would be useful esp for binary. -> just add a for loop. -> 

        prompt_list = self.get_prompts(batch)
        preds_list = [] 

        for tup in prompt_list:

            if self.mode=='identify_n_error_sentences':
                _, prompt, sys_prompt = tup
            else:
                prompt, sys_prompt = tup

            
            try:
                completion = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": sys_prompt}, # You will be provided with unstructured data, and your task is to parse it into CSV format.
                        {"role": "user", "content": prompt}
                    ],
                    **self.settings
                )

                # print("generating...")

                if completion and completion.choices:
                    # generated_text = completion['choices'][0]['message']['content']
                    generated_text = completion.choices[0].message.content
                    # print(f"generated type = {type(generated_text)}")
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