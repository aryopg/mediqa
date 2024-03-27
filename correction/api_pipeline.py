import json
import os
import ast

import yaml
from openai import OpenAI

# def dict_to_list(topk_dict, k=self.k_in_topk):

#     # return 
#     dict_vals = [([topk_dict[key]] if isinstance(topk_dict[key], str) else topk_dict[key]) for key in topk_dict.keys()]
#     flattened_list = [item for sublist in dict_vals for item in sublist]

#     return flattened_list

# def json_string_to_list(input_string):
#     res = None
#     try:
#         res = dict_to_list(ast.literal_eval(results[0]))
#     except:
#         res = [""]*3

#     return res

# def get_dict_from_gpt_response(gpt_output):
#     json_str = gpt_output
#     json_str = json_str[json_str.find("{"):json_str.rfind("}")+1]
    
#     from json_repair import repair_json
#     json_str = repair_json(json_str, skip_json_loads=True)
    
#     dic = json.loads(json_str)
    
#     return dic

class APIPipeline:
        # "gpt-3.5-turbo"
    def __init__(self, model_name="gpt-3.5-turbo", settings=None):
        
        self.api_key = os.getenv('OPENAI_API_KEY')
        self.model_name = model_name

        self.client = OpenAI(api_key=self.api_key)

        self.settings = settings if settings is not None else {
            "temperature": 0.0,
            "max_tokens": 4096,
            "top_p": 1.0,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
        }
 
    
    def chat(self, prompt): # how do i add self-consistency here? it would be useful esp for binary. -> just add a for loop. -> 

        # preds_list = [] 

        sys_prompt = "Your job is to review a clinical note that potentially contains a medical error."

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
        
        return generated_text
    
