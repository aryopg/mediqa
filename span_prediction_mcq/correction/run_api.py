import json
import os
import ast

import yaml
from openai import OpenAI


class API:

    def __init__(self, model_name, settings=None):
        
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
 
    
    def chat(self, prompt): 
        
        sys_prompt = "Your job is to review a clinical note that potentially contains a medical error."

        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": sys_prompt}, 
                    {"role": "user", "content": prompt}
                ],
                **self.settings
            )

            # print("generating...")

            if completion and completion.choices:
                # generated_text = completion['choices'][0]['message']['content']
                generated_text = completion.choices[0].message.content
            else:
                generated_text = "No response from API call."

        except Exception as e:
            generated_text = f"API Call Unsuccessful. Error: {str(e)}"
        
        return generated_text
    
