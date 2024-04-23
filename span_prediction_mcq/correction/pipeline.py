from typing import Tuple, Dict, Union
import ast
import os
import json
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
import re
import string
from datetime import datetime


from json_repair import repair_json
from run_api import API 
from dataset import MEDIQADataset

ARTICLES_REGEX = re.compile(r"\b(a|an|the)\b", re.UNICODE)


def get_dict_from_gpt_response(gpt_output: str) -> Dict:
    """
    Extracts and repairs a JSON string from GPT output and returns a dictionary.
    
    Parameters:
    - gpt_output (str): The JSON string output from GPT.
    
    Returns:
    - Dict: A dictionary representation of the JSON string.
    """
    json_str = gpt_output[gpt_output.find("{"):gpt_output.rfind("}") + 1]
    json_str = repair_json(json_str, skip_json_loads=True)
    return json.loads(json_str)

def normalize_answer(s: str) -> str:
    return s.lower().strip().rstrip('.')

class MCQCorrection:

    def __init__(self, model_name, input_file, save_path, num_opts=4):
        self.model_name = model_name
        self.input_file = input_file
        self.num_opts = num_opts
        self.save_path = save_path
        self.dataloader = self._load_dataloader()
        self.pipeline = self._load_chat_pipeline()

    def _load_dataloader(self) -> DataLoader:
        """
        Loads and returns the data loader.
        
        Returns:
        - DataLoader: The data loader for the dataset.
        """
        dataset = MEDIQADataset(input_file=self.input_file)
        return DataLoader(dataset)

    def _load_chat_pipeline(self):
        """
        Loads and returns the API pipeline.
        
        Returns:
        - API: The interfacing with the API.
        """
        return API(model_name=self.model_name)


    def get_prompt(self, batch, mode='get_opts'):
        """
        Generates a prompt for GPT-based error correction, depending on the mode specified.
        
        Args:
        batch (dict): A dictionary containing details about the text and errors.
        mode (str): Determines the type of prompt to generate. 'get_opts' for options, 'get_correction' for correction.
        
        Returns:
        tuple: A tuple containing the formatted prompt and the potential error span text.
        """
        # Parse offsets correctly, handling both single and multiple offset formats
        offsets = ast.literal_eval(batch['predicted_error_span_offsets'][0]) if isinstance(batch['predicted_error_span_offsets'][0], str) else batch['predicted_error_span_offsets']
        
        # Extract and format the potential error span and the blanked out error sentence
        potential_error_span = batch['Text'][0][offsets[0]: offsets[1]]
        blanked_out_error_sent = batch['predicted_error_sent'][0].replace(potential_error_span, '<BLANK>')
        blank_clause = f' in the sentence "{blanked_out_error_sent}"' if "<BLANK>" in blanked_out_error_sent else ""
        
        # Initialize prompt basics
        blanked_out_full_text = batch['text_blanked_out'][0]
        prompt, format_json = "", "Output your response in JSON format."
        
        if mode == 'get_opts':
            # Prepare the prompt for getting options
            prompt = (f'In the following clinical note, what should the <BLANK>{blank_clause} be replaced with '
                    f'if "{potential_error_span}" is incorrect? Answer with 3 most promising options. '
                    f'Do not include "{potential_error_span}" or its medical synonyms in your answer. {format_json} '
                    f'with keys \'option_1\', \'option_2\' and \'option_3\'.\n\n'
                    f'Clinical note:\n\n{blanked_out_full_text}')
        elif mode == 'get_correction':
            
            # Prepare the prompt for getting the correct option among provided
            options = [batch['gpt_opt_1'][0], potential_error_span, batch['gpt_opt_2'][0], batch['gpt_opt_3'][0]]
            labels = ['A', 'B', 'C', 'D']
            options_string = [f"\t{label}. {option}" for label, option in zip(labels, options)]
            options_string = options_string[:self.num_opts]
            options_text = "\n\nOptions:\n\n" + '\n'.join(options_string)

            prompt = (f'In the following clinical note, what should the <BLANK>{blank_clause} be replaced with '
                    f'for it to be medically informative and accurate? Choose one from the options given below. {format_json} '
                    f'with a key \'Answer\'.\n\n'
                    f'Clinical note:\n\n{blanked_out_full_text}{options_text}')
        
        return prompt, potential_error_span
    

    def post_process(self, result_df_or_path): 
        '''
        Processes result DataFrame or CSV path to:
        1. Rename columns
        2. Add binary comparison flags
        3. Compute indices for MCQ answers
        4. Generate corrected sentences based on GPT predictions
        '''
        if isinstance(result_df_or_path, str):
            result_df = pd.read_csv(result_df_or_path)
        else:
            result_df = result_df_or_path
        
        # Rename columns
        result_df.rename(columns={'Error Sentence ID': 'Label Error Sentence ID'}, inplace=True)
        result_df.rename(columns={'Corrected Sentence': 'Label Corrected Sentence'}, inplace=True)
        result_df.rename(columns={'Error Flag': 'Label Error Flag'}, inplace=True)
        result_df.rename(columns={'predicted_error_sent_id': 'Error Sentence ID'}, inplace=True)
        
        # Define helper functions
        def get_binary_flag(row):
            return int(normalize_answer(row['span_detected']) not in normalize_answer(row['mcq_ans']))

        def get_mcq_ans_index(row):
            options = [row['gpt_opt_1'], row['gpt_opt_2'], row['gpt_opt_3']]
            mcq_answer = normalize_answer(row['mcq_ans'])
            res = next((i for i, opt in enumerate(options) if normalize_answer(opt) in mcq_answer), 99)
            return res

        def get_corrected_sentence(row):
            if row['mcq_ans_index'] != 99 and row['Error Sentence ID'] != -1 and row['Error Flag']==1:
                return normalize_answer(row['predicted_error_sent']).replace(normalize_answer(row['predicted_error_span']), row[f'gpt_opt_{row["mcq_ans_index"]+1}'])
            return "NA"

        # Apply functions to DataFrame
        result_df['Error Flag'] = result_df.apply(lambda x: get_binary_flag(x), axis=1)
        result_df['mcq_ans_index'] = result_df.apply(lambda x: get_mcq_ans_index(x), axis=1)
        result_df['Corrected Sentence'] = result_df.apply(lambda x: get_corrected_sentence(x), axis=1)

        # Save to file if path was originally provided
        if isinstance(result_df_or_path, str):
            save_path = result_df_or_path.rsplit('.', 1)[0] + "_post_processed.csv"
            result_df.to_csv(save_path, index=False)

        return result_df
    
    def make_submission_file(self, post_processed_df, time_string):
        df = post_processed_df
        columns_to_include = ['Text ID', 'Error Flag', 'Error sentence ID', 'Corrected sentence']
        df.rename(columns={'Error Sentence ID': 'Error sentence ID', 'Corrected Sentence':'Corrected sentence'}, inplace=True) 
        df['Corrected sentence'] = df['Corrected sentence'].replace('\n', ' ', regex=True)
        
        filename = os.path.join(self.save_path, f"submission_{time_string}.txt")
        # We could add some sanity checks here. 

        df.to_csv(filename, index=False, columns=columns_to_include, sep=' ', header=False, na_rep='NA') 

    def predict(self, make_submission_file=True):
        """Run predictions and save outputs, including post-processing."""
        
        time_string = datetime.now().strftime('%Y%m%d_%H%M%S')

        result_df = pd.DataFrame()
        for batch in tqdm(self.dataloader):

            prompt, potential_error_span = self.get_prompt(batch, mode='get_opts')
            options_response = self.pipeline.chat(prompt)
            
            options = get_dict_from_gpt_response(options_response)

            batch.update({
                'gpt_opt_1': [options['option_1']],
                'gpt_opt_2': [options['option_2']],
                'gpt_opt_3': [options['option_3']],
                'span_detected': [potential_error_span],
                'prompt_get_opts': [prompt]
            })

            # Prediction for MCQ selection
            prompt, _ = self.get_prompt(batch, mode='get_correction')
            mcq_response = self.pipeline.chat(prompt)
            mcq_result = get_dict_from_gpt_response(mcq_response)
            
            batch.update({
                'mcq_ans': [list(mcq_result.values())[0]],  # Assuming single value MCQ response
                'prompt_get_ans': [prompt]
            })

            # Convert batch dictionary to DataFrame and append to result_df
            batch_df = pd.DataFrame(batch)
            result_df = pd.concat([result_df, batch_df], ignore_index=True)

        # Save raw results and post-processed results
        save_path = os.path.join(self.save_path, "gpt_{gpt_version}_prediction_res_{time_string}.csv")
        gpt_version = "3" if "3.5" in self.model_name else "4"
        save_path = save_path.format(gpt_version=gpt_version, time_string=time_string)
        result_df.to_csv(save_path, index=False)
        post_processed_df = self.post_process(result_df)
        post_processed_df.to_csv(save_path.replace(".csv", "_post_processed.csv"), index=False)
        
        if make_submission_file:
            self.make_submission_file(post_processed_df, time_string=time_string)

        return result_df, post_processed_df
