from torch.utils.data import DataLoader
import ast
import json
import math
import os
import re
import time
import string
from datetime import datetime

import pandas as pd
from tqdm import tqdm
from json_repair import repair_json

from api_pipeline import APIPipeline
from dataset_2opts_ms import MEDIQADataset

ARTICLES_REGEX = re.compile(r"\b(a|an|the)\b", re.UNICODE)

def get_dict_from_gpt_response(gpt_output):
    json_str = gpt_output
    json_str = json_str[json_str.find("{"):json_str.rfind("}")+1]
    
    json_str = repair_json(json_str, skip_json_loads=True)
    
    dic = json.loads(json_str)
    
    return dic

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    # but maybe don't remove punctuation! -> for finding error span it doesn't matter, but for finding index it does matter a lot! 
    # nah... it does matter. removing article is okay maybe don't remove punctuation?

    def remove_articles(text):
        return ARTICLES_REGEX.sub(" ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))
    # return white_space_fix(remove_articles(lower(s)))
    # return s


def get_tokens(s):
    if not s:
        return []
    return normalize_answer(s).split()

class APITrainer: 
    def __init__(self, model_name="gpt-3.5-turbo"):

        self.dataloader = self._load_dataloader()
        self.pipeline = self._load_chat_pipeline(model_name=model_name)
    

    def _load_dataloader(self) -> dict:

        dataset = MEDIQADataset() # input_file_name=,)
        dataloader=DataLoader(dataset)#, shuffle=True)
        # import pdb; pdb.set_trace()
        return dataloader

    def _load_chat_pipeline(self, model_name): 
        return APIPipeline(model_name=model_name)
    
    def get_prompt(self, batch, opa, opb, mode='get_correction'): # batch is tuple of two rows. 
        
        b1, b2 = batch
        
        # blanked_out_full_text = batch['text_blanked_out'][0]
        blanked_out_full_text_1, blanked_out_full_text_2 = b1['text_blanked_out'][0], b2['text_blanked_out'][0]
        
        
        # offsets = batch['predicted_error_span_offsets']
        offsets1, offsets2 = b1['predicted_error_span_offsets'], b2['predicted_error_span_offsets']
        
        
        # if len(offsets)==1:
        #     offsets = ast.literal_eval(offsets[0])
        if len(offsets1)==1:
            offsets1 = ast.literal_eval(offsets1[0])
        if len(offsets2)==1:
            offsets2 = ast.literal_eval(offsets2[0])

        # import pdb; pdb.set_trace()
        # blank_clause = ''
        blank_clause1, blank_clause2 = '', ''
        
        # potential_error_span = batch['Text'][0][ offsets[0] : offsets[1] ] # potential fix
        potential_error_span1 = b1['Text'][0][ offsets1[0] : offsets1[1] ]
        potential_error_span2 = b2['Text'][0][ offsets2[0] : offsets2[1] ]
        
        
        
        # blanked_out_error_sent = batch['predicted_error_sent'][0].replace(potential_error_span, '<BLANK>')
        # if "<BLANK>" in blanked_out_error_sent:
        #     blank_clause = ' in the sentence "{blanked_out_error_sent}"'.format(blanked_out_error_sent=blanked_out_error_sent)
        
        blanked_out_error_sent1 = b1['predicted_error_sent'][0].replace(potential_error_span1, '<BLANK>')
        if "<BLANK>" in blanked_out_error_sent1:
            blank_clause1 = ' in the sentence "{blanked_out_error_sent}"'.format(blanked_out_error_sent=blanked_out_error_sent1)
            
        blanked_out_error_sent2 = b2['predicted_error_sent'][0].replace(potential_error_span2, '<BLANK>')
        if "<BLANK>" in blanked_out_error_sent2:
            blank_clause2 = ' in the sentence "{blanked_out_error_sent}"'.format(blanked_out_error_sent=blanked_out_error_sent2)
            
        
        prompt = ""
        prefix, format_json = "", ""
        
        # if mode=='get_opts':
        #     prefix = 'In the following clinical note, what should the <BLANK>{blank_clause} be replaced with if "{potential_error_span}" is incorrect? Do not answer with "{potential_error_span}" or its medical synonyms in your answer.'
        #     prefix = prefix.format(blank_clause=blank_clause, potential_error_span=potential_error_span)
        #     format_json = "Output your response in JSON format, with keys 'option'."
        #     prompt = prefix + ' ' + format_json + "\n\nClinical note:\n\n" + blanked_out_full_text
            # import pdb; pdb.set_trace()
        if mode=='get_correction':
            
            # opts_lst = batch['gpt_options'] # assuming this is list -> i mean okay for gpt generated ones, sure. 
            # import pdb; pdb.set_trace()
            options = "\n\nOptions:\n\n" + f"\tA. {opa}\n" + f"\tB. {opb}\n" # + f"\tC. {batch['gpt_opt_2'][0]}\n" + f"\tD. {batch['gpt_opt_3'][0]}\n"
            
            prefix = 'In the following clinical note, what should the <BLANK>{blank_clause} be replaced with for it to be medically informative and accurate? Choose one from the options given below.' # four options given below.'
            prefix = prefix.format(blank_clause=blank_clause1)
            format_json = "Output your response in JSON format, with a key 'Answer'."
            
            prompt = prefix + ' ' + format_json + "\n\nClinical note:\n\n" + blanked_out_full_text_1 + options
            
            # import pdb; pdb.set_trace()
        
        return prompt, opa # potential_error_span
    
    def post_process_for_submission(self, result_df_or_path):
        '''
        1. rename
        2. Add binary -> compare mcq_ans to span_detected
        '''
        result_df = None
        if isinstance(result_df_or_path, str):
            result_df = pd.read_csv(result_df_or_path)
        else:
            result_df = result_df_or_path
            
        
        result_df.rename(columns={'predicted_error_sent_id': 'Error Sentence ID'}, inplace=True) 
        
        # result_df['mcq_ans_index'] = None
        
        def get_binary_flag(row):
            # import pdb; pdb.set_trace()
            
            ori = row['opa'] if row['ori_opt']=='a' else row['opb']
            if normalize_answer(ori) in normalize_answer(row['mcq_ans']):
                row['Error Sentence ID'] = -1
            return row
        
        def get_mcq_ans_index(row):
            # ans_idx = -1
            # # import pdb; pdb.set_trace()
            # a = row['gpt_opt']
            
            ori = row['opa'] if row['ori_opt']=='a' else row['opb']
            
            if ori in row['mcq_ans']:
                ans_idx = 99 # original span
            else:
                ans_idx = 0 
                
            return 'ori' if ans_idx == 99 else 'the_other' # ans_idx
        
        def get_corrected_sentence(row):
            
            res = "NA"
            
            ori = row['opa'] if row['ori_opt']=='a' else row['opb']
            
            # error_spans = row['predicted_error_span'].lower().split(". ")
            error_spans = row['predicted_error_span'].split(". ")
            
            opts = ['opa', 'opb']
            for error_span in error_spans: # if there is error 
                if (row['mcq_ans_index']=='the_other') and (row['Error Sentence ID'] != -1):# -1 = no error
                    # import pdb; pdb.set_trace()
                    # res = row['predicted_error_sent'].lower().replace(error_span, row[opts[row['mcq_ans_index']]]) # no it works. 
                    
                    col = 'opb' if row['ori_opt']=='a' else 'opa'
                    res = row['predicted_error_sent'].lower().replace(error_span, row[col])
                    # import pdb; pdb.set_trace()
                else: # row['mcq_ans_index']==99:
                    res = "NA"
                    
            return res
        
        result_df = result_df.apply(get_binary_flag, axis=1)
        result_df['Error Flag'] = (result_df['Error Sentence ID'] != -1).astype(int)
        # import pdb; pdb.set_trace()
        
        result_df['mcq_ans_index'] = result_df.apply(get_mcq_ans_index, axis=1)
        result_df['Corrected Sentence'] = result_df.apply(get_corrected_sentence, axis=1) # how are we going to handle the edge case? 
        
        # import pdb; pdb.set_trace()
        # result_df['Error Sentence ID'] = [result_df['Error Sentence ID'][0].item()]
        result_df['Error Sentence ID'] = result_df['Error Sentence ID'].apply(lambda x: x.item() if isinstance(x, int)==False else x)
        
        if isinstance(result_df_or_path, str):
            result_df.to_csv(result_df_or_path.split(".")[0] + "_post_processed_ms_only_final.csv", index=False)
            
        print("POST PROCESSING DONE !!")

        # Apply the function across the DataFrame rows
        return result_df # = result_df.apply(check_and_update, axis=1)
        
        
        
        
        

    def predict(self): # ah maybe there's mode. get options and choose. does the pipeline also have a mode distinction? i guess yea. just save everything to json and parse from there. 
        
        # is there any way we can ensure 50% binary prediction? -> mcq. other than that not really. 

        prev_df = pd.DataFrame() # we want those three columns, and they can be calculated from... well okay first i get df with mcq ansewr and then i compare them to the original predicted error span. and set flag. ill have another script for that. 
        # work with dictionary, and only save as df at the last stage. 
        time_string = datetime.now().strftime('%H%M%S')
        save_path = f"/home/co-chae/rds/hpc-work/mediqa_output/gpt_correction/gpt3/gpt_prediction_2opts_res_{time_string}.csv"

        for batch in tqdm(self.dataloader): 
            
            
            b1, b2 = batch
            # import pdb; pdb.set_trace()
            
            
            
            # batch_df = pd.DataFrame(batch)
            # batch['gpt_opt'] = "" # , "", ""
            # prompt, potential_error_span = self.get_prompt(batch, mode='get_opts')
            # resp_json_string = self.pipeline.chat(prompt) # get predictions (json parsed in pipeline)
            # resp = get_dict_from_gpt_response(resp_json_string)
            
            
            # opa, opb = self.get_prompt(batch, mode='get_opts')
            opa, opb = b1['predicted_error_span'], b2['predicted_error_span']
            
            b1['opa'], b1['opb'] = opa, opb
            b2['opa'], b2['opb'] = opa, opb
            
            b1['ori_opt'], b2['ori_opt'] = 'a', 'b'
            
            
            # batch['predicted_error_sent_id'] = [batch['predicted_error_sent_id'][0].item()]
            # batch['span_detected'] = [potential_error_span]
            # batch['prompt_get_opts'] = [prompt]
            
            # # import pdb; pdb.set_trace()
            
            # batch['gpt_opt'] = [resp['option']] # , [resp['option_2']], [resp['option_3']]
            
            prompt, _ = self.get_prompt(batch, opa, opb, mode='get_correction')
            
            resp_json_string = self.pipeline.chat(prompt) # get predictions (json parsed in pipeline)
            
            
            resp = get_dict_from_gpt_response(resp_json_string)
            
            
            # batch['mcq_ans'] = list(resp.values()) # although it's a single value
            # batch['prompt_used'] = [prompt]
            try:
                b1['mcq_ans'] = list(resp.values()) # although it's a single value
            except:
                import pdb; pdb.set_trace()
                b1['mcq_ans'] = ["error"]
            b1['prompt_used'] = [prompt]
                        
                        
            try:
                b2['mcq_ans'] = list(resp.values()) # although it's a single value
            except:
                import pdb; pdb.set_trace()
                b2['mcq_ans'] = ["error"]
            # b2['mcq_ans'] = list(resp.values()) # although it's a single value
            b2['prompt_used'] = [prompt]
       

            # prev_df, result_df = prev_df.reset_index(drop=True), result_df.reset_index(drop=True)
            b1_df = pd.DataFrame.from_dict(b1, orient='index').T
            b2_df = pd.DataFrame.from_dict(b2, orient='index').T
            batch_df = pd.concat([b1_df, b2_df], ignore_index=True)
            
            # import pdb; pdb.set_trace()
            # import pdb; pdb.set_trace()

            
            
            # result_df = pd.concat([batch_df, result_df], axis=1)#, ignore_index=True)
            result_df = pd.concat([prev_df, batch_df], ignore_index=True)
            prev_df = result_df
            
            print(f"shape={result_df.shape}")
        
       
            # result_df.to_csv(save_path, index=False)

            # rest_for, after = 5, 50
            # if step % after == 0:
                
            #     print(f"step = {step}")
            #     result_df.to_csv(save_path, index=False)
            #     print(f"Saved to {save_path} after {step} steps")
            #     print(f"Sleeping for {rest_for} seconds after {after} generations...")
            #     time.sleep(rest_for) 

            # num_test_steps = 1000
      
            # if step >= num_test_steps:
            #     print(f"break statement in Training loop after {step} steps!")
            #     break
            
        result_df.to_csv(save_path, index=False)
        result_df_post_processed = self.post_process_for_submission(result_df)
        result_df_post_processed.to_csv(save_path.split(".")[0] + "_post_processed.csv", index=False)
        
                



    ## can i add compute_metrics here. 
            
