"""
create dataset compatible with squad training code for span prediction. 
1. "answers": { "text": [original_error_span], "answer_start": find_all_subsequences(text=example['context'], sub_sequence=original_error_span) }

5. maybe add redact function for UW

don't forget to apply normalization
"""


import collections
import re
import string
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd
from datasets import Dataset
from transformers import AutoConfig, AutoModelForQuestionAnswering, AutoTokenizer

ARTICLES_REGEX = re.compile(r"\b(a|an|the)\b", re.UNICODE)

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

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


def get_tokens(s):
    if not s:
        return []
    return normalize_answer(s).split()


def find_error_span(sentence1, sentence2):
    
    # Check if sentences are null and split them into words
    # sentence1_words = [] if pd.isnull(sentence1) else sentence1.split()
    # sentence2_words = [] if pd.isnull(sentence2) else sentence2.split()
    sentence1_words = [] if pd.isnull(sentence1) else get_tokens(sentence1)
    sentence2_words = [] if pd.isnull(sentence2) else get_tokens(sentence2)

    # Initialize indices lists
    diff_word_idx_sent1 = [idx for idx, word in enumerate(sentence1_words) if word not in sentence2_words] # not in 2
    diff_word_idx_sent2 = [idx for idx, word in enumerate(sentence2_words) if word not in sentence1_words] # not in 1

    # Identify contiguous differing span in sentence1
    if diff_word_idx_sent1:
        differing_parts_sentence1 = sentence1_words[min(diff_word_idx_sent1):max(diff_word_idx_sent1)+1]
    else:
        differing_parts_sentence1 = []

    # Identify contiguous differing span in sentence2
    if diff_word_idx_sent2:
        differing_parts_sentence2 = sentence2_words[min(diff_word_idx_sent2):max(diff_word_idx_sent2)+1]
    else:
        differing_parts_sentence2 = []

    return " ".join(differing_parts_sentence1), " ".join(differing_parts_sentence2)

def find_all_subsequences(text, sub_sequence, verbose=True):
    
    "get error span index in the full text -> idk where i should apply brackets"

    if len(sub_sequence)==0:
        return []
    
    text, sub_sequence = normalize_answer(text), normalize_answer(sub_sequence)
        

    start_indices = []
    start_index = text.find(sub_sequence)
    
    while start_index != -1:
        start_indices.append(start_index)
        # Move the search window to look for the next occurrence
        start_index = text.find(sub_sequence, start_index + 1) # if not found, then start_index will be assigned the value -1.
        
        # Check if the subsequence is found more than twice -> more than once!
        if len(start_indices) > 1:
            error_msg = f"\n\nThe subsequence '{sub_sequence}' is found more than once in the text. The last one is taken as span index."
            if verbose:
                error_msg = error_msg + f"\nText:\n{text}" + "\n------------------------------------------"
            # raise ValueError(error_msg)
            print(error_msg)
    
    return [] if len(start_indices)==0 else [start_indices[-1]]

def transform_error_span(example):
    original_error_span = example['Error Span']
    return {
        "answers": {
            "text": [original_error_span],
            "answer_start": find_all_subsequences(text=example['context'], sub_sequence=original_error_span)
        }
    }
    
######################### for UW only #########################

# def handle_redacts(full_text):
#     # <NAME/> <AGE/> <DATE/> <TIME/> 
#     # Miller unspecified age usual date usual time 
    
# prompt = "In the following clinical note, the sentence '{error sentence}' contains a medical error. " # -> yes. 

###########################################################################################

file_path_validate = "./data/Feb_1_2024_MS_Train_Val_Datasets/MEDIQA-CORR-2024-MS-ValidationSet-1-Full.csv"
file_path_train = "./data/Feb_1_2024_MS_Train_Val_Datasets/MEDIQA-CORR-2024-MS-TrainingData.csv"
file_path_validate_uw = "./data/Feb_5_2024_UW_Validation_Set_Updated/MEDIQA-CORR-2024-UW-ValidationSet-1-Full_Feb.csv"

res = None

full_text_as_input = False # ah if false "context" could end up being null. change to empty string. 

for file_path in [file_path_validate, file_path_train, file_path_validate_uw]: #, file_path_train, file_path_test]:
    df = pd.read_csv(file_path)

    df['question'] = "Which part in the given clinical note is clinically incorrect?"
        
    df['Error Span'], df['Corrected Span'] = None, None

    res = df.apply(lambda row: find_error_span(row['Error Sentence'], row['Corrected Sentence']), axis=1)

    df['Error Span'] = res.apply(lambda x: x[0]) 
    df['Corrected Span'] = res.apply(lambda x: x[1])
    
    if full_text_as_input:
        df.rename(columns={'Text ID': 'id', 'Text': 'context'}, inplace=True) # for error span prediction, 'context' could come from error sentence not the entire text. 
    else:
        df.rename(columns={'Text ID': 'id', 'Error Sentence': 'context'}, inplace=True) 
        
    df['context'] = df['context'].fillna('')

    hf_dataset = Dataset.from_pandas(df)

    hf_dataset = hf_dataset.map(transform_error_span) # adds answer_start and transform to squad format <- does get 'context' according to 'full_text_as_input' set. 

    # Assuming 'dataset' is your loaded dataset with a nested structure
    
    save_path = './biolinkbert/data/ms_val_processed'
    if 'Validation' in file_path:
        if '-MS-' in file_path:
            save_path = './biolinkbert/data/ms_val_processed'
        if '-UW-' in file_path:
            save_path = './biolinkbert/data/uw_val_processed'
    else:
        save_path = './biolinkbert/data/ms_train_processed'
        
    if full_text_as_input:
        save_path = save_path + '_full_text_as_input.json'
    else:
        save_path = save_path + '_sent_as_input.json'
        
    hf_dataset.to_json(save_path)
