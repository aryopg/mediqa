"""
create dataset compatible with squad training code for span prediction. 
1. "answers": { "text": [original_error_span], "answer_start": find_all_subsequences(text=example['context'], sub_sequence=original_error_span) }
2. ah no we don't need multiple span option. this is kind of it. 
3. and then correction pipeline to do the correction. could use harness? i think multi-turn is necessary. 

4. i need the same thing for UW. 
5. maybe add redact function for UW
"""




import collections
import re
import string
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd
from datasets import Dataset, concatenate_datasets
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

    # try:
    #     res = [start_indices[0]]
    # except:
    #     print(f"hmm {sub_sequence}, {text}")
    
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

def handle_redacts(full_text):
    # <NAME/> <AGE/> <DATE/> <TIME/> 
    # Miller unspecified age usual date usual time 
    
prompt = "In the following clinical note, the sentence '{error sentence}' contains a medical error. " # -> yes. 

###########################################################################################

file_path_validate = "./data/Feb_1_2024_MS_Train_Val_Datasets/MEDIQA-CORR-2024-MS-ValidationSet-1-Full.csv"
# file_path_train = "/Users/chaeeunlee/Documents/VSC_workspaces/mediqa/data/Feb_1_2024_MS_Train_Val_Datasets/MEDIQA-CORR-2024-MS-TrainingData.csv"
file_path_validate_uw = "./data/Feb_5_2024_UW_Validation_Set_Updated/MEDIQA-CORR-2024-UW-ValidationSet-1-Full_Feb.csv"

res = None

for file_path in [file_path_validate, file_path_validate_uw]: #, file_path_train, file_path_test]:
    df = pd.read_csv(file_path)

    df['question'] = "Which part in the given clinical note is clinically incorrect?"
    df.rename(columns={'Text ID': 'id', 'Text': 'context'}, inplace=True) # for error span prediction, 'context' could come from error sentence not the entire text. 

    df['Error Span'], df['Corrected Span'] = None, None

    res = df.apply(lambda row: find_error_span(row['Error Sentence'], row['Corrected Sentence']), axis=1)

    df['Error Span'] = res.apply(lambda x: x[0]) 
    df['Corrected Span'] = res.apply(lambda x: x[1])

    hf_dataset = Dataset.from_pandas(df)

    hf_dataset = hf_dataset.map(transform_error_span) # adds answer_start and transform to squad format

    # Assuming 'dataset' is your loaded dataset with a nested structure
    if 'Validation' in file_path:
        if '-MS-' in file_path:
            hf_dataset.to_json('./biolinkbert/data/ms_val_processed.json')
        if '-UW-' in file_path:
            hf_dataset.to_json('./biolinkbert/data/ws_val_processed.json')
    else:
        hf_dataset.to_json('./biolinkbert/data/ms_train_processed.json')
