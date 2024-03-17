from dataclasses import dataclass, field
from typing import Optional

import pandas as pd
from datasets import Dataset
from transformers import AutoConfig, AutoModelForQuestionAnswering, AutoTokenizer

    # else:
    #     data_files = {}
    #     if data_args.train_file is not None:
    #         data_files["train"] = data_args.train_file
    #         extension = data_args.train_file.split(".")[-1]

    #     if data_args.validation_file is not None:
    #         data_files["validation"] = data_args.validation_file
    #         extension = data_args.validation_file.split(".")[-1]
    #     if data_args.test_file is not None:
    #         data_files["test"] = data_args.test_file
    #         extension = data_args.test_file.split(".")[-1]
    #     raw_datasets = load_dataset(extension, data_files=data_files, cache_dir=model_args.cache_dir) #field="data",

    # question_column_name = "question" if "question" in column_names else column_names[0]
    # context_column_name = "context" if "context" in column_names else column_names[1]
    # answer_column_name = "answers" if "answers" in column_names else column_names[2]
def find_error_span(sentence1, sentence2):
    # Check if sentences are null and split them into words
    sentence1_words = [] if pd.isnull(sentence1) else sentence1.split()
    sentence2_words = [] if pd.isnull(sentence2) else sentence2.split()

    # Initialize indices lists
    diff_word_idx_sent1 = [idx for idx, word in enumerate(sentence1_words) if word not in sentence2_words]
    diff_word_idx_sent2 = [idx for idx, word in enumerate(sentence2_words) if word not in sentence1_words]

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

    if len(sub_sequence)==0:
        return []
        

    start_indices = []
    start_index = text.find(sub_sequence)
    
    while start_index != -1:
        start_indices.append(start_index)
        # Move the search window to look for the next occurrence
        start_index = text.find(sub_sequence, start_index + 1)
        
        # Check if the subsequence is found more than twice -> more than once!
        if len(start_indices) > 1:
            error_msg = f"The subsequence '{sub_sequence}' is found more than once in the text."
            if verbose:
                error_msg = error_msg + f"\nText:\n{text}" + "\n------------------------------------------"
            # raise ValueError(error_msg)
            print(error_msg)

    try:
        res = [start_indices[0]]
    except:
        print(f"hmm {sub_sequence}, {text}")
    
    return res

def transform_error_span(example):
    original_error_span = example['Error Span']
    return {
        "answers": {
            "text": [original_error_span],
            "answer_start": find_all_subsequences(text=example['context'], sub_sequence=original_error_span)
        }
    }

###########################################################################################

file_path_validate = "/Users/chaeeunlee/Documents/VSC_workspaces/mediqa/data/Feb_1_2024_MS_Train_Val_Datasets/MEDIQA-CORR-2024-MS-ValidationSet-1-Full.csv"

for file_path in [file_path_validate]:#, file_path_train, file_path_test]:
    df = pd.read_csv(file_path_validate)

    df['question'] = "Which part in the given clinical note is clinically incorrect?"
    df.rename(columns={'Text ID': 'id', 'Text': 'context'}, inplace=True)

    df['Error Span'], df['Corrected Span'] = None, None

    # df.loc[df['Corrected Sentence'].isna(), 'Correction'] = "NA"

    res = df.apply(lambda row: find_error_span(row['Error Sentence'], row['Corrected Sentence']), axis=1)
    # df['Correction'] = df.apply(lambda row: find_error_span(row['Error Sentence'], row['Corrected Sentence']) if pd.notnull(row['Corrected Sentence']) else None, axis=1)

    df['Error Span'] = res.apply(lambda x: x[0]) # the fuck is this syntax? -> ah so res is two-column df and this one's just taking the first col and adding that to df. 
    df['Corrected Span'] = res.apply(lambda x: x[1])

    hf_dataset = Dataset.from_pandas(df)

    hf_dataset = hf_dataset.map(transform_error_span)

    # Assuming 'dataset' is your loaded dataset with a nested structure
    hf_dataset.to_json('./biolinkbert/data/ms_val_processed.json')


    ## Saving logicd

    # waht are some other stuff that have to be done programmatically?
    # 