# cd mediqa
# python span_prediction_mcq/train/preprocess.py
import collections
import re
import os
import string
from dataclasses import dataclass, field
from typing import Optional
import hydra
from omegaconf import DictConfig


import pandas as pd
from datasets import Dataset
from transformers import AutoConfig, AutoModelForQuestionAnswering, AutoTokenizer

ARTICLES_REGEX = re.compile(r"\b(a|an|the)\b", re.UNICODE)

def normalize_answer(s):

    def remove_articles(text):
        return ARTICLES_REGEX.sub(" ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return lower(s) 

def get_tokens(s):
    if not s:
        return []
    return normalize_answer(s).split()

def trim_span(text):

    substrings = text.split(". ")
    res = max(substrings, key=len, default="")  
    
    return res


def parse_sentences(indexed_sents): 
    sents = indexed_sents.split('\n')
    sents = [item.strip() for item in sents if item.strip()]

    res = []
    for sent in sents:
        if not sent[0].isdigit() or " " not in sent:
            idx, prev_sent = res.pop()
            res.append((idx, prev_sent + " " + sent))
        else:
            idx, sent = sent.split(' ', 1)
            res.append((int(idx), sent))

    return dict(res)

def get_error_context(indexed_sents, error_sent_id): 
    """
    Retrieve the context around a specified sentence, identified by its ID, within a collection
    of indexed sentences.

    Parameters:
    indexed_sents (list of str): Indexed sentences from the dataset (column 'Sentences').
    error_sent_id (int): The ID of the sentence around which context is to be retrieved.

    Returns:
    str: A string composed of the sentences immediately before, the target sentence, and 
         immediately after the target sentence, if they exist.
    """
    indexed_dict = parse_sentences(indexed_sents)
    
    context_left = indexed_dict.get(error_sent_id - 1, "")
    error_sentence = indexed_dict.get(error_sent_id, "")
    context_right = indexed_dict.get(error_sent_id + 1, "")

    full_context = " ".join(filter(None, [context_left, error_sentence, context_right]))

    return full_context.strip()


def find_error_span(sentence1, sentence2):
    """
    Identifies and returns the contiguous differing spans between two sentences.

    Parameters:
    sentence1 (str): The first sentence for comparison.
    sentence2 (str): The second sentence for comparison.

    Returns:
    tuple: A tuple containing the differing spans from sentence1 and sentence2.
    """

    # Tokenize sentences or return empty list if sentence is null
    def get_tokens(sentence):
        return [] if pd.isnull(sentence) else sentence.split()

    sentence1_words = get_tokens(sentence1)
    sentence2_words = get_tokens(sentence2)

    # Create sets of words for efficient comparison
    set1 = set(sentence1_words)
    set2 = set(sentence2_words)

    # Find indices of words in each sentence that don't appear in the other
    diff_indices1 = [i for i, word in enumerate(sentence1_words) if word not in set2]
    diff_indices2 = [i for i, word in enumerate(sentence2_words) if word not in set1]

    # Extract contiguous differing spans using the found indices
    span1 = " ".join(sentence1_words[min(diff_indices1):max(diff_indices1)+1]) if diff_indices1 else ""
    span2 = " ".join(sentence2_words[min(diff_indices2):max(diff_indices2)+1]) if diff_indices2 else ""
    # .lower().split(". ")[0]  

    return trim_span(span1), trim_span(span2)


def find_all_subsequences(text, sub_sequence, indexed_sents, gt_error_sent_id, verbose=True):
    """
    Finds all start indices of subsequences within the text and identifies the correct occurrence based on ground truth.

    Parameters:
    text (str): The main text to search within.
    sub_sequence (str): The subsequence to find within the text.
    indexed_sents (list of str): Indexed sentences that might include the subsequence.
    gt_error_sent_id (int): Ground truth error sentence ID to help identify the correct subsequence.
    verbose (bool): Flag to turn on verbose output for debugging.

    Returns:
    list of int: Start indices of the found subsequence, typically one, unless errors are caught.
    """
    if not sub_sequence:
        return []

    # Normalize and search for subsequence
    text, sub_sequence = normalize_answer(text), normalize_answer(sub_sequence)
    start_indices = [m.start() for m in re.finditer(re.escape(sub_sequence), text)]

    if len(start_indices) > 1:
        # Parse sentences and find indices close to the error sentence ID
        sents_parsed = parse_sentences(indexed_sents)
        error_sent_text = sents_parsed.get(gt_error_sent_id, "")

        # Find which index corresponds to the ground truth error sentence
        correct_index = next((i for i, start in enumerate(start_indices) if text[start:start+len(sub_sequence)] in error_sent_text), -1)
        # sub_sequence = re.split(r'[. \n]+', predicted_error_span.lower())[0]

        if correct_index == -1:
            if verbose:
                print(f"Predicted error span not found in any of the indexed sentences.")
            return []
        start_indices = [start_indices[correct_index]]

    return start_indices


def transform_error_span(example):
    original_error_span = example['Error Span']
    return {
        "answers": {
            "text": [original_error_span],
            "answer_start": find_all_subsequences(text=example['context'], sub_sequence=original_error_span, indexed_sents=example['Sentences'], gt_error_sent_id=int(example['Error Sentence ID'])) # find_all_subsequences(text, sub_sequence, indexed_sents, gt_error_sent_id, verbose=True):
        }
    }

def preprocess_data(file_path, prediction=False, error_samples_only=True):
    """
    Process the dataset by identifying error spans and correcting spans, and save in JSON format.

    Parameters:
    file_path (str): Path to the dataset CSV file.
    error_samples_only (bool): Whether to filter the dataset to only include samples with errors.
    full_text_as_input (bool): Whether to use the full text as the context for the model input.

    Returns:
    None: Saves the processed files as JSON.
    """
    
    if not prediction:
        # import pdb; pdb.set_trace()
        df = pd.read_csv(file_path)

        if error_samples_only:
            df = df[df['Error Sentence ID'] != -1]

        df['Error Span'], df['Corrected Span'] = zip(*df.apply(lambda row: find_error_span(row['Error Sentence'], row['Corrected Sentence']), axis=1))

        df.rename(columns={'Text ID': 'id', 'Text': 'context'}, inplace=True)
        
        df['context'] = df['context'].fillna('')
        df['question'] = "Which part in the given clinical note is clinically incorrect?"
        
        hf_dataset = Dataset.from_pandas(df)
        hf_dataset = hf_dataset.map(transform_error_span)
        
    
    else:
        df = pd.read_csv(file_path, encoding='MacRoman')
        df.rename(columns={'Text ID': 'id', 'Text': 'context'}, inplace=True)
        
        df['context'] = df['context'].fillna('')
        df['question'] = "Which part in the given clinical note is clinically incorrect?"
        hf_dataset = Dataset.from_pandas(df)

    save_path = construct_save_path(file_path, error_samples_only)
    hf_dataset.to_json(save_path)
    print(f"Data saved to {save_path}")

def construct_save_path(file_path, error_only, full_text=True):
    """
    Construct the path where the processed dataset will be saved based on the parameters.

    Parameters:
    file_path (str): Path to the original dataset file.
    error_only (bool): Whether the dataset includes only error samples.
    full_text (bool): Whether the full text is used as context.

    Returns:
    str: The constructed save path for the processed dataset.
    """
    base_path = os.path.dirname(file_path)
    
    if 'Validation' in file_path:
        if '-MS-' in file_path:
            split = 'ms_val_processed'
        elif '-UW-' in file_path:
            split = 'uw_val_processed'
            
    elif 'Training' in file_path:
        split = 'ms_train_processed'
    elif 'Test-Set' in file_path:
        split = 'test_set_processed' 
        
    subset = "_all_instances" if not error_only else "_error_only"

    file_name = '_full_text_as_input.json' 

    return f"{base_path}/{split}{subset}{file_name}"


@hydra.main(config_path='../conf', config_name='config_preprocess')
def main(cfg: DictConfig):
    
    base_path = cfg.paths.base_path
    ms_train_path = cfg.paths.ms_train.original
    ms_val_path = cfg.paths.ms_val.original
    uw_val_path = cfg.paths.uw_val.original
    test_path = cfg.paths.test.original
    
    # preprocess_data(os.path.join(base_path, ms_train_path), error_samples_only=True, prediction=False)
    # preprocess_data(os.path.join(base_path, ms_val_path), error_samples_only=True, prediction=False)
    # preprocess_data(os.path.join(base_path, uw_val_path), error_samples_only=False, prediction=False)
    # preprocess_data(os.path.join(base_path, test_path), error_samples_only=False, prediction=True)
    
    preprocess_data(ms_train_path, error_samples_only=True, prediction=False)
    preprocess_data(ms_val_path, error_samples_only=True, prediction=False)
    preprocess_data(uw_val_path, error_samples_only=False, prediction=False)
    preprocess_data(test_path, error_samples_only=False, prediction=True)

if __name__ == "__main__":
    main()
