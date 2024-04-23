
# cd mediqa
# python span_prediction_mcq/correction/preprocess_for_correction.py

import json
import re
import pandas as pd
import hydra
from omegaconf import DictConfig, OmegaConf


def find_closest_index(hits, target):
    """
    Finds the index of the number closest to the target in a list.
    
    Args:
    offsets (list of int/float): The list of offsets to search.
    target (int/float): The target number to find the closest to.
    
    Returns:
    int: The index of the closest number.
    """

    offset_ids, offsets = list(hits.keys()), list(hits.values())
    idx = min(range(len(offsets)), key=lambda idx: abs(offsets[idx] - target))
    
    return offset_ids[idx]

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

def get_error_sent_id_from_prediction(indexed_sents_parsed, predicted_error_span, predicted_start_offset):
    """
    Determines the sentence ID and sentence text from parsed sentences that best matches the predicted error span.

    Args:
    indexed_sents_parsed (dict): The parsed sentences with their IDs.
    predicted_error_span (str): The predicted error span.
    predicted_start_offset (int): The starting offset of the predicted error span.

    Returns:
    tuple: The sentence ID and the sentence text that matches the predicted error span.
    """
    sub_sequence = predicted_error_span  
    
    hits = {}
    total_length = 0  # This tracks the cumulative length of sentences to adjust indices properly
    
    for sent_id, sentence in indexed_sents_parsed.items():
        sentence_lower = sentence.lower()
        found_index = sentence_lower.find(sub_sequence)
        if found_index != -1:
            adjusted_index = total_length + found_index
            hits[sent_id] = adjusted_index
        total_length += len(sentence) + 1  # Add 1 for the space or newline that was originally between sentences
    
    # Find the sentence ID whose error span start index is closest to the predicted start offset
    closest_sent_id = find_closest_index(hits, predicted_start_offset) if hits else -1
    error_sentence = indexed_sents_parsed[closest_sent_id] if closest_sent_id != -1 else ""
    
    return (closest_sent_id, error_sentence.lower().strip())

# Identify error sentence based on predictions
def get_error_sentence(parsed_sents, error_span, start_offset):
    
    sub_sequence = trim_span(error_span.lower())
    closest_sent_id, error_sentence = get_error_sent_id_from_prediction(parsed_sents, sub_sequence, start_offset)
    
    return closest_sent_id, error_sentence

def get_first_valid_text(predictions):
    for prediction in predictions:
        if prediction["text"][0] != ".":
            return prediction["text"], prediction["offsets"]
    return "", [0, 0]

def blank_out_main_text(full_text, start_offset, end_offset):
    return full_text[:start_offset] + "<BLANK>" + full_text[end_offset:]

def trim_span(text):

    substrings = re.split(r'\. |\n', text) # text.split(". ")
    res = max(substrings, key=len, default="")  
    
    return res.strip()


@hydra.main(config_path='../conf', config_name='config_preprocess')
def main(configs: DictConfig):
    
    # Set which dataset to use
    which_dataset = configs.dataset  
    config = getattr(configs.paths, which_dataset)
    
    predicted_file = config.predicted
    original_file = config.original
    save_path = config.save_path
    encoding = config.encoding

    
    # Load data
    with open(predicted_file, 'r', encoding='utf-8') as f:
        data_predicted = json.load(f)
        
    df = pd.read_csv(original_file, encoding=encoding)

    df['sentences_parsed'] = df['Sentences'].apply(parse_sentences)
    df['predicted_error_span'], df['predicted_error_span_offsets'] = zip(*df['Text ID'].map(lambda x: get_first_valid_text(data_predicted[str(x)])))
    df['predicted_error_sent_id'], df['predicted_error_sent'] = zip(*df.apply(lambda row: get_error_sentence(row['sentences_parsed'], row['predicted_error_span'], row['predicted_error_span_offsets'][0]), axis=1))
    df['text_blanked_out'] = df.apply(lambda row: blank_out_main_text(row.loc['Text'], row.loc['predicted_error_span_offsets'][0], row.loc['predicted_error_span_offsets'][1]), axis=1)


    df.to_csv(save_path, index=False)

if __name__ == "__main__":
    main()
