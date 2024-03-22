import json
import os
from difflib import SequenceMatcher

import pandas as pd

pred_res_json = "/Users/chaeeunlee/Documents/VSC_workspaces/mediqa/biolinkbert/output/eval_predictions.json"

csv_file_path = "/Users/chaeeunlee/Documents/VSC_workspaces/mediqa/data/Feb_1_2024_MS_Train_Val_Datasets/MEDIQA-CORR-2024-MS-ValidationSet-1-Full.csv"
df = pd.read_csv(csv_file_path)

def parse_split_ids(indexed_sents): # cos splits aren't always based on sentences. 
    """
    indexed_sents: indexed sents from dataset (col 'Sentences')
    res = tuple (id, sent)
    
    """
    str_lst = indexed_sents.split('\n')
    
    str_lst = [item.strip() for item in str_lst]
    str_lst = [string for string in str_lst if string != ""]
    
    # print(str_lst)

    # res = [(item.split(' ', 1)[0], item.split(' ', 1)[1:]) for item in str_lst]

    res = []
    for indexed_sent in str_lst: # string
        
        if (not indexed_sent[0].isdigit()) or (" " not in indexed_sent):
            prev_idx, prev_sent = res.pop(-1)
            res.append( (prev_idx, prev_sent + " " + indexed_sent) )
        
        else:
            idx, sent = indexed_sent.split(' ', 1)[0], indexed_sent.split(' ', 1)[1:] 
            res.append((int(idx), sent[0]))
            # print(sent)

    return res


def get_overlap_ratio(s1, s2):
    """
    Calculate the ratio of overlapping characters between two strings.
    """
    matcher = SequenceMatcher(None, s1, s2)
    return matcher.find_longest_match(0, len(s1), 0, len(s2)).size

def get_most_overlap_index(predicted_error_sentence, indexed_sentences):
    """
    Get the index of the element in indexed_sentences with the most string overlap with predicted_error_sentence. -> lots of bracketing problem. 
    """
    max_overlap = 0
    max_overlap_index = -1
    
    # indexed_sentences

    for i, sentence in enumerate(indexed_sentences):
        # try:
        overlap = get_overlap_ratio(predicted_error_sentence, sentence[1][0])
        # except:
        #     print(f"sentence = {sentence}")
        #     continue
        
        if overlap > max_overlap:
            max_overlap = overlap
            max_overlap_index = i
            

    max_overlap_index = int(indexed_sentences[i][0])

    # if ". " in predicted_error_sentence:
    #     max_overlap_index -=1

    return max_overlap_index


df['Sentences'] = df['Sentences'].apply(parse_split_ids)

# Read the JSON file
# path = '/Users/chaeeunlee/Documents/VSC_workspaces/mediqa/biolinkbert/output/eval_predictions.json'
path = '/Users/chaeeunlee/Documents/VSC_workspaces/mediqa/biolinkbert/output/predict_nbest_predictions.json'
with open(path) as f:
    data = json.load(f)

# Extract just the values
values_list = [value[0] for value in data.values()]

first_non_period_texts = []

for key in data:
    for item in data[key]:
        if item["text"][0] != ".":
            first_non_period_texts.append(item["text"])
            break  # Break after adding the first non-period text for the current key

# first_non_period_texts now contains the first "text" value that is not "." for each key
# print(first_non_period_texts)
df['predicted error span'] = first_non_period_texts

df['predicted sent id'] = df.apply(lambda row: get_most_overlap_index(row['predicted error span'], row['Sentences']), axis=1)

filtered_df = df[df['Error Sentence ID'] != -1]

hits = (filtered_df['Error Sentence ID'] == filtered_df['predicted sent id']).sum()
# hits = (abs(filtered_df['Error Sentence ID']-filtered_df['predicted sent id'])<=1).sum()



total_rows = len(filtered_df)
accuracy = hits / total_rows

print(f"Accuracy: {accuracy:.2f}")