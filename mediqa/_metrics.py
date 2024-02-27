def parse_split_ids(indexed_sents): # cos splits aren't always based on sentences. 
    """
    indexed_sents: indexed sents from dataset (col 'Sentences')
    res = tuple (id, sent)
    """
    str_lst = indexed_sents.split('\n')
    str_lst = [item.strip() for item in str_lst]

    res = [(int(item.split(' ', 1)[0]), item.split(' ', 1)[1:]) for item in str_lst]

    return res

def parse_detected_colname(col_name: str):
    """
    returns: from "Treatment 0" to ("treatments", 0)

    this func depends on the very specific way i named annoatated column names. 
    """
    categories = ['treatments', 'diagnoses', 'interpretation of exam results'] # 'medical exam results'
    for category in categories:
        if category in col_name:
            return (category, int(col_name.strip()[-1]))
        
    return None

def get_error_sent_indices(detected_df):
    """
    no_columns_list: each sub list is for each row, in which tuple of columns containing No is contained. 
    """

    # Initialize a list to store lists of column names
    no_columns_list = []

    # Iterate over DataFrame rows
    for index, row in detected_df.iterrows():

        no_columns_row = []

        for column, value in row.items(): # col name and the value in that specific row. 

            if value == "No":

                no_columns_row.append(column)
                # print(f"column = {column, type(column)}") # it is what i expected. 
                

        # [("treatments", 0), ]
        no_columns_row = [parse_detected_colname(item) for item in no_columns_row] if len(no_columns_row)>0 else []
        no_columns_list.append(no_columns_row)

    return no_columns_list

def evaluate_error_detection(detected_df):

    no_columns_list = get_error_sent_indices(detected_df)
    # Assert to ensure the input data is as expected
    assert len(no_columns_list) == detected_df.shape[0], "Length mismatch between no_columns_list and dataset size"

    row_scores = []

    for row_idx, detected_errors_in_row in enumerate(no_columns_list):
        # Initialize metrics for each row
        num_true_positive, num_false_positive = 0, 0 # at sentence level! reset at every row. hence row_score. 
        false_positive_detection_level = False

        # Check if there's supposed to be an error in the sentence
        has_error = detected_df['Error Sentence ID'][row_idx] != -1

        # If there are detected errors in the row
        if detected_errors_in_row:
            false_positive_detection_level = not has_error

            for category, sent_idx in detected_errors_in_row:
                # ref_sent comes from detected_df
                ref_sent = detected_df[category][row_idx].split('$')[sent_idx].strip()  # Assuming joined_to_list is a split operation

                # Check if the detected error matches the true error
                if ref_sent in str(detected_df['Error Sentence'][row_idx]):
                    num_true_positive += 1
                else:
                    num_false_positive += 1

        else: # no error was detected by the model 
            # No detected errors when there shouldn't be any counts as a true positive
            if not has_error: # ground truth label says no error. 
                num_true_positive += 1

        row_scores.append((has_error, len(detected_errors_in_row), num_true_positive, num_false_positive, false_positive_detection_level))

    # return row_scores


    ## For the entire dataset
    hit_count = 0
    correct_count = 0
    false_positive_detection_level_count = 0

    # Iterate over the list of tuples
    for tup in row_scores:
        # Check if the third element of the tuple is greater than 0
        if tup[2] > 0:
            hit_count += 1

    for tup in row_scores:
        # Check if the third element of the tuple is greater than 0
        if (tup[2] > 0) and (tup[3] == 0):
            correct_count += 1

    for tup in row_scores:
        # Check if the third element of the tuple is greater than 0
        if (tup[-1]==True):
            false_positive_detection_level_count += 1

    return {'num_rows': len(row_scores), 'num_hits': hit_count, 'num_correct': correct_count, 'num_correct_binary': len(row_scores) - false_positive_detection_level_count}


## eval func for binary error detection
## eval func for 


def error_sent_identification_acc(df): # from single sent detection

    # no_columns_list = []
    num_corr = 0
    # import pdb; pdb.set_trace()
    
    # Iterate over DataFrame rows
    for row_idx, row in df.iterrows():
        # print(row['Identified Error Sentence'])
        error_sent_pred = row['Identified Error Sentence']
        if not isinstance(error_sent_pred, str):
            continue
        ref_sent = error_sent_pred.strip() 
        if ref_sent in str(df['Error Sentence'][row_idx]):
            num_corr +=1

    acc = num_corr / df.shape[0]
    return acc

def binary_classification_acc(df):
    # has_error = 'Yes'
    num_corr=0
    for row_idx, row in df.iterrows():
        has_error_pred =  bool(row['Final Answer'] == 'Yes')
        print(f"row['Error Flag'] = {row['Error Flag']}")

        has_error_gt = bool(row['Error Flag']==1)

        if has_error_pred==has_error_gt:
            num_corr +=1

    acc = num_corr / df.shape[0]
    return acc






