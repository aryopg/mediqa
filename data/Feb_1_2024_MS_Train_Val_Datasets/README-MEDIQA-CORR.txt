MEDIQA-CORR 2024 
https://sites.google.com/view/mediqa2024/mediqa-corr

The MS Training Set contains 2,189 clinical texts. 
The MS Validation Set (#1) contains 574 clinical texts. 
The UW Validation Set (#2) contains 160 clinical texts. 

=================
Task Description
=================
Each clinical text is either correct or contains one error. 

The task consists in: 
(a) predicting the error flag (1: the text contains an error, 0: the text has no errors),
and for flagged texts (with error):
(b) extracting the sentence that contains the error, and (c) generating a corrected sentence. 

=========
Test Set
=========

The test will include clinical texts from the MS and UW datasets. 


The test set will be a csv file (similar to MEDIQA-CORR-2024-MS-ValidationSet-1-Input.csv) 

and will contain the following 3 items: Text ID, Text, and Sentences.


===============
Run Submission
===============

- Each team is allowed to submit a maximum of 10 runs.
- The run file should be a TEXT file and contains one line per <Text-ID>.
- Each <Text-ID> must be included in the run file exactly once.

- The submission format should follow the data format and consists of:
[Text ID] [Error Flag] [Error sentence ID or -1 for texts without errors] [Corrected sentence or NA for texts without errors]


E.g.:
text-id-1 0 -1 NA
text-id-2 1 8 "correction of sentence 8..."
text-id-3 1 6 "correction of sentence 6..."
text-id-4 0 -1 NA
text-id-5 1 15 "correction of sentence 15..."

