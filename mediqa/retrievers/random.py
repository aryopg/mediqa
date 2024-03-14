import random
from typing import Dict

import numpy as np
import pandas as pd

from mediqa.dataset import MEDIQADataset

from .utils import tokenize_and_stem


class Random:
    """
    Randomly select k examples within the same section and type as the query

    """

    def __init__(self, corpus: MEDIQADataset, top_k=5, **kwargs) -> None:
        self.corpus_df = pd.DataFrame(corpus.data)
        self.top_k = top_k

    def get_document_scores(self, query, text_id) -> Dict[str, list]:
        # Given the section and type, narrow down the search space
        pos_docs = self.corpus_df.loc[
            (self.corpus_df["label_flags"] == "1") & (self.corpus_df["ids"] != text_id)
        ]["ids"].tolist()
        neg_docs = self.corpus_df.loc[
            (self.corpus_df["label_flags"] == "0") & (self.corpus_df["ids"] != text_id)
        ]["ids"].tolist()
        relevant_pos_examples = [
            {"id": s_id, "score": 1.0} for s_id in random.sample(pos_docs, self.top_k)
        ]
        relevant_neg_examples = [
            {"id": s_id, "score": 1.0} for s_id in random.sample(neg_docs, self.top_k)
        ]

        return {
            "pos": relevant_pos_examples,
            "neg": relevant_neg_examples,
        }
