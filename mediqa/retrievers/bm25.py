from typing import Dict

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi

from mediqa.dataset import MEDIQADataset

from .utils import tokenize_and_stem


class BM25(BM25Okapi):
    """
    Slightly modified version of the BM25Okapi that takes into consideration the statement id, type, and section

    """

    def __init__(
        self,
        corpus: MEDIQADataset,
        tokenizer=None,
        k1=1.5,
        b=0.75,
        epsilon=0.25,
        top_k=5,
    ) -> None:
        self.corpus_df = pd.DataFrame(corpus.data)

        self.top_k = top_k

        corpus = []
        for text in self.corpus_df["texts"].values:
            processed_text = tokenize_and_stem(text)
            corpus += [processed_text]

        super().__init__(corpus, tokenizer, k1, b, epsilon)

    def get_document_scores(self, query, text_id) -> Dict[str, list]:
        # Given the section and type, narrow down the search space
        doc_ids = self.corpus_df.index.tolist()

        # Tokenize and stem query
        processed_query = tokenize_and_stem(query)

        # Get BM25 score
        scores = self.get_batch_scores(processed_query, doc_ids)

        # Rank documents based on scores
        ranked_documents = sorted(
            zip(doc_ids, scores), key=lambda x: x[1], reverse=True
        )

        # Print the most similar documents
        # k examples for positive and negative
        relevant_pos_examples = []
        relevant_neg_examples = []
        for idx, score in ranked_documents:
            doc = self.corpus_df.iloc[idx]
            if text_id == doc["ids"]:
                # Filter out the sentence itself if found in the ranking
                continue
            else:
                if doc["label_flags"] == "1":
                    # Take only top k examples per label
                    if len(relevant_pos_examples) >= self.top_k:
                        continue
                    relevant_pos_examples += [
                        {
                            "id": doc["ids"],
                            "score": score,
                        }
                    ]
                elif doc["label_flags"] == "0":
                    # Take only top k examples per label
                    if len(relevant_neg_examples) >= self.top_k:
                        continue
                    relevant_neg_examples += [
                        {
                            "id": doc["ids"],
                            "score": score,
                        }
                    ]

            # Take only top k examples
            if (
                len(relevant_pos_examples) >= self.top_k
                and len(relevant_neg_examples) >= self.top_k
            ):
                break

        return {
            "pos": relevant_pos_examples,
            "neg": relevant_neg_examples,
        }
