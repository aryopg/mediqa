import argparse
import json
import os
import sys

sys.path.append(os.getcwd())

from typing import List

import hydra
import pandas as pd
from hydra.core.hydra_config import HydraConfig
from tqdm import tqdm

from mediqa import retrievers
from mediqa.configs import RetrieverConfigs, TrainingConfigs, register_base_configs
from mediqa.dataset import MEDIQADataset


def load_data(configs: TrainingConfigs):
    datasets = {}
    # Convert data into datasets
    for split in ["train", "valid", "test"]:
        print(f"Setup {split} data loader")
        datasets[split] = MEDIQADataset(
            configs.data,
            configs.prompt,
            configs.trainer,
            None,
            split=split,
        )

    return datasets


def retrieve(
    retriever_configs: RetrieverConfigs,
    query_corpus: MEDIQADataset,
    knowledge_corpus: MEDIQADataset,
):
    # Create BM25 object
    retriever = getattr(retrievers, retriever_configs.name)(
        knowledge_corpus, **retriever_configs.configs
    )

    query_corpus = pd.DataFrame(query_corpus.data)

    relevant_documents = {}
    for (
        text_id,
        text,
    ) in tqdm(query_corpus[["ids", "texts"]].values):
        documents = retriever.get_document_scores(text, text_id)
        relevant_documents[text_id] = documents

    return relevant_documents


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(configs: TrainingConfigs):
    print(configs)
    # Load data
    datasets: dict = load_data(configs)

    hydra_cfg = HydraConfig.get()
    outputs_dir = configs.retriever.icl_examples_dir
    os.makedirs(outputs_dir, exist_ok=True)

    for split in ["train", "valid", "test"]:
        # Run retrieval
        relevant_documents = retrieve(
            configs.retriever,
            query_corpus=datasets[split],
            knowledge_corpus=datasets["train"],
        )
        # Save the in context examples as a json file
        with open(
            os.path.join(
                outputs_dir,
                f"{split}.json",
            ),
            "w",
        ) as json_file:
            json.dump(relevant_documents, json_file)


if __name__ == "__main__":
    register_base_configs()
    main()
