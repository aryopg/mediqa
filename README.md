# MEDIQA 2024


## Resources

- [Competition Website](https://sites.google.com/view/mediqa2024/mediqa-corr)
- [Competition CodaBench Page](https://www.codabench.org/competitions/1900/)
- [Task Github Repo](https://github.com/abachaa/MEDIQA-CORR-2024/tree/main)


## How to run

### Create Environment
```bash
conda env create -f environment.yml
```

### Create an environment file

To download from/upload to HF hub, create a `.env` file containing
```
- HF_DOWNLOAD_TOKEN
- HF_UPLOAD_TOKEN
- HF_USERNAME
- OPENAI_API_KEY
```

Check `.env.example`. You can use it to create your own `.env` file.

### Dataset download

Download datasets from the MEDIQA website (`Jan_26_2024_MS_Datasets` and `Jan_31_2024_UW_Dataset`), place them in a `data` folder such that the structure would look like this:
```
data/Jan_26_2024_MS_Datasets/...
data/Jan_31_2024_UW_Dataset/...
```

## Experiment!

### Baselines

```bash
python scripts/main.py experiment=2_shot/gpt35
```

### CoT few shot

We generated CoT reasonings using GPT3.5 model.
We prompted the model to reason why the groundtruth correction is plausible given the incorrect clinical text.
We, then, manually postprocessed the reason to fit more into the narrative of a CoT reason.
For instance, around 100 instances of the generated reasons contain phrases that are not fitting to the CoT scenario, such as "The correction provided is more plausible ..." (notice that in the CoT scenario, the model generates the correction at the end, thus it does not make sense to say "The correction provided" early in the reason)

### Retrieval Augmented

We used ElasticSearch default BM25 settings as outlined [here](https://www.elastic.co/guide/en/elasticsearch/reference/current/index-modules-similarity.html#bm25):
```
k1 = 1.2
b = 0.75
discount_overlaps = True
```

- PMC-Patient ([link](https://huggingface.co/datasets/zhengyun21/PMC-Patients))
- PubMed
- UMLS Metathesaurus ([link](https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/data/kbs/2023-04-23/umls_2022_ab_cat0129.jsonl))


## Results

https://www.overleaf.com/1195375627krhjfkwgqdjr#c3a746