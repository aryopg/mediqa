# MEDIQA 2024


## Resources

- [Competition Website](https://sites.google.com/view/mediqa2024/mediqa-corr)
- [Competition CodaBench Page](https://www.codabench.org/competitions/1900/)
- [Task Github Repo](https://github.com/abachaa/MEDIQA-CORR-2024/tree/main)


## How to run

### Create Environment
```bash
conda env create -f environment.yml
conda activate mediqa
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

## Experiment


### CoT few shot

We generated CoT reasonings using GPT3.5 model.
We prompted the model to reason why the groundtruth correction is correct given the incorrect clinical text.

```bash
# Generate Brief CoT reasons
python scripts/generate_cot_reasons.py experiment=cot_brief_2_shot/gpt35
# Generate Long CoT reasons
python scripts/generate_cot_reasons.py experiment=cot_long_2_shot/gpt35
# Generate SOAP CoT reasons
python scripts/generate_cot_reasons.py experiment=cot_soap_2_shot/gpt35
```

Admittedly, the chosen config file is "hacky" because actually all we need from the config files are the name (`cot_brief`, `cot_long`, or `cot_soap`) and path to store the CoT reasons.

### Baselines

```bash
python scripts/main.py experiment=EXPERIMENT_NAME
```

`EXPERIMENT_NAME` should be replaced by the path to the experiment that you'd like to run. For instance, if you want to run our best performing solution (8-shot + Brief CoT + Type Hint + Span Hint), you should do:

```bash
python scripts/main.py experiment=cot_brief_8_shot/gpt35_with_span_hint
```

You can inspect the YAML config file by navigating to: `configs/experiment/cot_brief_8_shot/gpt35_with_span_hint.yaml`
