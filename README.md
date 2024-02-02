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
```

### Dataset download

Download datasets from the MEDIQA website (`Jan_26_2024_MS_Datasets` and `Jan_31_2024_UW_Dataset`), place them in a `data` folder such that the structure would look like this:
```
data/Jan_26_2024_MS_Datasets/...
data/Jan_31_2024_UW_Dataset/...
```

### Experiment!

```bash
python scripts/main.py experiment=0_shot/mistral_7b_instruct
```