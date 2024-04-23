

## BioLinkBERT Error Span Prediction + LLM Error Correction with MCQ Prompts

### Error Span Prediction

To preprocess the training, validation, and test datasets for the BioLinkBERT error span prediction, use the following script:

```bash
python span_prediction_mcq/train/preprocess.py
```

You can inspect the YAML config file by navigating to: `./span_prediction_mcq/conf/conf_training/exp/train.yaml`.

Start the training process with:

```bash
python span_prediction_mcq/train/main.py +exp=train
```
To run predictions, use:

```bash
python span_prediction_mcq/train/main.py +exp=predict model.model_name_or_path=output/train_res/checkpoint-directory
```
### LLM Error Correction with MCQ Prompts

For MCQ-style error correction, first preprocess the dataset:

```bash
python span_prediction_mcq/correction/preprocess_for_correction.py dataset=test
```

Then, run the correction with:

```bash
python span_prediction_mcq/correction/main.py model_name=gpt-3.5-turbo num_opts=2
```