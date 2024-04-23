# cd mediqa
# python span_prediction_mcq/correction/main.py

import os
import sys
import hydra
from omegaconf import DictConfig, OmegaConf

from dotenv import load_dotenv
load_dotenv(".env")

from pipeline import MCQCorrection


@hydra.main(version_base=None, config_path='../conf', config_name='config_correction')
def main(configs: DictConfig) -> None:

    pipeline = MCQCorrection(model_name=configs.model_name, input_file=configs.correction_input, save_path=configs.save_path, num_opts=configs.num_opts)
    pipeline.predict()


if __name__ == "__main__":
    main()
