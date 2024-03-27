import logging
import os
import sys


from dotenv import load_dotenv
load_dotenv(".env")

# sys.path.append(os.getcwd())

from api_trainer import APITrainer



result_file_path = '/home/co-chae/rds/hpc-work/mediqa_output/gpt_correction/gpt3/gpt_prediction_res_144056.csv'


def main() -> None:

    trainer = APITrainer()
    
    # trainer.predict()
    trainer.post_process_for_submission(result_df_or_path=result_file_path)


if __name__ == "__main__":

    main()
