import os

import wandb


class WandbLogger:
    def __init__(self, wandb_args):

        self.wandb_args = wandb_args

        self.reset_wandb_env()
        wandb_init = self.setup_wandb()
        
        self.wandb_run = wandb.init(**wandb_init, settings=wandb.Settings(start_method='thread'))
    

    def setup_wandb(self):
        wandb_init = dict()
        wandb_init['project'] = self.wandb_args.project_name
        wandb_init['group'] = self.wandb_args.session_name
        wandb_init['name'] = self.wandb_args.name # f'training_{conf.experiment.dataset}'

        wandb_init['notes'] = self.wandb_args.session_name 
        os.environ['WANDB_START_METHOD'] = 'thread' 

        return wandb_init

    def reset_wandb_env(self):
        exclude = {'WANDB_PROJECT', 'WANDB_ENTITY', 'WANDB_API_KEY',}
        for k, v in os.environ.items():
            if k.startswith('WANDB_') and k not in exclude:
                del os.environ[k]

    def info(self, wandb_dict): # method name changed from log_wandb so that it's compatible with python logger option. 

        for metric in wandb_dict.keys():
            self.wandb_run.define_metric(metric)

        self.wandb_run.log(wandb_dict)

    def finish_wandb(self):
        wandb.finish()