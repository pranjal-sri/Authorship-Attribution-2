import os
import wandb
import torch
from dotenv import load_dotenv

class WandbLogger:

    def is_distributed(self) -> bool:
        return torch.distributed.is_available() and torch.distributed.is_initialized()
    def is_main_process(self) -> bool:
        if not self.is_distributed():
            return True
        return torch.distributed.get_rank() == 0
    
    def __init__(self):
        load_dotenv()
        self.api_key = os.environ.get('WANDB_API_KEY')
        self.project = os.environ.get('WANDB_PROJECT')
        
        if not self.api_key or not self.project:
            raise ValueError("WANDB_API_KEY and WANDB_PROJECT environment variables must be set")
        
        wandb.login(key=self.api_key)
        
    def init(self, config=None):
        if self.is_main_process():
            wandb.init(
            project=self.project,
            config=config
        )

        if self.is_distributed():
            torch.distributed.barrier()
    
    def log(self, metrics, step=None):
        if self.is_main_process():
            wandb.log(metrics, step=step)
        if self.is_distributed():
            torch.distributed.barrier()
            
        
    
    def finish(self):
        if self.is_main_process():
            wandb.finish() 

        if self.is_distributed():
            torch.distributed.barrier()
