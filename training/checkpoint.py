import os
import torch
from pathlib import Path

class ModelCheckpoint:
    def __init__(self, checkpoint_dir: str, model_name: str = "model"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.model_name = model_name
        self.best_mrr = float('-inf')
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def is_distributed(self) -> bool:
        return torch.distributed.is_available() and torch.distributed.is_initialized()
    def is_main_process(self) -> bool:
        if not self.is_distributed():
            return True
        return torch.distributed.get_rank() == 0

    def _get_actual_model(self, model):
        # If the model is wrapped in DDP, get the actual model
        if hasattr(model, 'module'):
            model = model.module
        # If the model is compiled, get the original model
        if hasattr(model, '_orig_mod'):
            model = model._orig_mod
        return model

    def save_checkpoint(self, model, optimizer, epoch, mrr, loss):
        if self.is_main_process() and mrr > self.best_mrr:
            self.best_mrr = mrr
            model = self._get_actual_model(model)
            checkpoint_path = self.checkpoint_dir / f"{self.model_name}_best.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'mrr': mrr,
                'loss': loss
            }, checkpoint_path)
            print(f"New best model saved with MRR: {mrr:.4f} at {checkpoint_path}")
        
        if self.is_distributed():
            torch.distributed.barrier()

    def load_checkpoint(self, model, optimizer):
        checkpoint_path = self.checkpoint_dir / f"{self.model_name}_best.pth"
        epoch, mrr, loss = None, None, None
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path)
            model = self._get_actual_model(model)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epoch = checkpoint['epoch']
            mrr = checkpoint['mrr']
            loss = checkpoint['loss']
            print(f"Loaded checkpoint from {checkpoint_path} (epoch {epoch}, MRR: {mrr:.4f})")

            if self.is_distributed():
                torch.distributed.barrier()
            return epoch, mrr, loss
            
        else:
            print(FileNotFoundError(f"No checkpoint found at {checkpoint_path}"))
            return None, None, None
        # Ensure all processes wait until loading is done
        
        