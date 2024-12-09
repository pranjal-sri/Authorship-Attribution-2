import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
import os
import random
import numpy as np

from transformers import AutoTokenizer
import json
from functools import partial

from models.granular_roberta import GranularRoberta
from training.trainer import Trainer, TrainingConfig
from training.losses import MNRL_loss
from training.schedulers import WarmupStepWiseScheduler, LinearScheduler
from training.callbacks import EpochCallback
from data.dataset import ReutersRSTDataset
from training.checkpoint import ModelCheckpoint
from dotenv import load_dotenv

load_dotenv()

# Global variables for DDP setup
DDP_ENABLED = 'RANK' in os.environ and 'WORLD_SIZE' in os.environ and 'LOCAL_RANK' in os.environ
LOCAL_RANK = int(os.environ.get('LOCAL_RANK', 0))
RANK = int(os.environ.get('RANK', 0))
IS_MAIN_PROCESS = not DDP_ENABLED or RANK == 0
DEVICE = f"cuda:{LOCAL_RANK}" if DDP_ENABLED else ("cuda" if torch.cuda.is_available() else "cpu")
TO_COMPILE = False # Weird bug, compiling the model causes the training to fail

# Configuration constants
TRAIN_VAL_SPLIT_FILE = '/local/nlp/pranjal_sri/dev/authorship-attribution/train_validation_split_reuters.json'

DATASET_BASE_PATH = '/local/nlp/pranjal_sri/dev/authorship-attribution/ReutersDataset/content/ReutersRST_Dataset'
BASE_MODEL_NAME = "sentence-transformers/paraphrase-distilroberta-base-v1"
CHECKPOINT_DIR = 'checkpoints_11292924_run1'
MODEL_NAME = 'granular_roberta'
RUN_NAME = "SGD-test"
NUM_EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 1e-2
NUM_WORKERS = 12
TO_LOG = True
USE_AUTOCAST = True
AUTOCAST_DTYPE = torch.bfloat16
SCHEDULE = {
    'START_LR': 1e-4,
    'WARMED_UP_SCHEDULE': (0.2, 1e-3, 'linear_only', None),
    'FINE_TUNING_SCHEDULE': [
        (0.3, 5e-5, 'last_n_encoder', 1),
        (0.4, 5e-5, 'last_n_encoder', 2),
        (0.5, 1e-5, 'last_n_encoder', 3),
        (0.6, 5e-6, 'last_n_encoder', 4),
        (0.7, 1e-6, 'last_n_encoder', 5),
        (0.8, 1e-7, 'last_n_encoder', 6),
    ]
}

DDP_BACKEND = "gloo"
if IS_MAIN_PROCESS:
    print(f"DDP_ENABLED: {DDP_ENABLED}")

def set_seeds(seed=42):
    """Set seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Optional: for complete determinism (may slow down training)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def setup_ddp():
    
    if not DDP_ENABLED:
        return
    if IS_MAIN_PROCESS:
        print("\n=== Setting up DDP ===")
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    dist.init_process_group(backend=DDP_BACKEND)
    torch.cuda.set_device(LOCAL_RANK)

def cleanup_ddp():
    if DDP_ENABLED:
        dist.destroy_process_group()

def main():
    # DDP setup
    setup_ddp()
    try:
        # Set seeds before any other operations
        set_seeds()
        
        if IS_MAIN_PROCESS:
            print("\n=== Starting Training Setup ===")
        
        # Load train-validation split
        with open(TRAIN_VAL_SPLIT_FILE, 'r') as f:
            train_validation_files = json.load(f)

        if IS_MAIN_PROCESS:
            print(f"Loaded train-validation split: {len(train_validation_files['train'])} train, {len(train_validation_files['valid'])} validation files")
        
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
        train_ds = ReutersRSTDataset(tokenizer, train_validation_files['train'],
                                    DATASET_BASE_PATH)
        
        if IS_MAIN_PROCESS:
            print(f"\nDataset Setup:")
            print(f"Training samples: {len(train_ds)}")

        # Create samplers and dataloaders
        train_sampler = DistributedSampler(train_ds, shuffle=False) if DDP_ENABLED else None
        # Create training dataloader
        train_dataloader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                                    sampler=train_sampler,
                                    shuffle=False,
                                    num_workers=NUM_WORKERS,
                                    pin_memory=True)


        # Create validation dataset and dataloader (only on main process if DDP enabled)

        valid_ds = ReutersRSTDataset(tokenizer, train_validation_files['valid'],
                                    DATASET_BASE_PATH)
        val_dataloader = DataLoader(valid_ds, batch_size=BATCH_SIZE,
                                shuffle=False,
                                num_workers=NUM_WORKERS,
                                pin_memory=True)
        
        # Initialize model and move to device
        model = GranularRoberta()
        model.to(DEVICE)
        if TO_COMPILE:
            model = torch.compile(model)
        
        # Wrap model with DDP if enabled
        if DDP_ENABLED:
            if IS_MAIN_PROCESS:
                print("Initializing DDP wrapper")
            model = DDP(model, device_ids=[LOCAL_RANK], find_unused_parameters=True)
        
        raw_model = model.module if DDP_ENABLED else model

        if IS_MAIN_PROCESS:
            print(f"\nInitialized model and moved to device {DEVICE}")
            print(f"DDP wrapper applied: {DDP_ENABLED}")
            print(f"Model compiled: {TO_COMPILE}")


        # Setup datasets

        if IS_MAIN_PROCESS:     
            print(f"Validation samples: {len(valid_ds)}")
    


        # Setup optimizer and scheduler
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
        # optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, nesterov = True, momentum = 0.9)

        warmup_r, warmup_lr, warmup_mode, warmup_num_encoder_layers = SCHEDULE['WARMED_UP_SCHEDULE']
        scheduler = WarmupStepWiseScheduler(initial_lr= SCHEDULE['START_LR'], 
                                lr_schedule= [(int(r*NUM_EPOCHS), lr) for (r, lr, _, _) in SCHEDULE['FINE_TUNING_SCHEDULE']], 
                                warmup_steps= int(warmup_r * NUM_EPOCHS),
                                warmup_lr= warmup_lr)
        # scheduler = LinearScheduler(initial_lr=1e-3, end_lr=1e-5, num_epochs=NUM_EPOCHS, sqrt=False)

        # Initialize checkpoint handler
        checkpoint = ModelCheckpoint(
            checkpoint_dir=CHECKPOINT_DIR,
            model_name=MODEL_NAME
        )

        # Initialize trainer
        trainer = Trainer(
            model=model,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            loss_fn=MNRL_loss,
            optimizer=optimizer,
            device=DEVICE,
            scheduler=scheduler,
            checkpoint=checkpoint,
        )

        # Training configuration
        train_config = TrainingConfig(
            epochs=NUM_EPOCHS,
            validate=True,
            validate_every_n_epochs=1,
            to_log=TO_LOG,
            use_autocast=USE_AUTOCAST,
            autocast_dtype=AUTOCAST_DTYPE
        )

        # Setup epoch callbacks for fine-tuning
        epoch_callback = EpochCallback()
        
        # Add initial warmup callback
        def callback_fn(trainer, mode, num_layers):
            if DDP_ENABLED:
                dist.barrier()
            model = trainer.model
            if DDP_ENABLED:
                model = model.module
            if TO_COMPILE:
                model = model._orig_mod
            new_model = GranularRoberta()
            new_model.load_state_dict(model.state_dict())
            model = new_model
                
            
            model.set_training_granularity(
                mode,
                num_encoder_layers=num_layers
            )
            model.to(DEVICE)
            if TO_COMPILE:
                model = torch.compile(model)
            if DDP_ENABLED:
                dist.barrier()
                model = DDP(model, device_ids=[LOCAL_RANK], find_unused_parameters=True)
            trainer.model = model
            trainer.optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)


        epoch_callback.add_callback(
            0,
            partial(callback_fn, trainer=trainer, mode=warmup_mode, num_layers=warmup_num_encoder_layers)
        )
        
        # Add fine-tuning schedule callbacks
        for (r, _, mode, num_encoder_layers) in SCHEDULE['FINE_TUNING_SCHEDULE']:
            epoch = int(r * NUM_EPOCHS)
            num_layers = (int(num_encoder_layers) 
                        if num_encoder_layers is not None 
                        else None)
            
            epoch_callback.add_callback(
                epoch,
                partial(callback_fn, trainer=trainer, mode=mode, num_layers=num_encoder_layers)
            )
        
        

        if IS_MAIN_PROCESS:
            print("\nTraining Schedule:")
            print(f"Initial warmup: {warmup_r*100}% of epochs with lr={warmup_lr}")
            print("Fine-tuning stages:")
            for idx, (r, lr, mode, layers) in enumerate(SCHEDULE['FINE_TUNING_SCHEDULE'], 1):
                print(f"  Stage {idx}: At {r*100}% epochs - lr={lr}, mode={mode}, layers={layers}")

        # Before starting training
        if IS_MAIN_PROCESS:
            print("\n=== Starting Training ===")
            print(f"Total epochs: {NUM_EPOCHS}")
            print(f"Batch size: {BATCH_SIZE}")
            print(f"Initial learning rate: {LEARNING_RATE}")
            print("=" * 50 + "\n")

        # Train the model
        torch.set_float32_matmul_precision('high')
        training_losses, val_losses, val_mrrs = trainer.train(train_config, epoch_callback)
        
        if IS_MAIN_PROCESS:
            print("\n=== Training Completed ===")
            print(f"Final training loss: {training_losses[-1]:.4f}")
            if val_losses:
                print(f"Final validation loss: {val_losses[-1]:.4f}")
                print(f"Final validation MRR: {val_mrrs[-1]:.4f}")

    # Cleanup DDP if enabled
    finally:
        cleanup_ddp()

if __name__ == "__main__":
    main()