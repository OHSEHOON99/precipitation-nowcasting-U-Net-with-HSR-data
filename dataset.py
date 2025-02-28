from pathlib import Path
import logging
import os
from torch.utils.data import DataLoader
from TSDataset import TSDataset, ToTensor


def load_datasets(data_path, transform=ToTensor()):
    try:
        data_dir = Path(data_path)
        train_set = TSDataset(data_dir / 'train', transform=transform)
        val_set = TSDataset(data_dir / 'valid', transform=transform)
    except FileNotFoundError as e:
        logging.error(f"Dataset path not found: {e}")
        raise
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        raise

    return train_set, val_set


def create_dataloaders(train_set, val_set, batch_size, device):
    loader_args = dict(batch_size=batch_size, 
                       num_workers=os.cpu_count(), 
                       pin_memory=(device.type == 'cuda'))
    
    train_loader = DataLoader(train_set, shuffle=True, drop_last=False, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)
    
    return train_loader, val_loader