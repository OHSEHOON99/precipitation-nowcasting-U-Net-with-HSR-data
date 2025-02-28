import logging
from datetime import datetime
import io
import os

import torch
from torch import optim
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import wandb
import numpy as np

from evaluate import Evaluater
from dataset import load_datasets, create_dataloaders
from loss import BalancedMSEMAE
from utils import *


def setup_wandb(model_name, N_epochs, N_batch, learning_rate, save_checkpoint, amp, run_name):
    wandb.login(key='a40ff5a6f0abbaf9771e3512b265e4d7dba35e37')
    experiment = wandb.init(project='0920', resume='allow', anonymous='must')
    
    wandb.run.name = run_name
    wandb.run.save()

    experiment.config.update({
        "model": model_name,
        "epochs": N_epochs,
        "batch_size": N_batch,
        "learning_rate": learning_rate,
        "save_checkpoint": save_checkpoint,
        "amp": amp
    })

    return experiment


class EarlyStopping(object):
    def __init__(self, patience=2, save_path="model.pth"):
        self._min_loss = np.inf
        self._patience = patience
        self._path = save_path
        self.__counter = 0
 
    def should_stop(self, model, loss):
        if loss < self._min_loss:
            self._min_loss = loss
            self.__counter = 0
            torch.save(model.state_dict(), self._path)
        elif loss > self._min_loss:
            self.__counter += 1
            if self.__counter >= self._patience:
                return True
        return False
   
    def load(self, model):
        model.load_state_dict(torch.load(self._path))
        return model
    
    @property
    def counter(self):
        return self.__counter


def train_model(
        data_path,
        save_path,
        model,
        device,
        N_epochs: int = 100,
        N_batch: int = 4,
        learning_rate: float = 1e-4,
        save_checkpoint: bool = False,
        amp: bool = False,
        weight_decay: float = 1e-8,
        gradient_clipping: float = 1.0,
        early_stopping_patience: int = 10,  # Early stopping patience setting
):
    # Log model information
    model_name = model.__class__.__name__
    run_name = f'{model_name}_epochs={N_epochs}_bs={N_batch}_lr={learning_rate}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'

    # wandb setup and initialization
    experiment = setup_wandb(model_name, N_epochs, N_batch, learning_rate, save_checkpoint, amp, run_name)

    # Load datasets
    train_set, val_set = load_datasets(data_path)
    train_loader, val_loader = create_dataloaders(train_set, val_set, N_batch, device)

    # Training setup
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    criterion = BalancedMSEMAE()
    evaluater = Evaluater(seq_len=model.n_classes)
    global_step = 0
    grad_scaler = torch.amp.GradScaler(enabled=amp and device.type == 'cuda')

    # Create EarlyStopping object
    early_stopping = EarlyStopping(patience=early_stopping_patience, save_path=os.path.join(save_path, f'{run_name}_best_model.pth'))

    # Start training
    for epoch in range(N_epochs):
        model.train()
        train_loss = 0.0

        total_batches = len(train_loader)
        with tqdm(total=total_batches, desc=f'Epoch {epoch + 1}/{N_epochs}', unit='batch') as pbar:
            for batch_idx, batch in enumerate(train_loader):
                inputs, targets = batch['input'], batch['label']
                inputs = inputs.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                targets = targets.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)

                optimizer.zero_grad(set_to_none=True)

                # Perform forward pass with AMP if enabled
                with torch.amp.autocast('cuda', enabled=amp):
                    predictions = model(inputs)
                    loss = criterion(predictions, targets)

                grad_scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                grad_scaler.step(optimizer)
                grad_scaler.update()

                train_loss += loss.item()
                global_step += 1

                # Log loss every 10 batches
                if batch_idx % 10 == 0:
                    pbar.set_postfix({'loss': loss.item()})

                pbar.update(1)

                # Log metrics to wandb every 25 global steps
                if global_step % 25 == 0:
                    experiment.log({'train_loss': loss.item(), 'global_step': global_step})

        # Validation loop
        val_loss, histograms, visualizations = validate_model(model, val_loader, criterion, device, amp, evaluater)

        # Calculate performance metrics
        precision, recall, f1, far, csi, hss, gss, mse, mae = evaluater.print_stat_readable()

        # Log performance metrics and other data
        log_data = {
            'epoch': epoch + 1,
            'learning rate': optimizer.param_groups[0]['lr'],
            'train_loss': train_loss / len(train_loader),
            'val_loss': val_loss / len(val_loader),
            'mse': mse.mean(),
            'mae': mae.mean(),
            'visualization': visualizations,  # Add the visualized image
            **histograms  # Record weight and gradient histograms
        }

        # Add performance metrics for different thresholds
        thresholds = [0.5, 2, 5, 10, 30]
        for i, threshold in enumerate(thresholds):
            log_data.update({
                f'precision({threshold})': precision[:, i].mean(),
                f'recall({threshold})': recall[:, i].mean(),
                f'f1_score({threshold})': f1[:, i].mean(),
                f'far({threshold})': far[:, i].mean(),
                f'csi({threshold})': csi[:, i].mean(),
                f'hss({threshold})': hss[:, i].mean(),
                f'gss({threshold})': gss[:, i].mean()
            })

        # Log all data at once
        experiment.log(log_data)

        # Step the scheduler
        scheduler.step(val_loss)

        logging.info(f'Epoch {epoch + 1}: train_loss={train_loss / len(train_loader):.4f}, val_loss={val_loss / len(val_loader):.4f}')

        # Check early stopping condition
        if early_stopping.should_stop(model, val_loss):
            logging.info(f"Early stopping at epoch {epoch + 1}")
            break

        evaluater.clear_all()

    # Load the best model when early stopping is triggered
    model = early_stopping.load(model)

    # Save the final model
    torch.save(model, os.path.join(save_path, f'{run_name}_final_model.pth'))

    # Finish wandb experiment
    experiment.finish()


def validate_model(model, val_loader, criterion, device, amp_enabled, evaluater, n_classes=1, n_channels=6):
    """
    Validates the model on the given validation dataset and computes evaluation metrics.

    Args:
        model (torch.nn.Module): The model to be validated.
        val_loader (DataLoader): DataLoader for the validation dataset.
        criterion (function): Loss function used for validation.
        device (torch.device): Device to run the validation (CPU or CUDA).
        amp_enabled (bool): Whether to enable Automatic Mixed Precision (AMP).
        evaluater (Evaluater): Instance of the Evaluater class for performance metrics calculation.
        n_classes (int, optional): Number of output classes (default is 1).
        n_channels (int, optional): Number of input channels (default is 6).

    Returns:
        tuple: (val_loss, histograms, visualizations)
            - val_loss (float): Total validation loss.
            - histograms (dict): Histograms of model weights and gradients.
            - visualizations (wandb.Image or None): Visualization of input, target, and prediction for a specific batch.
    """
    model.eval()  # Set the model to evaluation mode
    val_loss = 0.0
    histograms = {}
    idx = 0  # Initialize index for image visualization
    visualizations = None  # Placeholder for the visualization to be logged later

    with torch.no_grad():  # Disable gradient computation for validation
        for batch in val_loader:
            inputs, targets = batch['input'], batch['label']
            inputs = inputs.to(device=device, dtype=torch.float32)
            targets = targets.to(device=device, dtype=torch.float32)

            # Perform forward pass with AMP if enabled
            with torch.amp.autocast('cuda', enabled=amp_enabled):
                predictions = model(inputs)
                loss = criterion(predictions, targets)

            val_loss += loss.item()  # Accumulate the loss

            # Update performance metrics
            evaluater.update(gt=targets.cpu().numpy(), pred=predictions.cpu().numpy())

            # Visualize a specific batch (e.g., idx == 15)
            if idx == 15:
                visualizations = visualize_batch(inputs, targets, predictions, n_classes, n_channels)

            idx += 1  # Increment index

        # Record model weights and gradients as histograms
        for tag, value in model.named_parameters():
            tag = tag.replace('/', '.')
            if not (torch.isinf(value) | torch.isnan(value)).any():
                histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
            if value.grad is not None and not (torch.isinf(value.grad) | torch.isnan(value.grad)).any():
                histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

    return val_loss, histograms, visualizations


def visualize_batch(inputs, targets, predictions, n_classes, n_channels, batch_idx=0):
    """
    Function to visualize input, target, and prediction data, and return an image for logging in wandb.
    
    Args:
        inputs (torch.Tensor): Model input data.
        targets (torch.Tensor): Ground truth data.
        predictions (torch.Tensor): Model prediction data.
        n_classes (int): Number of classes (output channels).
        n_channels (int): Number of input channels.
        batch_idx (int): Index within the batch to visualize (default: 0).
    
    Returns:
        wandb.Image: Returns the visualized image for wandb logging.
    """
    # Set up the plot (configured according to n_channels)
    fig, axes = plt.subplots(nrows=3, ncols=n_channels, figsize=(24, 12))
    titles = ["Input", "Target", "Prediction"]

    # Set titles
    for i in range(len(titles)):
        axes[i, 0].text(-0.1, 0.5, titles[i], fontsize=20, ha='right', va='center', transform=axes[i, 0].transAxes)

    # Denormalize the data and convert dBZ to rainfall rate
    inputs_rainfall = dBZ_to_rfrate(pixel_to_dBZ(inputs[batch_idx].cpu().numpy()))
    targets_rainfall = dBZ_to_rfrate(pixel_to_dBZ(targets[batch_idx].cpu().numpy()))
    predictions_rainfall = dBZ_to_rfrate(pixel_to_dBZ(predictions[batch_idx].cpu().numpy()))

    # Compute the maximum value among all data to ensure consistent color scale
    max_value = max(
        inputs_rainfall.max(),
        targets_rainfall.max(),
        predictions_rainfall.max()
    )

    # Visualize input images (based on n_channels)
    for i in range(n_channels):
        im = axes[0, i].imshow(inputs_rainfall[i, :, :], cmap='jet', vmin=0, vmax=max_value)
        axes[0, i].axis('off')  # Disable axis

    # Visualize target and prediction images (based on n_classes)
    for i in range(n_classes):
        im = axes[1, i].imshow(targets_rainfall[i, :, :], cmap='jet', vmin=0, vmax=max_value)  # Target image
        axes[1, i].axis('off')

        im = axes[2, i].imshow(predictions_rainfall[i, :, :], cmap='jet', vmin=0, vmax=max_value)  # Prediction image
        axes[2, i].axis('off')

    # Disable axes for the remaining channels beyond n_classes
    for i in range(n_classes, n_channels):
        axes[1, i].axis('off')
        axes[2, i].axis('off')

    # Automatically adjust spacing to prevent the colorbar from overlapping
    plt.subplots_adjust(right=0.85)

    # Add colorbar (placed on the right side)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])  # Add colorbar at the right edge of the figure
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Rainfall rate (mm/hr)', fontsize=16)

    # Save the plot to a buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    
    # Load the image from the buffer using PIL
    visualization = Image.open(buf)

    # Create a wandb image for logging
    wandb_image = wandb.Image(visualization)

    # Close the buffer and clear the plot
    buf.close()
    plt.close(fig)

    return wandb_image