import argparse
import logging
import os
import torch
import wandb

from model import UNet, ConvLSTM
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from evaluate import *
from loss import *
from TSDataset import *
from utils import *


dir_checkpoint = '/home/u2018144071/사기설/checkpoint'
torch.cuda.empty_cache()

# Example usage
# python train.py --data_path /home/u2018144071/사기설/data/dataset_10.0_0.05_6_3_0 --learning-rate 0.001 --epochs 100 --n_channels 6 --output_labels 3 --save_path /home/u2018144071/사기설/dataset_10.0_0.05_6_3_0.pt


def train_model(
        data_path,
        save_path,
        model,
        device,
        epochs: int = 800,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        save_checkpoint: bool = True,
        amp: bool = False,
        weight_decay: float = 1e-8,
        # momentum: float = 0.999, (for SGD)
        gradient_clipping: float = 1.0      # inital_value : 1.0
):
    # 1. Create dataset
    train_set = TSDataset(os.path.join(data_path, 'train'), transform=ToTensor())
    val_set = TSDataset(os.path.join(data_path, 'valid'), transform=ToTensor())

    n_train = len(train_set)
    n_val = len(val_set)

    n_channels = model.n_channels
    n_classes = model.n_classes

    # 2. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True)

    # tensor : (C, H, W)
    # batch : (batch_size, C, H, W)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    # Wandb : Initialize logging
    experiment = wandb.init(project='U-Net', resume='allow', anonymous='must')

    # wandb.config(): Update configuration settings
    experiment.config.update(
        dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
             save_checkpoint=save_checkpoint, amp=amp))

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Mixed Precision: {amp}
    ''')
    
    # Model Source (Pytorch-UNet : https://github.com/milesial/Pytorch-UNet.git)
    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.Adam(
        model.parameters(), 
        lr=learning_rate, 
        betas=(0.5, 0.999), 
        weight_decay=weight_decay)
    
    # scheduler : The scheduler adjusts the learning rate when the training is not progressing.
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    # criterion = nn.L1Loss()
    criterion = BalancedMSEMAE()
    global_step = 0

    # The evaluator is defined for the purpose of metric evaluation.
    evaluater = Evaluater(seq_len=n_classes)

    # 5. Begin training
    for epoch in range(1, epochs + 1):
        # Initialize the evaluator for each epoch.
        evaluater.clear_all()

        model.train()
        epoch_loss = 0

        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                # images : (batch_size, C, H, W)
                images, labels = batch['input'], batch['label']

                assert images.shape[1] == model.n_channels, \
                    f'Network has been defined with {model.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                labels = labels.to(device=device, dtype=torch.float32)  # The label should also be in float format.

                optimizer.zero_grad(set_to_none=True)

                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    # Passed to the model.
                    labels_pred = model(images)
                    loss = criterion(labels_pred, labels)

                grad_scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                grad_scaler.step(optimizer)
                grad_scaler.update()

                labels_pred_np = labels_pred.detach().cpu().numpy()
                labels_np = labels.cpu().numpy()

                # input : (batch_size, seq_len, H, W)
                # output : (seq_len, batch_size, len(thresholds))
                evaluater.update(labels_pred_np, labels_np)
                # print(evaluater.calculate_stat())

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()

                # Logs data in dictionary format to Weights & Biases (wandb).
                experiment.log({
                    'train loss': loss.item(),
                    'step': global_step,
                    'epoch': epoch
                })
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                # Evaluation round: Perform evaluations five times per epoch.
                division_step = (n_train // (5 * batch_size))   # n_train : length of traindataset.
                if division_step > 0:
                    if global_step % division_step == 0:    # global_step : The total number of steps completed up to the current point.
                        histograms = {}
                        for tag, value in model.named_parameters():
                            tag = tag.replace('/', '.')
                            if not (torch.isinf(value) | torch.isnan(value)).any():
                                histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                            if not (torch.isinf(value.grad) | torch.isnan(value.grad)).any():
                                histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())
                        
                        val_loss = evaluate(model, val_loader, device, criterion, amp)
                        scheduler.step(val_loss)

                        logging.info('Validation Loss: {}'.format(val_loss))

                        try:
                            inputs = {}
                            outputs = {}

                            # Visualizing timeseries
                            # images: (batch_size, C, H, W)
                            # labels: (batch_size, 1, H, W)
                            # labels_pred: (batch_size, 1, H, W)
                            for i in range(n_channels):
                                time = 5 * (n_channels-i-1)
                                inputs[f'before {time} minutes'] = wandb.Image(images[0,i,...].cpu())

                            for i in range(n_classes):
                                time = 5 * (i+1)
                                outputs[f'after {time} minutes'] = wandb.Image(labels[0,i,...].cpu())
                                outputs[f'after {time} minutes(pred)'] = wandb.Image(labels_pred[0,i,...].float().cpu())

                            experiment.log({
                                'learning rate': optimizer.param_groups[0]['lr'],
                                'validation Loss': val_loss,
                                'inputs': inputs,
                                'labels': outputs,
                                'step': global_step,
                                'epoch': epoch,
                                **histograms
                            })
                        except:
                            pass
        
        # Record epoch loss.
        epoch_loss = epoch_loss / len(train_loader)
        experiment.log({
                'epoch loss': epoch_loss,
                'step': global_step,
                'epoch': epoch
            })

        if save_checkpoint:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            state_dict = model.state_dict()
            torch.save(state_dict, f'{dir_checkpoint}/checkpoint_epoch{epoch}.pth')
            logging.info(f'Checkpoint {epoch} saved!')

        precision, recall, f1, far, csi, hss, gss, mse, mae = evaluater.print_stat_readable()

        thresholds = [0.5, 2, 5, 10, 30]

        for i, threshold in enumerate(thresholds):
            experiment.log({
                f'precision({threshold})': precision[:, i].mean(),
                f'recall({threshold})': recall[:, i].mean(),
                f'f1_score({threshold})': f1[:, i].mean(),
                f'far({threshold})': far[:, i].mean(),
                f'csi({threshold})': csi[:, i].mean(),
                f'hss({threshold})': hss[:, i].mean(),
                f'gss({threshold})': gss[:, i].mean(),
                f'mse({threshold})': mse.mean(),
                f'mae({threshold})': mae.mean(),
                f'step': global_step,
                f'epoch': epoch
            })

    # model save
    torch.save(model, save_path)

def get_args():
    parser = argparse.ArgumentParser(description='Deep Learning-Based Short-Term Rainfall Prediction Model')

    parser.add_argument('--data_path', type=str, help='Path to the dataset')
    parser.add_argument('--save_path', type=str, help='Save Path')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=1000, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=4, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-3,
                        help='Learning rate', dest='lr')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--n_channels', '-ch', type=int, default=6, help='Number of input channels')
    parser.add_argument('--output_labels', '-c', type=int, default=1, help='Number of output_labels')

    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_output_labels is the number of probabilities you want to get per pixel
    model = UNet(n_channels=args.n_channels, n_classes=args.output_labels, bilinear=args.bilinear)
    model = model.to(memory_format=torch.channels_last)

    logging.info(f'Network:\n'
                 f'\t{model.n_channels} input channels\n'
                 f'\t{model.n_classes} output channels (output_labels)\n'
                 f'\t{"Bilinear" if model.bilinear else "Transposed conv"} upscaling')

    model.to(device=device)

    train_model(
        data_path=args.data_path,
        save_path=args.save_path,
        model=model,
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        amp=args.amp
    )