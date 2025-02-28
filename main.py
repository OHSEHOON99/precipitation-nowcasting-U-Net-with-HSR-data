import argparse
import logging
import torch

from models import UNet, AttUNet, DualAttUNet, SEUNet
from trainer import train_model

# Optional: Clean up matplotlib configuration if necessary
import matplotlib
matplotlib.use('Agg')  # Avoid using interactive backends when running headless (e.g., on servers)


# Variables and setup
dir_checkpoint = '/home/sehoon/Desktop/Sehoon/측량학회/checkpoint/'
torch.cuda.empty_cache()


# Argument parsing function
def get_args():
    parser = argparse.ArgumentParser(description='Deep Learning-Based Short-Term Rainfall Prediction Model')

    parser.add_argument('--data_path', type=str, required=True, help='Path to the dataset')
    parser.add_argument('--save_path', type=str, required=True, help='Path to save the trained model')
    parser.add_argument('--epochs', '-e', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', '-b', type=int, default=4, help='Batch size for training')
    parser.add_argument('--learning_rate', '-l', type=float, default=1e-4, help='Learning rate for optimizer')
    parser.add_argument('--amp', action='store_true', help='Use automatic mixed precision (AMP) for training')
    parser.add_argument('--bilinear', action='store_true', help='Use bilinear upsampling in UNet (default: False)')
    parser.add_argument('--n_inputs', '-ch', type=int, default=6, help='Number of input channels')
    parser.add_argument('--n_labels', '-c', type=int, default=1, help='Number of output channels')
    parser.add_argument('--device_num', type=int, default=0, help='CUDA device number (if using GPU)')

    return parser.parse_args()


if __name__ == '__main__':

    # Parse command-line arguments
    args = get_args()

    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    # Device selection
    device = torch.device(f'cuda:{args.device_num}' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Define the model
    model = SEUNet(n_channels=args.n_inputs, n_classes=args.n_labels, bilinear=args.bilinear)

    model_name = model.__class__.__name__

    logging.info(f'Network: {model_name}\n'
                 f'\t{args.n_inputs} input channels\n'
                 f'\t{args.n_labels} output channels\n'
                 f'\t{"Bilinear" if args.bilinear else "Transposed conv"} upscaling')

    # Move model to the specified device
    model.to(device=device)

    # Start the training process
    train_model(
        data_path=args.data_path,
        save_path=args.save_path,
        model=model,
        device=device,
        N_epochs=args.epochs,
        N_batch=args.batch_size,
        learning_rate=args.learning_rate,
        amp=args.amp
    )