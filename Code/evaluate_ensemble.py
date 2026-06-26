import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from evaluate import Evaluater
from evaluate_models import load_model_configs, metrics_to_rows, select_device, write_results
from model_registry import load_model_from_config
from TSDataset import TSDataset, ToTensor
from utils import dBZ_to_rfrate, pixel_to_dBZ


def combine_outputs(model_outputs, weights, mode):
    if len(model_outputs) != len(weights):
        raise ValueError(f"Expected {len(model_outputs)} weights, got {len(weights)}")
    weight_sum = sum(weights)
    if weight_sum <= 0:
        raise ValueError("Weight sum must be positive")

    if mode == "arithmetic":
        weighted = [weight * output for weight, output in zip(weights, model_outputs)]
        return torch.sum(torch.stack(weighted, dim=0), dim=0) / weight_sum

    eps = 1e-10
    if mode == "geometric":
        weighted = [(output.clamp_min(eps)) ** weight for weight, output in zip(weights, model_outputs)]
        product = torch.prod(torch.stack(weighted, dim=0), dim=0)
        return product ** (1 / weight_sum)

    if mode == "harmonic":
        reciprocal_sum = torch.zeros_like(model_outputs[0])
        for weight, output in zip(weights, model_outputs):
            reciprocal_sum += weight / output.clamp_min(eps)
        return weight_sum / reciprocal_sum

    raise ValueError(f"Unsupported ensemble mode: {mode}")


def prediction_to_uint16_png(prediction):
    array = prediction.detach().cpu().numpy()
    if array.ndim == 3:
        array = array[0]
    dbz = pixel_to_dBZ(array)
    stored = np.clip(dbz * 100 + 1000, 0, 65535).astype(np.uint16)
    return Image.fromarray(stored)


def visualize_sample(inputs, targets, model_outputs, ensemble_prediction, output_path):
    inputs_rainfall = dBZ_to_rfrate(pixel_to_dBZ(inputs.detach().cpu().numpy()))
    targets_rainfall = dBZ_to_rfrate(pixel_to_dBZ(targets.detach().cpu().numpy()))
    model_rainfall = [dBZ_to_rfrate(pixel_to_dBZ(output.detach().cpu().numpy())) for output in model_outputs]
    ensemble_rainfall = dBZ_to_rfrate(pixel_to_dBZ(ensemble_prediction.detach().cpu().numpy()))
    max_value = max(
        inputs_rainfall.max(),
        targets_rainfall.max(),
        ensemble_rainfall.max(),
        max(output.max() for output in model_rainfall),
    )

    cols = max(inputs_rainfall.shape[0], len(model_outputs), 1)
    fig, axes = plt.subplots(nrows=4, ncols=cols, figsize=(cols * 3, 12))
    row_names = ["Input", "Target", "Models", "Ensemble"]
    for row_idx, name in enumerate(row_names):
        axes[row_idx, 0].set_ylabel(name)

    for idx in range(cols):
        if idx < inputs_rainfall.shape[0]:
            axes[0, idx].imshow(inputs_rainfall[idx], cmap="jet", vmin=0, vmax=max_value)
        axes[0, idx].axis("off")

        if idx < targets_rainfall.shape[0]:
            axes[1, idx].imshow(targets_rainfall[idx], cmap="jet", vmin=0, vmax=max_value)
        axes[1, idx].axis("off")

        if idx < len(model_rainfall):
            axes[2, idx].imshow(model_rainfall[idx][0], cmap="jet", vmin=0, vmax=max_value)
        axes[2, idx].axis("off")

        if idx == 0:
            axes[3, idx].imshow(ensemble_rainfall[0], cmap="jet", vmin=0, vmax=max_value)
        axes[3, idx].axis("off")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def evaluate_ensemble(
        models,
        test_data_path,
        device,
        batch_size,
        amp_enabled,
        seq_len,
        weights,
        mode,
        output_dir,
        save_predictions=False,
        save_visualizations=0,
):
    test_set = TSDataset(Path(test_data_path), transform=ToTensor())
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    evaluater = Evaluater(seq_len=seq_len)
    for model in models:
        model.eval()

    ensemble_id = f"{mode}_weights_{'_'.join(str(weight) for weight in weights)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = Path(output_dir)
    prediction_dir = output_dir / "ensemble_predictions" / ensemble_id
    visualization_dir = output_dir / "visualizations" / ensemble_id
    saved_visualizations = 0
    sample_index = 0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"Testing {ensemble_id}", unit="batch"):
            inputs = batch["input"].to(device=device, dtype=torch.float32)
            targets = batch["label"].to(device=device, dtype=torch.float32)
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                model_outputs = [model(inputs) for model in models]
                ensemble_predictions = combine_outputs(model_outputs, weights, mode)

            evaluater.update(gt=targets.cpu().numpy(), pred=ensemble_predictions.cpu().numpy())

            for batch_idx in range(ensemble_predictions.size(0)):
                if save_predictions:
                    prediction_path = prediction_dir / f"sample_{sample_index:06d}.png"
                    prediction_path.parent.mkdir(parents=True, exist_ok=True)
                    prediction_to_uint16_png(ensemble_predictions[batch_idx]).save(prediction_path)

                if saved_visualizations < save_visualizations:
                    visualize_sample(
                        inputs[batch_idx],
                        targets[batch_idx],
                        [output[batch_idx] for output in model_outputs],
                        ensemble_predictions[batch_idx],
                        visualization_dir / f"sample_{sample_index:06d}_visualization.png",
                    )
                    saved_visualizations += 1
                sample_index += 1

    return ensemble_id, evaluater.print_stat_readable()


def load_grid_config(path):
    with open(path, "r", encoding="utf-8") as file_obj:
        payload = json.load(file_obj)
    return payload.get("modes", ["arithmetic"]), payload["weights"]


def build_parser():
    parser = argparse.ArgumentParser(description="Evaluate a weighted ensemble of precipitation nowcasting models.")
    parser.add_argument("--data-path", required=True, help="Path to the test split directory")
    parser.add_argument("--models-config", required=True, help="JSON file describing model checkpoints")
    parser.add_argument("--output", default="outputs/ensemble_results.xlsx", help="Output Excel file")
    parser.add_argument("--output-dir", default="outputs/ensemble", help="Directory for optional predictions and visualizations")
    parser.add_argument("--batch-size", type=int, default=4, help="Evaluation batch size")
    parser.add_argument("--device", default="auto", help="Device string, for example auto, cpu, cuda:0")
    parser.add_argument("--device-num", type=int, default=0, help="CUDA device number used when --device=auto")
    parser.add_argument("--amp", action="store_true", help="Use CUDA automatic mixed precision")
    parser.add_argument("--allow-full-model", action="store_true", help="Allow loading trusted full-model .pth files")
    parser.add_argument("--mode", choices=["arithmetic", "geometric", "harmonic"], default="arithmetic", help="Ensemble mode")
    parser.add_argument("--weights", nargs="+", type=float, help="Weights for the models in --models-config order")
    parser.add_argument("--grid-config", help="JSON file containing modes and multiple weight combinations")
    parser.add_argument("--save-predictions", action="store_true", help="Save ensemble prediction PNGs under --output-dir")
    parser.add_argument("--save-visualizations", type=int, default=0, help="Number of sample visualizations to save")
    return parser


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = build_parser().parse_args()
    device = select_device(args.device, args.device_num)
    amp_enabled = args.amp and device.type == "cuda"
    logging.info("Using device %s", device)

    configs = load_model_configs(args.models_config)
    models = [load_model_from_config(config, device=device, allow_full_model=args.allow_full_model) for config in configs]
    seq_len = configs[0].get("n_classes", getattr(models[0], "n_classes", 1))

    if args.grid_config:
        modes, weight_sets = load_grid_config(args.grid_config)
    else:
        if args.weights is None:
            raise ValueError("Provide --weights or --grid-config")
        modes, weight_sets = [args.mode], [args.weights]

    rows = []
    for mode in modes:
        for weights in weight_sets:
            ensemble_id, metrics = evaluate_ensemble(
                models=models,
                test_data_path=args.data_path,
                device=device,
                batch_size=args.batch_size,
                amp_enabled=amp_enabled,
                seq_len=seq_len,
                weights=weights,
                mode=mode,
                output_dir=args.output_dir,
                save_predictions=args.save_predictions,
                save_visualizations=args.save_visualizations,
            )
            rows.extend(metrics_to_rows(ensemble_id, metrics))

    write_results(rows, args.output)


if __name__ == "__main__":
    main()
