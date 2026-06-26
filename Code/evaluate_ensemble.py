import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
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

    def channel_count(array):
        return 1 if array.ndim == 2 else array.shape[0]

    def channel(array, idx=0):
        return array if array.ndim == 2 else array[idx]

    rainfall_arrays = [inputs_rainfall, targets_rainfall, ensemble_rainfall, *model_rainfall]
    max_value = max(float(np.nanmax(array)) for array in rainfall_arrays if array.size)
    max_value = max(max_value, 1.0)

    norm = Normalize(vmin=0, vmax=max_value)
    cmap = "jet"

    panels = []
    input_count = channel_count(inputs_rainfall)
    for idx in range(input_count):
        panels.append((f"Input {idx + 1}", channel(inputs_rainfall, idx)))

    target_count = channel_count(targets_rainfall)
    for idx in range(target_count):
        title = "Target" if target_count == 1 else f"Target {idx + 1}"
        panels.append((title, channel(targets_rainfall, idx)))

    for idx, output in enumerate(model_rainfall):
        panels.append((f"Model {idx + 1} Prediction", channel(output)))

    ensemble_count = channel_count(ensemble_rainfall)
    for idx in range(ensemble_count):
        title = "Ensemble Prediction" if ensemble_count == 1 else f"Ensemble {idx + 1}"
        panels.append((title, channel(ensemble_rainfall, idx)))

    cols = min(max(input_count, len(model_rainfall), 1), len(panels), 6)
    rows = int(np.ceil(len(panels) / cols))
    fig_width = max(cols * 2.4 + 1.2, 7.0)
    fig_height = max(rows * 2.6, 3.0)
    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(fig_width, fig_height), squeeze=False)
    fig.subplots_adjust(left=0.04, right=0.88, bottom=0.06, top=0.9, wspace=0.08, hspace=0.28)

    for ax, (title, array) in zip(axes.ravel(), panels):
        ax.imshow(array, cmap=cmap, norm=norm)
        ax.set_title(title, fontsize=9)
        ax.axis("off")

    for ax in axes.ravel()[len(panels):]:
        ax.axis("off")

    mappable = ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array([])
    cbar_ax = fig.add_axes([0.91, 0.13, 0.018, 0.72])
    cbar = fig.colorbar(mappable, cax=cbar_ax)
    cbar.set_label("Rainfall rate (mm/hr)", fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, facecolor="white")
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
