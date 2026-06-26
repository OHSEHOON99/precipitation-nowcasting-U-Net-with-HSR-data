import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from evaluate import Evaluater
from model_registry import load_model_from_config
from TSDataset import TSDataset, ToTensor


THRESHOLDS = [0.5, 2, 5, 10, 30]


def select_device(device_arg, device_num):
    if device_arg != "auto":
        return torch.device(device_arg)
    return torch.device(f"cuda:{device_num}" if torch.cuda.is_available() else "cpu")


def load_model_configs(path):
    with open(path, "r", encoding="utf-8") as file_obj:
        payload = json.load(file_obj)
    return payload["models"] if isinstance(payload, dict) else payload


def evaluate_model(model, test_data_path, device, batch_size, amp_enabled, seq_len):
    test_set = TSDataset(Path(test_data_path), transform=ToTensor())
    loader_args = {
        "batch_size": batch_size,
        "num_workers": 0,
        "pin_memory": device.type == "cuda",
    }
    test_loader = DataLoader(test_set, shuffle=False, drop_last=False, **loader_args)
    evaluater = Evaluater(seq_len=seq_len)

    model.eval()
    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"Testing {model.__class__.__name__}", unit="batch"):
            inputs = batch["input"].to(device=device, dtype=torch.float32)
            targets = batch["label"].to(device=device, dtype=torch.float32)
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                predictions = model(inputs)
            evaluater.update(gt=targets.cpu().numpy(), pred=predictions.cpu().numpy())

    return evaluater.print_stat_readable()


def metrics_to_rows(model_name, metrics):
    precision, recall, f1, far, csi, hss, gss, mse, mae = metrics
    rows = []
    threshold_metrics = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "far": far,
        "csi": csi,
        "hss": hss,
        "gss": gss,
    }
    for metric_name, values in threshold_metrics.items():
        for idx, threshold in enumerate(THRESHOLDS):
            rows.append({
                "model": model_name,
                "metric": metric_name,
                "threshold": threshold,
                "value": float(np.nanmean(values[:, idx])),
            })
    rows.extend([
        {"model": model_name, "metric": "mse", "threshold": "all", "value": float(np.nanmean(mse))},
        {"model": model_name, "metric": "mae", "threshold": "all", "value": float(np.nanmean(mae))},
    ])
    return rows


def write_results(rows, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results = pd.DataFrame(rows)
    summary = results.pivot_table(index="model", columns=["metric", "threshold"], values="value", aggfunc="first")
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        results.to_excel(writer, sheet_name="metrics", index=False)
        summary.to_excel(writer, sheet_name="summary")
    csv_path = output_path.with_suffix(".csv")
    results.to_csv(csv_path, index=False)
    logging.info("Saved results to %s and %s", output_path, csv_path)


def build_parser():
    parser = argparse.ArgumentParser(description="Evaluate one or more precipitation nowcasting models.")
    parser.add_argument("--data-path", required=True, help="Path to the test split directory")
    parser.add_argument("--models-config", required=True, help="JSON file describing model checkpoints")
    parser.add_argument("--output", default="outputs/performance_results.xlsx", help="Output Excel file")
    parser.add_argument("--batch-size", type=int, default=4, help="Evaluation batch size")
    parser.add_argument("--device", default="auto", help="Device string, for example auto, cpu, cuda:0")
    parser.add_argument("--device-num", type=int, default=0, help="CUDA device number used when --device=auto")
    parser.add_argument("--amp", action="store_true", help="Use CUDA automatic mixed precision")
    parser.add_argument("--allow-full-model", action="store_true", help="Allow loading trusted full-model .pth files")
    return parser


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = build_parser().parse_args()
    device = select_device(args.device, args.device_num)
    amp_enabled = args.amp and device.type == "cuda"
    logging.info("Using device %s", device)

    rows = []
    for config in load_model_configs(args.models_config):
        model_name = config.get("name", config.get("architecture", "model"))
        model = load_model_from_config(config, device=device, allow_full_model=args.allow_full_model)
        seq_len = config.get("n_classes", getattr(model, "n_classes", 1))
        metrics = evaluate_model(model, args.data_path, device, args.batch_size, amp_enabled, seq_len)
        rows.extend(metrics_to_rows(model_name, metrics))

    write_results(rows, args.output)


if __name__ == "__main__":
    main()
