import torch

from models import UNet, AttUNet, DualAttUNet, SEUNet, SAR_UNet


MODEL_REGISTRY = {
    "unet": UNet,
    "seunet": SEUNet,
    "attunet": AttUNet,
    "dualattunet": DualAttUNet,
    "sarunet": SAR_UNet,
}


def build_model(architecture, n_channels=6, n_classes=1, bilinear=False):
    try:
        model_cls = MODEL_REGISTRY[architecture.lower()]
    except KeyError as exc:
        choices = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(f"Unknown model architecture '{architecture}'. Choose one of: {choices}") from exc
    return model_cls(n_channels=n_channels, n_classes=n_classes, bilinear=bilinear)


def load_checkpoint(path, device, allow_full_model=False):
    load_kwargs = {"map_location": device}
    try:
        load_kwargs["weights_only"] = not allow_full_model
        return torch.load(path, **load_kwargs)
    except TypeError:
        load_kwargs.pop("weights_only", None)
        return torch.load(path, **load_kwargs)


def load_model_from_config(config, device, allow_full_model=False):
    checkpoint = load_checkpoint(config["path"], device=device, allow_full_model=allow_full_model)
    if isinstance(checkpoint, torch.nn.Module):
        if not allow_full_model:
            raise ValueError(
                f"{config['path']} contains a serialized model object. "
                "Pass --allow-full-model only for checkpoints you trust."
            )
        return checkpoint.to(device)

    state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model = build_model(
        config.get("architecture", config.get("model", "seunet")),
        n_channels=config.get("n_channels", 6),
        n_classes=config.get("n_classes", 1),
        bilinear=config.get("bilinear", False),
    )
    model.load_state_dict(state_dict)
    return model.to(device)
