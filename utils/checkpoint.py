"""Load checkpoints across the VARG interface naming cleanup."""

from collections import OrderedDict
from pathlib import Path

import torch


def canonicalize_state_dict(state_dict):
    """Rename module paths only; parameter tensors are not transformed.

    Supports checkpoints saved before this interface cleanup and optional DDP
    prefixes. Multiple keys mapping to one destination are rejected explicitly.
    """
    renamed = OrderedDict()
    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[len("module."):]
        replacements = (
            ("mix_net.", "conditioner."),
            ("conditioner.Feat_Encoder.", "conditioner.style_encoder."),
            ("conditioner.low_pro_mlp.", "conditioner.style_projector."),
            ("conditioner.var.", "conditioner.sat."),
            ("conditioner.h_cnn_encoder.", "conditioner.hcem.heu."),
            ("conditioner.cont_arg_m.", "conditioner.hcem.cam."),
        )
        for old, new in replacements:
            if key.startswith(old):
                key = new + key[len(old):]
        if key in renamed:
            raise ValueError(f"Duplicate checkpoint key after renaming: {key}")
        renamed[key] = value
    return renamed


def load_varg_checkpoint(model, checkpoint_path, strict=True):
    """Load a local tensor checkpoint using the current VARG module paths."""
    path = Path(checkpoint_path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    state_dict = payload.get("state_dict", payload)
    return model.load_state_dict(canonicalize_state_dict(state_dict), strict=strict)
