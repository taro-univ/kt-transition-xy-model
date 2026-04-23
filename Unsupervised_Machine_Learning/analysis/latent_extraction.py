# analysis/latent_extraction.py

import argparse
from pathlib import Path
from typing import List

import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader

import sys
from pathlib import Path

# Add project root (two levels up from this file) to sys.path.
# This enables absolute imports such as `from dataset...` when running as a script.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset.unsupervised_xy_dataset import create_xy_dataset_from_config
from models.autoencoder import build_autoencoder_from_config
from models.vae import build_vae_from_config
from models.contrastive_encoder import SimCLREncoder, MultiScaleSimCLREncoder


"""
latent_extraction.py
--------------------------------------------------

Extract latent representations from trained models and save them as a compressed .npz.

Supported model types
---------------------
- autoencoder:
    Uses the latent z returned by model(x).
- vae:
    Uses mu (mean) as the representation (NOT a sampled z) to reduce randomness.
- contrastive:
    Uses encoder output h = encode(x) (pre-projector) as the representation.

Reproducibility notes
---------------------
- This script itself is deterministic: it does not use RNG for latent extraction.
- Reproducibility depends on:
    * using the exact same checkpoint
    * dataset order (shuffle=False in DataLoader)
    * deterministic backend settings (if you require bitwise identical outputs)
- For contrastive models, augmentation is disabled by using dataset mode="ae"
  (i.e., a single-view dataset without random augmentations).
"""

# ---------------------------------------------------------
# Common helpers
# ---------------------------------------------------------

def load_config_and_paths(config_arg: str | None, default_name: str):
    """
    Load YAML config and normalize relative paths to project_root.

    - logging.{checkpoint_dir,latent_dir,plots_dir} are converted to absolute paths
      relative to project_root if needed.
    - data.npz_path is also converted to absolute path relative to project_root if needed.
    """
    script_path = Path(__file__).resolve()
    project_root = script_path.parents[1]

    if config_arg is None:
        config_path = project_root / "configs" / default_name
    else:
        config_path = Path(config_arg)

    print(f"[INFO] Loading config from: {config_path}")
    with open(config_path, "r", encoding="utf-8-sig") as f:
        cfg = yaml.safe_load(f)

    # Normalize logging paths relative to project_root
    logging_cfg = cfg.get("logging", {})
    for key in ["checkpoint_dir", "latent_dir", "plots_dir"]:
        if key in logging_cfg:
            p = Path(logging_cfg[key])
            if not p.is_absolute():
                logging_cfg[key] = str(project_root / p)
    cfg["logging"] = logging_cfg

    # Normalize npz path relative to project_root
    data_cfg = cfg["data"]
    npz_path = Path(data_cfg["npz_path"])
    if not npz_path.is_absolute():
        data_cfg["npz_path"] = str(project_root / npz_path)
    cfg["data"] = data_cfg

    return cfg, project_root


def _build_contrastive_encoder(cfg, data_cfg, device):
    """
    Build a SimCLREncoder or MultiScaleSimCLREncoder from config.

    Used for both 'contrastive' and 'helicity_contrastive' model types.
    The helicity head is not constructed here (not needed for latent extraction).
    """
    mcfg = cfg["model"]
    in_channels = int(data_cfg.get("in_channels", 1))
    lattice_size = int(data_cfg.get("lattice_size", 32))
    encoder_channels = list(mcfg["encoder_channels"])
    proj_hidden = int(mcfg["projector"]["hidden_dim"])
    proj_out = int(mcfg["projector"]["out_dim"])
    use_bn = bool(mcfg.get("use_batchnorm", True))
    act_name = str(mcfg.get("activation", "relu"))
    encoder_type = str(mcfg.get("encoder_type", "single")).lower()

    if encoder_type == "multi_scale":
        kernel_sizes = mcfg.get("kernel_sizes", [3, 7, 15])
        return MultiScaleSimCLREncoder(
            in_channels=in_channels,
            lattice_size=lattice_size,
            encoder_channels=encoder_channels,
            proj_hidden=proj_hidden,
            proj_out=proj_out,
            use_batchnorm=use_bn,
            activation_name=act_name,
            kernel_sizes=kernel_sizes,
        ).to(device)
    else:
        return SimCLREncoder(
            in_channels=in_channels,
            lattice_size=lattice_size,
            encoder_channels=encoder_channels,
            proj_hidden=proj_hidden,
            proj_out=proj_out,
            use_batchnorm=use_bn,
            activation_name=act_name,
        ).to(device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_type",
        type=str,
        default="autoencoder",
        choices=["autoencoder", "vae", "contrastive", "helicity_contrastive"],
        help="Which model type to extract latent representations from.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config. If omitted, chosen automatically from model_type.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint. If omitted, uses best_xxx.pt under checkpoint_dir.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output .npz path. If omitted, uses results/latent/{model_type}/{model_type}_latent.npz",
    )
    args = parser.parse_args()

    if args.model_type == "autoencoder":
        default_cfg_name = "autoencoder.yaml"
        default_ckpt_name = "best_autoencoder.pt"
    elif args.model_type == "vae":
        default_cfg_name = "vae.yaml"
        default_ckpt_name = "best_vae.pt"
    elif args.model_type == "contrastive":
        default_cfg_name = "contrastive.yaml"
        default_ckpt_name = "best_contrastive.pt"
    else:  # helicity_contrastive
        default_cfg_name = "helicity_contrastive.yaml"
        default_ckpt_name = "best_helicity_contrastive.pt"

    cfg, project_root = load_config_and_paths(args.config, default_cfg_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # Dataset:
    # Even for contrastive models, latent extraction should be done WITHOUT augmentation.
    # Therefore we use mode="ae" (single-view) here.
    data_cfg = cfg["data"]
    full_dataset = create_xy_dataset_from_config(data_cfg, mode="ae")
    loader = DataLoader(
        full_dataset,
        batch_size=int(data_cfg.get("batch_size", 128)),
        shuffle=False,
        num_workers=int(data_cfg.get("num_workers", 4)),
        pin_memory=True,
    )

    # Build model
    if args.model_type == "autoencoder":
        model = build_autoencoder_from_config(cfg["model"], cfg["data"]).to(device)
    elif args.model_type == "vae":
        model = build_vae_from_config(cfg["model"], cfg["data"]).to(device)
    else:  # contrastive or helicity_contrastive
        # Both share the same encoder architecture; the helicity head is not needed for extraction.
        model = _build_contrastive_encoder(cfg, data_cfg, device)

    # Load checkpoint
    logging_cfg = cfg.get("logging", {})
    ckpt_dir = Path(
        logging_cfg.get(
            "checkpoint_dir", f"results/checkpoint/{args.model_type}"
        )
    )
    if args.checkpoint is None:
        ckpt_path = ckpt_dir / default_ckpt_name
    else:
        ckpt_path = Path(args.checkpoint)

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print(f"[INFO] Loading checkpoint from {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Latent extraction
    zs = []
    Ts = []

    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, (list, tuple)):
                x, T = batch
            else:
                x = batch
                T = None

            x = x.to(device, non_blocking=True)

            if args.model_type == "autoencoder":
                _x_rec, z = model(x)
            elif args.model_type == "vae":
                _x_rec, mu, logvar, z = model(x)
                # Use mu as the representation (deterministic w.r.t. model weights and input)
                z = mu
            else:  # contrastive or helicity_contrastive
                # encode() returns the pre-projector representation h
                z = model.encode(x)

            zs.append(z.detach().cpu().numpy())

            if T is not None:
                Ts.append(T.detach().cpu().numpy())

    z_all = np.concatenate(zs, axis=0)
    if Ts:
        T_all = np.concatenate(Ts, axis=0)
    else:
        T_all = None

    print(f"[INFO] Collected latent shape = {z_all.shape}")

    # Output path
    if args.out is None:
        latent_dir = Path(logging_cfg.get("latent_dir", f"results/latent/{args.model_type}"))
        latent_dir.mkdir(parents=True, exist_ok=True)
        out_path = latent_dir / f"{args.model_type}_latent.npz"
    else:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)

    if T_all is not None:
        np.savez_compressed(out_path, z=z_all, T=T_all)
    else:
        np.savez_compressed(out_path, z=z_all)

    print(f"[INFO] Saved latent to {out_path}")


if __name__ == "__main__":
    main()


"""
Usage
-----
# Autoencoder latent
python Unsupervised_Machine_Learning/analysis/latent_extraction.py --model_type autoencoder

# VAE latent (uses mu)
python Unsupervised_Machine_Learning/analysis/latent_extraction.py --model_type vae

# Contrastive latent (uses encoder pre-projector h)
python Unsupervised_Machine_Learning/analysis/latent_extraction.py --model_type contrastive

# Helicity-aware contrastive latent (proposed method)
python Unsupervised_Machine_Learning/analysis/latent_extraction.py --model_type helicity_contrastive
"""
