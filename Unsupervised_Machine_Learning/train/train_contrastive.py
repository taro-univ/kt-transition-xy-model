import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
from pathlib import Path
from typing import Dict, Any, List

import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from models.contrastive_encoder import SimCLREncoder, MultiScaleSimCLREncoder
from dataset.unsupervised_xy_dataset import create_xy_dataset_from_config


"""
train_contrastive.py
--------------------------------------------------

Baseline SimCLR-style contrastive training (NT-Xent).

Reproducibility notes (important)
---------------------------------
This script sets torch seeds, but exact reproducibility still depends on:
- DataLoader workers: if num_workers > 0, each worker has its own RNG state.
- random_split: uses torch RNG (so manual_seed affects it).
- CuDNN / CUDA determinism flags (training script does not enforce them here).
If you need bitwise determinism, set:
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False
and fix DataLoader generator/worker_init_fn in the outer training harness.
"""


# ============================================================
# NT-Xent loss (SimCLR)
# ============================================================

def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float) -> torch.Tensor:
    """
    SimCLR NT-Xent loss for 2B samples (positives: i <-> i+B).

    Notes
    -----
    - We concatenate [z1, z2] to form (2B, D).
    - Self-similarity is masked out by setting diagonal to -1e9.
    """
    batch_size = z1.size(0)
    device = z1.device

    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    z = torch.cat([z1, z2], dim=0)  # (2B, D)

    sim = torch.matmul(z, z.T) / temperature  # (2B, 2B)

    # Mask self similarity
    mask = torch.eye(2 * batch_size, device=device, dtype=torch.bool)
    sim = sim.masked_fill(mask, -1e9)

    # Positive for i is i+B (i<B) or i-B (i>=B)
    labels = torch.arange(2 * batch_size, device=device)
    labels = (labels + batch_size) % (2 * batch_size)

    loss = F.cross_entropy(sim, labels)
    return loss


# ============================================================
# Common utilities
# ============================================================

def setup_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def setup_optimizer(model: nn.Module, cfg_train: Dict[str, Any]):
    opt_name = cfg_train.get("optimizer", "adam").lower()
    lr = float(cfg_train.get("lr", 1e-3))
    weight_decay = float(cfg_train.get("weight_decay", 0.0))

    if opt_name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt_name == "sgd":
        return torch.optim.SGD(
            model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer: {opt_name}")


def setup_scheduler(optimizer, cfg_train: Dict[str, Any]):
    sch_cfg = cfg_train.get("scheduler", {})
    sch_type = sch_cfg.get("type", "none").lower()

    if sch_type == "none":
        return None
    elif sch_type == "cosine":
        T_max = int(sch_cfg.get("T_max", cfg_train.get("num_epochs", 200)))
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)
    elif sch_type == "step":
        step_size = int(sch_cfg.get("step_size", 50))
        gamma = float(sch_cfg.get("gamma", 0.5))
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=step_size, gamma=gamma
        )
    else:
        raise ValueError(f"Unknown scheduler: {sch_type}")


def ensure_dirs(logging_cfg: Dict[str, Any]):
    for key in ["checkpoint_dir", "latent_dir", "plots_dir"]:
        d = logging_cfg.get(key, None)
        if d is None:
            continue
        Path(d).mkdir(parents=True, exist_ok=True)


# ============================================================
# Training loop
# ============================================================

def train_contrastive(cfg: Dict[str, Any]) -> None:
    # Seed control (torch only)
    seed = int(cfg.get("seed", 42))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = setup_device()
    print(f"[INFO] Using device: {device}")

    # Dataset
    data_cfg = cfg["data"]
    augment_cfg = cfg.get("augment", {})

    full_dataset = create_xy_dataset_from_config(
        data_cfg, mode="contrastive", augment_config=augment_cfg
    )

    batch_size = int(data_cfg.get("batch_size", 256))
    num_workers = int(data_cfg.get("num_workers", 4))

    # Train/val split (random_split uses torch RNG)
    n_total = len(full_dataset)
    n_val = max(int(0.1 * n_total), 1)
    n_train = n_total - n_val
    train_dataset, val_dataset = random_split(full_dataset, [n_train, n_val])

    print(f"[INFO] Dataset size: total={n_total}, train={n_train}, val={n_val}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,  # simplify NT-Xent implementation
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # Model
    model_cfg = cfg["model"]
    in_channels = int(data_cfg.get("in_channels", 1))
    lattice_size = int(data_cfg.get("lattice_size", 32))
    encoder_channels = list(model_cfg["encoder_channels"])
    proj_hidden = int(model_cfg["projector"]["hidden_dim"])
    proj_out = int(model_cfg["projector"]["out_dim"])
    use_bn = bool(model_cfg.get("use_batchnorm", True))
    act_name = str(model_cfg.get("activation", "relu"))

    encoder_type = str(model_cfg.get("encoder_type", "single"))

    if encoder_type == "single":
        model = SimCLREncoder(
            in_channels=in_channels,
            lattice_size=lattice_size,
            encoder_channels=encoder_channels,
            proj_hidden=proj_hidden,
            proj_out=proj_out,
            use_batchnorm=use_bn,
            activation_name=act_name,
        ).to(device)
    elif encoder_type == "multi_scale":
        kernel_sizes = model_cfg.get("kernel_sizes", [3, 7, 15])
        model = MultiScaleSimCLREncoder(
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
        raise ValueError(f"Unknown encoder_type: {encoder_type}")

    print(model)

    # Optimizer / Scheduler / AMP
    train_cfg = cfg["train"]
    optimizer = setup_optimizer(model, train_cfg)
    scheduler = setup_scheduler(optimizer, train_cfg)

    use_amp = bool(train_cfg.get("amp", True)) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    grad_clip = train_cfg.get("grad_clip", None)
    if grad_clip is not None:
        grad_clip = float(grad_clip)

    # Logging
    logging_cfg = cfg.get("logging", {})
    ensure_dirs(logging_cfg)

    ckpt_dir = Path(logging_cfg.get("checkpoint_dir", "results/checkpoint/contrastive"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    log_interval = int(logging_cfg.get("log_interval", 50))
    val_interval = int(logging_cfg.get("val_interval", 1))

    best_val = float("inf")
    num_epochs = int(train_cfg.get("num_epochs", 200))

    # Loss config
    contr_cfg = cfg.get("contrastive", {})
    temperature = float(contr_cfg.get("temperature", 0.5))

    for epoch in range(1, num_epochs + 1):
        model.train()
        running = 0.0

        for it, batch in enumerate(train_loader, start=1):
            # Dataset may return (v1,v2,T) or (v1,v2)
            if len(batch) == 3:
                v1, v2, _T = batch
            else:
                v1, v2 = batch

            v1 = v1.to(device, non_blocking=True)
            v2 = v2.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                z1 = model(v1)
                z2 = model(v2)
                loss = nt_xent_loss(z1, z2, temperature)

            scaler.scale(loss).backward()
            if grad_clip is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()

            running += loss.item() * v1.size(0)

            if it % log_interval == 0:
                print(
                    f"[Epoch {epoch:03d}][Iter {it:04d}/{len(train_loader):04d}] "
                    f"loss={loss.item():.6f}"
                )

        train_loss = running / n_train

        # Validation
        if (epoch % val_interval) == 0 and len(val_loader) > 0:
            model.eval()
            val_running = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    if len(batch) == 3:
                        v1, v2, _T = batch
                    else:
                        v1, v2 = batch
                    v1 = v1.to(device, non_blocking=True)
                    v2 = v2.to(device, non_blocking=True)
                    z1 = model(v1)
                    z2 = model(v2)
                    vloss = nt_xent_loss(z1, z2, temperature)
                    val_running += vloss.item() * v1.size(0)
            val_loss = val_running / n_val
        else:
            val_loss = float("nan")

        if scheduler is not None:
            scheduler.step()

        print(
            f"[Epoch {epoch:03d}] train_loss={train_loss:.6f} "
            f"val_loss={val_loss:.6f}"
        )

        # Checkpointing: update best on non-NaN val_loss
        if not (val_loss != val_loss):  # not NaN
            if val_loss < best_val:
                best_val = val_loss
                best_path = ckpt_dir / "best_contrastive.pt"
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "cfg": cfg,
                    },
                    best_path,
                )
                print(
                    f"[INFO] Best model updated: {best_path} "
                    f"(val_loss={best_val:.6f})"
                )

        last_path = ckpt_dir / "last_contrastive.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "cfg": cfg,
            },
            last_path,
        )

    print("[INFO] Contrastive training finished.")


# ============================================================
# Entry point
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config YAML (default: ../config/contrastive.yaml)",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]

    if args.config is None:
        config_path = project_root / "config" / "contrastive.yaml"
    else:
        config_path = Path(args.config)

    print(f"[INFO] Loading config from: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
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

    train_contrastive(cfg)


if __name__ == "__main__":
    main()

"""
Usage note
----------
model:
  encoder_type: "multi_scale"
  encoder_channels: [96, 192, 384]   # example: 96 must be divisible by num_scales=3
  kernel_sizes: [3, 7, 15]
  use_batchnorm: true
  activation: "relu"
  projector:
    hidden_dim: 256
    out_dim: 64
"""
