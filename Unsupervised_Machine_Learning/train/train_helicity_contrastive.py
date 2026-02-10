# train/train_helicity_contrastive.py
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
from typing import Dict, Any, Optional

import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from models.contrastive_encoder import SimCLREncoder, MultiScaleSimCLREncoder
from models.helicity_head import HelicityModulusHead
from dataset.unsupervised_xy_dataset import create_xy_dataset_from_config
from utils.lambda_schedule import parse_lambda_spec


"""
train_helicity_contrastive.py
--------------------------------------------------

Contrastive learning with weak helicity modulus regression constraint (Υ head).

Key idea
--------
- Contrastive loss uses projected representations z.
- Helicity head is applied on encoder representations h (pre-projection).

Reproducibility notes
---------------------
This script sets torch seeds, but exact reproducibility still depends on:
- DataLoader workers RNG (num_workers > 0)
- random_split (torch RNG)
- CUDA/CuDNN deterministic flags
- augmentation RNG in dataset (numpy RNG inside dataset)

This file intentionally does NOT override global deterministic flags.
If you need strict determinism, set those flags in your main runner and
control DataLoader worker seeds.
"""


# ============================================================
# Utilities
# ============================================================

def setup_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def ensure_dirs(logging_cfg: Dict[str, Any]) -> None:
    ckpt_dir = Path(logging_cfg.get("checkpoint_dir", "results/checkpoint/helicity_contrastive"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)


def setup_optimizer(params, cfg_train: Dict[str, Any]):
    opt_cfg = cfg_train.get("optimizer", {})
    opt_type = str(opt_cfg.get("type", cfg_train.get("optimizer", "adam"))).lower()

    lr = float(opt_cfg.get("lr", cfg_train.get("lr", 1e-3)))
    wd = float(opt_cfg.get("weight_decay", cfg_train.get("weight_decay", 0.0)))

    if opt_type == "adam":
        betas = opt_cfg.get("betas", (0.9, 0.999))
        return torch.optim.Adam(params, lr=lr, weight_decay=wd, betas=betas)
    elif opt_type == "adamw":
        betas = opt_cfg.get("betas", (0.9, 0.999))
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd, betas=betas)
    elif opt_type == "sgd":
        momentum = float(opt_cfg.get("momentum", 0.9))
        return torch.optim.SGD(params, lr=lr, weight_decay=wd, momentum=momentum)
    else:
        raise ValueError(f"Unknown optimizer: {opt_type}")


def setup_scheduler(optimizer, cfg_train: Dict[str, Any]):
    sch_cfg = cfg_train.get("scheduler", {})
    sch_type = str(sch_cfg.get("type", "none")).lower()

    if sch_type == "none":
        return None
    elif sch_type == "cosine":
        T_max = int(sch_cfg.get("T_max", cfg_train.get("num_epochs", 200)))
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)
    elif sch_type == "step":
        step_size = int(sch_cfg.get("step_size", 50))
        gamma = float(sch_cfg.get("gamma", 0.5))
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    else:
        raise ValueError(f"Unknown scheduler: {sch_type}")


# ============================================================
# NT-Xent loss (SimCLR)
# ============================================================

def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float) -> torch.Tensor:
    """
    SimCLR NT-Xent loss (2B samples; positives: i <-> i+B).

    Implementation details
    ----------------------
    - z1,z2 normalized
    - sim matrix computed by cosine similarity / temperature
    - positives extracted from +/- batch_size diagonals
    - denominator computed via logsumexp
    """
    batch_size = z1.size(0)
    device = z1.device

    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)

    z = torch.cat([z1, z2], dim=0)  # (2B, D)

    # cosine similarity matrix
    sim = torch.matmul(z, z.T) / temperature  # (2B, 2B)

    # mask self-similarity
    mask = torch.eye(2 * batch_size, device=device, dtype=torch.bool)
    sim = sim.masked_fill(mask, -1e9)

    # positives: i <-> i+B
    positives = torch.cat(
        [torch.diag(sim, batch_size), torch.diag(sim, -batch_size)], dim=0
    )  # (2B,)

    # denominator: all except self
    denom = torch.logsumexp(sim, dim=1)  # (2B,)

    loss = -positives + denom
    return loss.mean()


# ============================================================
# Config helpers
# ============================================================

def _require_helicity_enabled(data_cfg: Dict[str, Any]) -> None:
    helicity_cfg = data_cfg.get("helicity", {}) or {}
    if not bool(helicity_cfg.get("enable", False)):
        raise ValueError(
            "Helicity self-supervision is required, but data.helicity.enable is False.\n"
            "Set:\n"
            "  data:\n"
            "    helicity:\n"
            "      enable: true\n"
            "      targets_path: ...\n"
            "      mode: by_index (recommended)\n"
            "      target_key: Y\n"
        )


def _get_lambdaY_spec(cfg: Dict[str, Any]) -> str:
    """
    λ_Y is specified as a string and parsed by parse_lambda_spec.

    To keep the training logic fixed and config flexible, accept multiple locations:
      1) cfg["helicity"]["lambda_Y"]
      2) cfg["loss"]["lambda_Y"]
      3) cfg["helicity"]["lambda"]     (backward compatibility)
      4) default "const(1e-2)"
    """
    if isinstance(cfg.get("helicity", None), dict):
        hcfg = cfg["helicity"]
        if "lambda_Y" in hcfg:
            return str(hcfg["lambda_Y"])
        if "lambda" in hcfg:
            return str(hcfg["lambda"])

    if isinstance(cfg.get("loss", None), dict):
        lcfg = cfg["loss"]
        if "lambda_Y" in lcfg:
            return str(lcfg["lambda_Y"])

    return "const(1e-2)"


def _build_model(
    cfg: Dict[str, Any],
    data_cfg: Dict[str, Any],
    device: torch.device,
):
    """
    Build encoder + helicity head.

    IMPORTANT
    ---------
    - Helicity head is attached to encoder representation h (pre-projection).
    """
    model_cfg = cfg["model"]

    in_channels = int(data_cfg.get("in_channels", 1))
    lattice_size = int(data_cfg.get("lattice_size", 32))

    encoder_channels = list(model_cfg["encoder_channels"])
    proj_hidden = int(model_cfg["projector"]["hidden_dim"])
    proj_out = int(model_cfg["projector"]["out_dim"])
    use_bn = bool(model_cfg.get("use_batchnorm", True))
    act_name = str(model_cfg.get("activation", "relu"))

    encoder_type = str(model_cfg.get("encoder_type", "single")).lower()
    # Accept aliases
    if encoder_type in ("cnn", "single_scale", "singlescale"):
        encoder_type = "single"

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

    # Helicity head: h -> scalar
    head_cfg = model_cfg.get("helicity_head", {}) or {}
    # Policy: Linear(D->D/4) + ReLU + Linear(D/4->1)
    hidden_dim = int(head_cfg.get("hidden_dim", max(16, model.enc_out_dim // 4)))
    helicity_head = HelicityModulusHead(in_dim=model.enc_out_dim, hidden_dim=hidden_dim).to(device)

    return model, helicity_head


# ============================================================
# Training
# ============================================================

def train_helicity_contrastive(cfg: Dict[str, Any]) -> None:
    # Seed control (torch only; dataset augmentations may use numpy RNG)
    seed = int(cfg.get("seed", 42))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = setup_device()
    print(f"[INFO] Using device: {device}")

    # Dataset (contrastive) - helicity supervision required
    data_cfg = cfg["data"]
    _require_helicity_enabled(data_cfg)

    full_dataset = create_xy_dataset_from_config(
        data_cfg,
        mode="contrastive",
        augment_config=data_cfg.get("augment", {}),
    )

    # Split (random_split uses torch RNG)
    val_ratio = float(data_cfg.get("val_ratio", 0.1))
    n_total = len(full_dataset)
    n_val = int(n_total * val_ratio)
    n_train = n_total - n_val
    train_dataset, val_dataset = random_split(full_dataset, [n_train, n_val])

    batch_size = int(data_cfg.get("batch_size", 256))
    num_workers = int(data_cfg.get("num_workers", 0))
    pin_memory = bool(data_cfg.get("pin_memory", True))
    drop_last = bool(data_cfg.get("drop_last", True))

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )

    # Model
    model, helicity_head = _build_model(cfg, data_cfg, device)
    print(model)
    print(helicity_head)

    # Optimizer / Scheduler / AMP
    train_cfg = cfg["train"]
    optimizer = setup_optimizer(list(model.parameters()) + list(helicity_head.parameters()), train_cfg)
    scheduler = setup_scheduler(optimizer, train_cfg)

    use_amp = bool(train_cfg.get("amp", True)) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    grad_clip = train_cfg.get("grad_clip", None)
    if grad_clip is not None:
        grad_clip = float(grad_clip)

    # Loss configs
    contr_cfg = cfg.get("contrastive", {}) or {}
    temperature = float(contr_cfg.get("temperature", 0.5))

    mse = nn.MSELoss()

    # λ_Y schedule (string spec)
    lambdaY_spec = _get_lambdaY_spec(cfg)
    lambdaY_fn = parse_lambda_spec(lambdaY_spec)
    print(f"[INFO] lambda_Y spec = {lambdaY_spec}")

    # Logging
    logging_cfg = cfg.get("logging", {}) or {}
    ensure_dirs(logging_cfg)
    ckpt_dir = Path(logging_cfg.get("checkpoint_dir", "results/checkpoint/helicity_contrastive"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    log_interval = int(logging_cfg.get("log_interval", 50))
    val_interval = int(logging_cfg.get("val_interval", 1))
    save_best = bool(logging_cfg.get("save_best", True))
    save_last = bool(logging_cfg.get("save_last", True))

    best_val = float("inf")
    num_epochs = int(train_cfg.get("num_epochs", 200))

    # Define progress to drive lambda schedule
    total_steps = max(1, num_epochs * len(train_loader))
    global_step = 0

    for epoch in range(1, num_epochs + 1):
        model.train()
        helicity_head.train()

        running_total = 0.0
        running_contr = 0.0
        running_Y = 0.0
        running_Y1 = 0.0
        running_Y2 = 0.0
        running_lamY = 0.0
        running_lamYLY = 0.0

        for it, batch in enumerate(train_loader, start=1):
            # Assumption: dataset returns (v1, v2, helicity, T)
            v1, v2, y_true, T = batch
            _ = T  # explicitly unused (kept for thesis traceability)

            v1 = v1.to(device, non_blocking=True)
            v2 = v2.to(device, non_blocking=True)
            y_true = y_true.to(device, non_blocking=True).float().view(-1)  # (B,)

            optimizer.zero_grad(set_to_none=True)

            global_step += 1
            progress = float(global_step) / float(total_steps)
            lamY = float(lambdaY_fn(progress, global_step, epoch))

            with torch.cuda.amp.autocast(enabled=use_amp):
                # h for helicity, z for contrastive
                h1 = model.encode(v1)
                h2 = model.encode(v2)
                z1 = model.project(h1)
                z2 = model.project(h2)

                loss_contr = nt_xent_loss(z1, z2, temperature)

                # Helicity head on h (NOT z)
                y_pred1 = helicity_head(h1).view(-1)
                y_pred2 = helicity_head(h2).view(-1)
                y_pred = 0.5 * (y_pred1 + y_pred2)

                loss_Y = mse(y_pred, y_true)
                loss_Y1 = mse(y_pred1, y_true)  # logging
                loss_Y2 = mse(y_pred2, y_true)  # logging

                loss = loss_contr + (lamY * loss_Y)

            scaler.scale(loss).backward()

            if grad_clip is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    list(model.parameters()) + list(helicity_head.parameters()),
                    grad_clip,
                )

            scaler.step(optimizer)
            scaler.update()

            bsz = v1.size(0)
            running_total += loss.item() * bsz
            running_contr += loss_contr.item() * bsz
            running_Y += loss_Y.item() * bsz
            running_Y1 += loss_Y1.item() * bsz
            running_Y2 += loss_Y2.item() * bsz
            running_lamY += lamY * bsz
            running_lamYLY += (lamY * loss_Y.item()) * bsz

            if it % log_interval == 0:
                denom = (it * bsz)
                print(
                    f"[Epoch {epoch:03d} | it {it:04d} | prog {progress:.3f}] "
                    f"loss={running_total/denom:.6f} "
                    f"(contr={running_contr/denom:.6f}, "
                    f"Y={running_Y/denom:.6f}, Y1={running_Y1/denom:.6f}, Y2={running_Y2/denom:.6f}, "
                    f"lambdaY={running_lamY/denom:.6g}, lambdaY*Y={running_lamYLY/denom:.6f})"
                )

        train_loss = running_total / max(1, n_train)
        train_contr = running_contr / max(1, n_train)
        train_Y = running_Y / max(1, n_train)
        train_Y1 = running_Y1 / max(1, n_train)
        train_Y2 = running_Y2 / max(1, n_train)
        train_lamY = running_lamY / max(1, n_train)
        train_lamYLY = running_lamYLY / max(1, n_train)

        if scheduler is not None:
            scheduler.step()

        # Validation
        if (epoch % val_interval == 0) and (n_val > 0):
            model.eval()
            helicity_head.eval()

            val_running_total = 0.0
            val_running_contr = 0.0
            val_running_Y = 0.0
            val_running_Y1 = 0.0
            val_running_Y2 = 0.0
            val_running_lamY = 0.0
            val_running_lamYLY = 0.0

            with torch.no_grad():
                for batch in val_loader:
                    v1, v2, y_true, T = batch
                    _ = T

                    v1 = v1.to(device, non_blocking=True)
                    v2 = v2.to(device, non_blocking=True)
                    y_true = y_true.to(device, non_blocking=True).float().view(-1)

                    # Use the same progress definition as training (global_step does not advance in validation)
                    progress = float(global_step) / float(total_steps)
                    lamY = float(lambdaY_fn(progress, global_step, epoch))

                    h1 = model.encode(v1)
                    h2 = model.encode(v2)
                    z1 = model.project(h1)
                    z2 = model.project(h2)

                    v_contr = nt_xent_loss(z1, z2, temperature)

                    y_pred1 = helicity_head(h1).view(-1)
                    y_pred2 = helicity_head(h2).view(-1)
                    y_pred = 0.5 * (y_pred1 + y_pred2)

                    v_Y = mse(y_pred, y_true)
                    v_Y1 = mse(y_pred1, y_true)
                    v_Y2 = mse(y_pred2, y_true)

                    v_total = v_contr + (lamY * v_Y)

                    bsz = v1.size(0)
                    val_running_total += v_total.item() * bsz
                    val_running_contr += v_contr.item() * bsz
                    val_running_Y += v_Y.item() * bsz
                    val_running_Y1 += v_Y1.item() * bsz
                    val_running_Y2 += v_Y2.item() * bsz
                    val_running_lamY += lamY * bsz
                    val_running_lamYLY += (lamY * v_Y.item()) * bsz

            val_loss = val_running_total / max(1, n_val)
            val_contr = val_running_contr / max(1, n_val)
            val_Y = val_running_Y / max(1, n_val)
            val_Y1 = val_running_Y1 / max(1, n_val)
            val_Y2 = val_running_Y2 / max(1, n_val)
            val_lamY = val_running_lamY / max(1, n_val)
            val_lamYLY = val_running_lamYLY / max(1, n_val)
        else:
            val_loss = float("nan")
            val_contr = float("nan")
            val_Y = float("nan")
            val_Y1 = float("nan")
            val_Y2 = float("nan")
            val_lamY = float("nan")
            val_lamYLY = float("nan")

        print(
            f"[Epoch {epoch:03d}] "
            f"train_loss={train_loss:.6f} (contr={train_contr:.6f}, "
            f"Y={train_Y:.6f}, Y1={train_Y1:.6f}, Y2={train_Y2:.6f}, "
            f"lambdaY={train_lamY:.6g}, lambdaY*Y={train_lamYLY:.6f}) | "
            f"val_loss={val_loss:.6f} (contr={val_contr:.6f}, "
            f"Y={val_Y:.6f}, Y1={val_Y1:.6f}, Y2={val_Y2:.6f}, "
            f"lambdaY={val_lamY:.6g}, lambdaY*Y={val_lamYLY:.6f})"
        )

        # Checkpoint
        if save_best and not (val_loss != val_loss):  # not NaN
            if val_loss < best_val:
                best_val = val_loss
                best_path = ckpt_dir / "best_helicity_contrastive.pt"
                torch.save(
                    {
                        "epoch": epoch,
                        "global_step": global_step,
                        "model_state_dict": model.state_dict(),
                        "helicity_head_state_dict": helicity_head.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "cfg": cfg,
                        "lambdaY_spec": lambdaY_spec,
                    },
                    best_path,
                )
                print(f"[INFO] Best model updated: {best_path} (val_loss={best_val:.6f})")

        if save_last:
            last_path = ckpt_dir / "last_helicity_contrastive.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "global_step": global_step,
                    "model_state_dict": model.state_dict(),
                    "helicity_head_state_dict": helicity_head.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "cfg": cfg,
                    "lambdaY_spec": lambdaY_spec,
                },
                last_path,
            )

    print("[INFO] Helicity-Contrastive training finished.")


# ============================================================
# Entry point
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config.")
    args = parser.parse_args()

    if args.config is None:
        raise ValueError("--config is not specified.")

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = (PROJECT_ROOT / config_path).resolve()

    print(f"[INFO] Loading config from: {config_path}")

    # Use utf-8-sig to avoid BOM issues on Windows
    with open(config_path, "r", encoding="utf-8-sig") as f:
        cfg = yaml.safe_load(f)

    train_helicity_contrastive(cfg)


if __name__ == "__main__":
    main()
