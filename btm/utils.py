"""Shared training, mode-connectivity and BTM utilities.

This module deliberately contains the reusable experiment logic that was
previously duplicated across dataset-specific notebooks.
"""
from __future__ import annotations

import copy
import math
import os
import random
import time
from collections import OrderedDict
from dataclasses import dataclass
from itertools import cycle
from pathlib import Path
from typing import Callable, Dict, Iterable, Mapping, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.func import functional_call
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm


# ---------------------------------------------------------------------------
# Reproducibility / paths
# ---------------------------------------------------------------------------

DATASET_CHOICES = ("eicu", "mimic3_ihm", "mimic3_ph")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(device: str | torch.device | None = None) -> torch.device:
    """Return an explicit PyTorch device without querying CUDA memory on CPU."""
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_time() -> str:
    return str(time.strftime("[%H:%M:%S]", time.localtime()))


def default_net_type(dataset: str) -> str:
    return "dnn" if dataset == "eicu" else "tcn2"


def experiment_paths(
    dataset: str,
    net_type: str,
    optim: str,
    *,
    root: str | os.PathLike = ".",
) -> dict[str, Path]:
    """Return consistent artifact paths for one dataset/model/optimiser."""
    dataset = dataset.lower()
    base = Path(root)
    stem = f"{net_type}_{optim}"
    trajectory_dir = base / "trajectories" / dataset
    return {
        "trajectory": trajectory_dir / f"{stem}_trajectories.pt",
        "modes": trajectory_dir / f"{stem}_modes.pt",
        "trajectory_dir": trajectory_dir,
    }


def get_training_hyperparameters(
    dataset: str,
    optim: str,
    *,
    stage: str = "teacher",
) -> tuple[float, float, int]:
    """Hyperparameters retained from the original notebooks.

    ``stage="teacher"`` reproduces teacher-trajectory settings.
    ``stage="btm_eval"`` reproduces the synthetic-set evaluation settings.
    """
    dataset = dataset.lower()
    optim_key = "sgd" if optim.lower() == "gsam" else optim.lower()

    teacher = {
        ("eicu", "sgd"): (0.02, 0.9, 100),
        ("mimic3_ihm", "sgd"): (0.02, 0.0, 60),
        ("mimic3_ihm", "adam"): (5e-5, 0.0, 60),
        ("mimic3_ph", "sgd"): (0.05, 0.9, 60),
        ("mimic3_ph", "adam"): (5e-5, 0.0, 60),
    }
    btm_eval = {
        ("eicu", "sgd"): (0.05, 0.9, 100),
        ("mimic3_ihm", "sgd"): (0.02, 0.9, 60),
        ("mimic3_ihm", "adam"): (5e-5, 0.0, 60),
        ("mimic3_ph", "sgd"): (0.10, 0.9, 50),
        ("mimic3_ph", "adam"): (5e-5, 0.0, 60),
    }
    table = teacher if stage == "teacher" else btm_eval if stage == "btm_eval" else None
    if table is None:
        raise ValueError("stage must be 'teacher' or 'btm_eval'.")
    try:
        lr, momentum, epochs = table[(dataset, optim_key)]
        if stage == "teacher" and optim.lower() == "gsam" and dataset == "mimic3_ihm":
            momentum = 0.9
        return lr, momentum, epochs
    except KeyError as exc:
        raise ValueError(
            f"No {stage} configuration for dataset={dataset!r}, optim={optim!r}."
        ) from exc


def get_mode_hyperparameters(dataset: str) -> dict[str, float | int | str]:
    """Mode-connection settings retained from the three original notebooks."""
    configs = {
        "eicu": dict(num_mc_samples=5, num_steps=200, lr=1e-2, t_sampling="uniform", alpha=3.0),
        "mimic3_ihm": dict(num_mc_samples=8, num_steps=300, lr=1e-3, t_sampling="uniform", alpha=3.0),
        "mimic3_ph": dict(num_mc_samples=8, num_steps=300, lr=1e-3, t_sampling="mid_beta", alpha=15.0),
    }
    try:
        return configs[dataset.lower()].copy()
    except KeyError as exc:
        raise ValueError(f"Unknown dataset: {dataset!r}") from exc


# ---------------------------------------------------------------------------
# Evaluation / teacher trajectory training
# ---------------------------------------------------------------------------

def _unpack_batch(batch):
    if not isinstance(batch, (tuple, list)) or len(batch) < 2:
        raise ValueError("Expected dataloader batches containing at least (x, y).")
    return batch[0], batch[1]


def _prepare_logits(logits: torch.Tensor, targets: torch.Tensor, task: str) -> torch.Tensor:
    if task == "binary":
        if logits.ndim == 2 and logits.shape[-1] == 1:
            return logits[:, 0]
        return logits.squeeze()
    return logits


@torch.no_grad()
def prediction_binary(
    model: nn.Module,
    loader: Iterable,
    loss_fn: Callable,
    device: str | torch.device,
) -> tuple[float, float, float]:
    model.eval()
    total_loss = 0.0
    total_count = 0
    logits_all, labels_all = [], []

    for batch in loader:
        x, y = _unpack_batch(batch)
        x = x.to(device, dtype=torch.float32)
        y = y.to(device, dtype=torch.float32)

        logits = _prepare_logits(model(x), y, "binary")
        loss = loss_fn(logits, y)

        bs = x.shape[0]
        total_loss += loss.item() * bs
        total_count += bs
        logits_all.append(logits.detach().cpu())
        labels_all.append(y.detach().cpu())

    logits_np = torch.cat(logits_all).numpy()
    labels_np = torch.cat(labels_all).numpy()
    auc = roc_auc_score(labels_np, logits_np)
    apr = average_precision_score(labels_np, logits_np)
    return float(auc), total_loss / max(total_count, 1), float(apr)


@torch.no_grad()
def prediction_multilabel(
    model: nn.Module,
    loader: Iterable,
    loss_fn: Callable,
    device: str | torch.device,
    *,
    per_class: bool = False,
) -> dict[str, object]:
    model.eval()
    total_loss = 0.0
    total_count = 0
    logits_all, targets_all = [], []

    for batch in loader:
        x, y = _unpack_batch(batch)
        x = x.to(device, dtype=torch.float32)
        y = y.to(device, dtype=torch.float32)
        logits = model(x)
        loss = loss_fn(logits, y)

        bs = x.shape[0]
        total_loss += loss.item() * bs
        total_count += bs
        logits_all.append(logits.detach().cpu())
        targets_all.append(y.detach().cpu())

    logits_np = torch.cat(logits_all).numpy()
    targets_np = torch.cat(targets_all).numpy()
    probs = torch.sigmoid(torch.from_numpy(logits_np)).numpy()

    n_classes = targets_np.shape[1]
    per_class_auc = np.full(n_classes, np.nan)
    per_class_apr = np.full(n_classes, np.nan)
    per_class_pos = targets_np.sum(axis=0).astype(int)

    for c in range(n_classes):
        y_true = targets_np[:, c]
        if len(np.unique(y_true)) < 2:
            continue
        per_class_auc[c] = roc_auc_score(y_true, probs[:, c])
        per_class_apr[c] = average_precision_score(y_true, probs[:, c])

    result = {
        "loss": total_loss / max(total_count, 1),
        "macro_auc": float(np.nanmean(per_class_auc)),
        "macro_apr": float(np.nanmean(per_class_apr)),
    }
    if per_class:
        result.update(
            per_class_auc=per_class_auc,
            per_class_apr=per_class_apr,
            per_class_pos=per_class_pos,
        )
    return result


@torch.no_grad()
def evaluate_loss(
    model: nn.Module,
    loader: Iterable,
    loss_fn: Callable,
    device: str | torch.device,
    *,
    task: str = "binary",
) -> float:
    model.eval()
    total_loss = 0.0
    total_count = 0
    for batch in loader:
        x, y = _unpack_batch(batch)
        x = x.to(device, dtype=torch.float32)
        y = y.to(device, dtype=torch.float32)
        logits = _prepare_logits(model(x), y, task)
        loss = loss_fn(logits, y)
        bs = x.shape[0]
        total_loss += loss.item() * bs
        total_count += bs
    return total_loss / max(total_count, 1)


def evaluate_model(
    model: nn.Module,
    loader: Iterable,
    loss_fn: Callable,
    device: str | torch.device,
    task: str,
) -> dict[str, float]:
    if task == "multilabel":
        metrics = prediction_multilabel(model, loader, loss_fn, device)
        return {
            "loss": float(metrics["loss"]),
            "auc": float(metrics["macro_auc"]),
            "apr": float(metrics["macro_apr"]),
        }
    auc, loss, apr = prediction_binary(model, loader, loss_fn, device)
    return {"loss": loss, "auc": auc, "apr": apr}


def train_teacher_trajectories(
    *,
    model_factory: Callable[[], nn.Module],
    train_loader: Iterable,
    val_loader: Iterable,
    test_loader: Iterable,
    task: str,
    device: str | torch.device,
    num_experts: int,
    optim: str,
    num_epochs: int,
    lr: float,
    momentum: float = 0.0,
    weight_decay: float = 5e-4,
    loss_fn: Optional[Callable] = None,
    verbose: bool = True,
) -> list[dict[str, object]]:
    """Train expert models and retain the initial state plus every epoch state."""
    loss_fn = loss_fn or nn.BCEWithLogitsLoss()
    trajectories = []

    for expert_idx in range(num_experts):
        if verbose:
            print("=" * 64)
            print(f"Training expert {expert_idx + 1}/{num_experts}")

        model = model_factory().to(device)
        trajectory = [
            {k: v.detach().clone().cpu() for k, v in model.state_dict().items()}
        ]

        optim_lower = optim.lower()
        if optim_lower in {"sgd", "gsam"}:
            optimizer = torch.optim.SGD(
                model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
            )
        elif optim_lower == "adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unsupported optimiser: {optim}")

        gsam_optim = lr_scheduler = None
        if optim_lower == "gsam":
            try:
                from pytorch_optimizer import GSAM, LinearScheduler, ProportionScheduler
            except ImportError as exc:
                raise ImportError(
                    "GSAM requires `pytorch-optimizer`; install requirements.txt."
                ) from exc

            total_steps = num_epochs * len(train_loader)
            lr_scheduler = LinearScheduler(optimizer, max_lr=lr, t_max=total_steps)
            rho_scheduler = ProportionScheduler(
                lr_scheduler, max_lr=lr, min_lr=lr, max_value=0, min_value=0
            )
            gsam_optim = GSAM(model.parameters(), optimizer, model, rho_scheduler)

            def gsam_loss_fn(pred, y):
                y = y.to(pred.device, dtype=torch.float32)
                pred = _prepare_logits(pred, y, task)
                return loss_fn(pred, y)

        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0

            for batch in train_loader:
                x, y = _unpack_batch(batch)
                x = x.to(device, dtype=torch.float32)
                y = y.to(device, dtype=torch.float32)
                optimizer.zero_grad()

                if optim_lower == "gsam":
                    gsam_optim.set_closure(gsam_loss_fn, x, y)
                    _, loss = gsam_optim.step()
                    lr_scheduler.step()
                    gsam_optim.update_rho_t()
                else:
                    logits = _prepare_logits(model(x), y, task)
                    loss = loss_fn(logits, y)
                    loss.backward()
                    optimizer.step()

                running_loss += loss.item()

            val = evaluate_model(model, val_loader, loss_fn, device, task)
            if verbose:
                print(
                    f"{get_time()} Epoch {epoch + 1}/{num_epochs} | "
                    f"train={running_loss / max(len(train_loader), 1):.4f} | "
                    f"val={val['loss']:.4f} | AUC={val['auc']:.4f} | APR={val['apr']:.4f}"
                )

            trajectory.append(
                {k: v.detach().clone().cpu() for k, v in model.state_dict().items()}
            )

        test = evaluate_model(model, test_loader, loss_fn, device, task)
        if verbose:
            print(
                f"Expert {expert_idx + 1}: test AUC={test['auc']:.4f}, "
                f"APR={test['apr']:.4f}"
            )

        trajectories.append(
            {
                "trajectory": trajectory,
                "test_auc": test["auc"],
                "test_apr": test["apr"],
            }
        )

    return trajectories


# ---------------------------------------------------------------------------
# Bezier / mode connectivity
# ---------------------------------------------------------------------------

def _to_device(mapping: Mapping[str, torch.Tensor], device) -> OrderedDict:
    return OrderedDict((k, v.to(device)) for k, v in mapping.items())


def _sample_t(mode: str, device, alpha: float | None = None) -> torch.Tensor:
    mode = mode.lower()
    if mode == "uniform":
        return torch.rand((), device=device)
    if mode in {"mid_beta", "beta"}:
        a = float(alpha if alpha is not None else 3.0)
        dist = torch.distributions.Beta(
            torch.tensor(a, device=device), torch.tensor(a, device=device)
        )
        return dist.sample()
    raise ValueError("t_sampling must be 'uniform' or 'mid_beta'.")


def _control_value(theta_c, parameter_name: str) -> torch.Tensor:
    if theta_c is None:
        raise KeyError(parameter_name)
    if parameter_name in theta_c:
        return theta_c[parameter_name]
    escaped = parameter_name.replace(".", "__")
    if escaped in theta_c:
        return theta_c[escaped]
    raise KeyError(parameter_name)


def _make_control_point(
    model: nn.Module,
    theta_a: Mapping[str, torch.Tensor],
    theta_b: Mapping[str, torch.Tensor],
    device,
) -> nn.ParameterDict:
    control = {}
    for name, _ in model.named_parameters():
        if name not in theta_a or name not in theta_b:
            raise KeyError(f"Checkpoint is missing model parameter {name!r}.")
        midpoint = 0.5 * (theta_a[name].to(device) + theta_b[name].to(device))
        control[name.replace(".", "__")] = nn.Parameter(midpoint.detach().clone())
    return nn.ParameterDict(control)


def state_from_curve(
    model: nn.Module,
    theta_a: Mapping[str, torch.Tensor],
    theta_b: Mapping[str, torch.Tensor],
    theta_c: Optional[Mapping[str, torch.Tensor]],
    t,
    device: str | torch.device,
    *,
    curve_type: str = "bezier",
    interpolate_floating_buffers: bool = True,
) -> OrderedDict:
    """Build a full model state at position ``t``.

    Trainable parameters follow the requested linear/quadratic Bezier path.
    Floating buffers (e.g. BatchNorm running statistics) are linearly
    interpolated; integer buffers use A except at t=1.
    """
    if curve_type not in {"bezier", "linear"}:
        raise ValueError("curve_type must be 'bezier' or 'linear'.")

    param_names = {name for name, _ in model.named_parameters()}
    buffer_names = {name for name, _ in model.named_buffers()}
    model_keys = param_names | buffer_names
    missing = model_keys - set(theta_a)
    missing |= model_keys - set(theta_b)
    if missing:
        raise KeyError(f"Checkpoint keys do not match model. Missing: {sorted(missing)[:5]}")

    dtype = next(v.dtype for v in theta_a.values() if torch.is_floating_point(v))
    t = torch.as_tensor(t, device=device, dtype=dtype).reshape(())
    if bool((t < 0) | (t > 1)):
        raise ValueError("t must be in [0, 1].")
    omt = 1.0 - t
    t_value = float(t.detach().cpu())

    state = OrderedDict()
    for name in model.state_dict().keys():
        a = theta_a[name].to(device)
        b = theta_b[name].to(device)

        if name in param_names:
            if curve_type == "linear":
                state[name] = omt * a + t * b
            else:
                c = _control_value(theta_c, name).to(device)
                state[name] = omt.square() * a + 2.0 * omt * t * c + t.square() * b
        elif torch.is_floating_point(a) and interpolate_floating_buffers:
            state[name] = omt * a + t * b
        else:
            state[name] = a.clone() if t_value < 1.0 else b.clone()
    return state


def bezier_interpolation(
    theta_a: Mapping[str, torch.Tensor],
    theta_b: Mapping[str, torch.Tensor],
    phi: Mapping[str, torch.Tensor],
    t,
    device: str | torch.device = "cuda",
) -> OrderedDict:
    """Compatibility helper for parameter-only/simple models."""
    dtype = next(v.dtype for v in theta_a.values() if torch.is_floating_point(v))
    t = torch.as_tensor(t, device=device, dtype=dtype).clamp(0.0, 1.0)
    omt = 1.0 - t
    out = OrderedDict()
    for key in theta_a:
        a, b = theta_a[key].to(device), theta_b[key].to(device)
        try:
            c = _control_value(phi, key).to(device)
            out[key] = omt.square() * a + 2 * omt * t * c + t.square() * b
        except KeyError:
            out[key] = omt * a + t * b if torch.is_floating_point(a) else a.clone()
    return out


def _curve_state(
    model,
    theta_a,
    theta_b,
    theta_c,
    t,
    device,
    *,
    curve_type="bezier",
    interpolate_floating_buffers=True,
):
    return state_from_curve(
        model, theta_a, theta_b, theta_c, t, device,
        curve_type=curve_type,
        interpolate_floating_buffers=interpolate_floating_buffers,
    )


def _loss_from_state(
    model: nn.Module,
    state: Mapping[str, torch.Tensor],
    dataloader: Iterable,
    loss_fn: Callable,
    device,
) -> float:
    total_loss = 0.0
    total_count = 0
    for batch in dataloader:
        x, y = _unpack_batch(batch)
        x = x.to(device, dtype=torch.float32)
        y = y.to(device, dtype=torch.float32)
        logits = functional_call(model, state, (x,), tie_weights=False)
        task = "multilabel" if y.ndim > 1 else "binary"
        logits = _prepare_logits(logits, y, task)
        loss = loss_fn(logits, y)
        bs = x.shape[0]
        total_loss += loss.item() * bs
        total_count += bs
    return total_loss / max(total_count, 1)


def _grad_norm(parameters: Iterable[torch.Tensor]) -> float:
    sq = 0.0
    for p in parameters:
        if p.grad is not None:
            sq += float(torch.sum(p.grad.detach() ** 2).cpu())
    return math.sqrt(sq)


def evaluate_curve_grid(
    model: nn.Module,
    dataloader: Iterable,
    loss_fn: Callable,
    theta_a: Mapping[str, torch.Tensor],
    theta_b: Mapping[str, torch.Tensor],
    theta_c: Optional[Mapping[str, torch.Tensor]] = None,
    *,
    num_points: int = 41,
    device: str | torch.device | None = None,
    curve_type: str = "bezier",
    interpolate_floating_buffers: bool = True,
):
    device = device or next(model.parameters()).device
    model = model.to(device).eval()
    theta_a = _to_device(theta_a, device)
    theta_b = _to_device(theta_b, device)

    ts = torch.linspace(0.0, 1.0, num_points, device=device)
    losses = []
    for t in ts:
        state = state_from_curve(
            model, theta_a, theta_b, theta_c, t, device,
            curve_type=curve_type,
            interpolate_floating_buffers=interpolate_floating_buffers,
        )
        losses.append(_loss_from_state(model, state, dataloader, loss_fn, device))
    return ts.detach().cpu().numpy(), np.asarray(losses, dtype=float)


def learn_low_loss_bezier_curve(
    model: nn.Module,
    dataloader: Iterable,
    theta_a: Mapping[str, torch.Tensor],
    theta_b: Mapping[str, torch.Tensor],
    loss_fn: Callable,
    *,
    device: str | torch.device | None = None,
    num_mc_samples: int = 8,
    num_steps: int = 300,
    lr: float = 1e-3,
    wd: float = 0.0,
    grad_clip: float | None = None,
    t_sampling: str = "uniform",
    alpha: float | None = None,
    interpolate_floating_buffers: bool = True,
    optim: str = "sgd",
    momentum: float = 0.9,
    eps: float = 1e-5,
    log_every: int | None = None,
    verbose: bool = True,
) -> dict[str, object]:
    """Optimise the quadratic Bezier control point between two checkpoints."""
    device = device or next(model.parameters()).device
    model = model.to(device)
    model.train()

    theta_a = _to_device(theta_a, device)
    theta_b = _to_device(theta_b, device)
    theta_c = _make_control_point(model, theta_a, theta_b, device)

    if optim.lower() == "sgd":
        optimiser = torch.optim.SGD(
            theta_c.parameters(), lr=lr, momentum=momentum, weight_decay=wd
        )
    elif optim.lower() == "adam":
        optimiser = torch.optim.Adam(theta_c.parameters(), lr=lr, weight_decay=wd)
    else:
        raise ValueError(f"Unsupported optimiser: {optim}")

    log_every = log_every or max(1, num_steps // 20)
    result = {
        "theta_C": theta_c,
        "train_losses": [],
        "control_grads": [],
        "steps": [],
        "stopped_early": False,
    }
    data_iter = cycle(dataloader)

    for step in range(num_steps):
        optimiser.zero_grad(set_to_none=True)
        losses = []

        for _ in range(num_mc_samples):
            x, y = _unpack_batch(next(data_iter))
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.float32)
            t = _sample_t(t_sampling, device, alpha)

            state_t = state_from_curve(
                model, theta_a, theta_b, theta_c, t, device,
                curve_type="bezier",
                interpolate_floating_buffers=interpolate_floating_buffers,
            )
            logits = functional_call(model, state_t, (x,), tie_weights=False)
            task = "multilabel" if y.ndim > 1 else "binary"
            logits = _prepare_logits(logits, y, task)
            losses.append(loss_fn(logits, y))

        loss = torch.stack(losses).mean()
        loss.backward()
        grad_norm = _grad_norm(theta_c.parameters())

        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(theta_c.parameters(), grad_clip)

        should_log = (
            step == 0 or (step + 1) % log_every == 0
            or step == num_steps - 1 or grad_norm < eps
        )
        if should_log:
            result["train_losses"].append(float(loss.detach().cpu()))
            result["control_grads"].append(float(grad_norm))
            result["steps"].append(step + 1)
            if verbose:
                print(
                    f"It {step + 1:03d}/{num_steps:03d} | "
                    f"train_loss={loss.item():.4f} | control_grad={grad_norm:.6e}"
                )

        if grad_norm < eps:
            result["stopped_early"] = True
            break
        optimiser.step()

    return result


def learn_mode_connections(
    *,
    trajectories: Sequence[Mapping[str, object]],
    model_factory: Callable[[], nn.Module],
    train_loader: Iterable,
    loss_fn: Callable,
    device: str | torch.device,
    num_mc_samples: int = 8,
    num_steps: int = 300,
    lr: float = 1e-3,
    t_sampling: str = "uniform",
    alpha: float | None = None,
    verbose: bool = True,
) -> list[list[object]]:
    """Learn one Bezier control point for every expert trajectory."""
    modes = []
    for idx, record in enumerate(trajectories):
        if verbose:
            print(f"\nMode connection {idx + 1}/{len(trajectories)}")
        trajectory = record["trajectory"]
        theta_a = trajectory[0]
        theta_b = trajectory[-1]
        result = learn_low_loss_bezier_curve(
            model_factory(),
            train_loader,
            theta_a,
            theta_b,
            loss_fn,
            device=device,
            num_mc_samples=num_mc_samples,
            num_steps=num_steps,
            lr=lr,
            t_sampling=t_sampling,
            alpha=alpha,
            verbose=verbose,
        )
        # Store CPU tensors so mode files are device-agnostic.
        theta_c = nn.ParameterDict(
            {
                k: nn.Parameter(v.detach().cpu().clone(), requires_grad=False)
                for k, v in result["theta_C"].items()
            }
        )
        modes.append([theta_a, theta_b, theta_c])
    return modes


# ---------------------------------------------------------------------------
# Synthetic data initialisation / evaluation
# ---------------------------------------------------------------------------

def _collect_dataset(loader: Iterable) -> tuple[torch.Tensor, torch.Tensor]:
    xs, ys = [], []
    for batch in loader:
        x, y = _unpack_batch(batch)
        xs.append(x.detach().cpu())
        ys.append(y.detach().cpu())
    if not xs:
        raise ValueError("Dataloader is empty.")
    return torch.cat(xs, dim=0), torch.cat(ys, dim=0)


@torch.no_grad()
def construct_dsub_new(
    model: nn.Module,
    dataloader: Iterable,
    ipc: int,
    device: str | torch.device,
    threshold: float | None = None,
) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
    """Select confident real examples and their soft binary labels."""
    model.eval()
    candidates = {0: [], 1: []}
    all_probs, all_labels = [], []

    cached = []
    for batch in dataloader:
        x, y = _unpack_batch(batch)
        x = x.to(device, dtype=torch.float32)
        y = y.to(device, dtype=torch.float32)
        probs = torch.sigmoid(_prepare_logits(model(x), y, "binary"))
        cached.append((x, y, probs))
        all_probs.append(probs)
        all_labels.append(y)

    probs_all = torch.cat(all_probs)
    labels_all = torch.cat(all_labels)
    positive_probs = probs_all[labels_all == 1]
    if threshold is None:
        threshold = float(positive_probs.median()) if len(positive_probs) else 0.5

    counts = {0: 0, 1: 0}
    for x, y, probs in cached:
        for cls in (0, 1):
            if counts[cls] >= ipc:
                continue
            mask = (
                (y == 1) & (probs > threshold)
                if cls == 1
                else (y == 0) & (probs < (1 - threshold))
            )
            need = ipc - counts[cls]
            if mask.any():
                candidates[cls].append((x[mask][:need], probs[mask][:need]))
                counts[cls] += min(int(mask.sum()), need)
        if counts[0] >= ipc and counts[1] >= ipc:
            break

    out = {}
    for cls in (0, 1):
        if not candidates[cls]:
            raise ValueError(f"Could not find confident examples for class {cls}.")
        out[cls] = (
            torch.cat([item[0] for item in candidates[cls]], dim=0),
            torch.cat([item[1] for item in candidates[cls]], dim=0),
        )
    return out


def initialize_synthetic_binary(
    dataloader: Iterable,
    ipc: int,
    *,
    device: str | torch.device,
    initial_lr: float,
    init: str = "real",
    seed: int | None = None,
    pretrained_model: nn.Module | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Initialise the binary synthetic set used by eICU and MIMIC-IHM."""
    if seed is not None:
        set_seed(seed)
    x_real, y_real = _collect_dataset(dataloader)
    y_real = y_real.float()
    init = init.lower()

    if init == "real_soft":
        if pretrained_model is None:
            raise ValueError("pretrained_model is required for init='real_soft'.")
        selected = construct_dsub_new(pretrained_model, dataloader, ipc, device)
        xs, ys = [], []
        for cls in (0, 1):
            x_cls, y_soft = selected[cls]
            if len(x_cls) < ipc:
                raise ValueError(f"Not enough soft-labelled samples for class {cls}.")
            xs.append(x_cls[:ipc])
            ys.append(y_soft[:ipc])
        x_syn = torch.cat(xs).detach().clone().to(device).requires_grad_(True)
        y_syn = torch.cat(ys).detach().clone().to(device).requires_grad_(True)
    else:
        xs, ys = [], []
        for cls in (0, 1):
            idx = torch.where(y_real == cls)[0]
            if len(idx) == 0:
                raise ValueError(f"No real samples found for class {cls}.")
            if init == "real":
                choice = idx[torch.randint(len(idx), (ipc,))]
                x_init = x_real[choice].clone()
            elif init in {"rand", "class_normal"}:
                x_cls = x_real[idx].float()
                mu = x_cls.mean(dim=0)
                std = x_cls.std(dim=0) + 1e-5
                x_init = mu + torch.randn((ipc, *mu.shape)) * std
            else:
                raise ValueError("Binary init must be 'real', 'rand', or 'real_soft'.")
            xs.append(x_init)
            ys.append(torch.full((ipc,), float(cls)))
        x_syn = torch.cat(xs).to(device, dtype=torch.float32).detach().requires_grad_(True)
        y_syn = torch.cat(ys).to(device, dtype=torch.float32).detach()

    syn_lr = torch.tensor(
        float(initial_lr), device=device, dtype=torch.float32, requires_grad=True
    )
    return x_syn, y_syn, syn_lr


def initialize_synthetic_multilabel(
    dataloader: Iterable,
    bpl: int,
    *,
    initial_lr: float,
    device: str | torch.device,
    seed: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Initialise phenotype data using the original budget-per-label heuristic."""
    if seed is not None:
        set_seed(seed)
    x_real, y_real = _collect_dataset(dataloader)
    if y_real.ndim != 2:
        raise ValueError(f"Expected multilabel targets [N,C], got {tuple(y_real.shape)}")
    y_soft = y_real.float()
    num_labels = y_soft.shape[1]
    avg_cardinality = float(y_soft.sum(dim=1).mean())
    if avg_cardinality <= 0:
        raise ValueError("Average label cardinality is zero.")

    n_syn = max(
        num_labels,
        int(math.ceil((bpl * num_labels) / avg_cardinality)),
    )
    label_freq = y_soft.sum(dim=0)
    inv_freq = 1.0 / torch.clamp(label_freq, min=1.0)
    sample_weights = (y_soft * inv_freq.unsqueeze(0)).sum(dim=1)
    positive = sample_weights > 0
    if positive.any():
        sample_weights = torch.clamp(sample_weights, min=sample_weights[positive].min())
    else:
        sample_weights = torch.ones_like(sample_weights)

    indices = torch.multinomial(sample_weights, n_syn, replacement=True)
    x_syn = x_real[indices].clone().to(device).detach().requires_grad_(True)
    y_syn = y_soft[indices].clone().to(device).detach().requires_grad_(True)
    syn_lr = torch.tensor(
        float(initial_lr), device=device, dtype=torch.float32, requires_grad=True
    )
    return x_syn, y_syn, syn_lr


def train_and_evaluate(
    net: nn.Module,
    image_syn: torch.Tensor,
    label_syn: torch.Tensor,
    val_loader: Iterable,
    device: str | torch.device,
    criterion: Callable,
    *,
    task: str = "binary",
    epochs: int = 100,
    lr: float = 0.01,
    syn_batch: int = 256,
    optim: str = "sgd",
    momentum: float = 0.9,
    weight_decay: float = 5e-4,
) -> tuple[float, float, float, nn.Module]:
    """Train a fresh model on synthetic data, then evaluate on validation data."""
    if optim.lower() == "sgd":
        optimizer = torch.optim.SGD(
            net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
        )
    else:
        optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)

    ds = TensorDataset(
        image_syn.detach().clone().to(device),
        label_syn.detach().clone().to(device),
    )
    loader = DataLoader(ds, batch_size=min(syn_batch, len(ds), 256), shuffle=True)

    for _ in range(epochs):
        net.train()
        for x, y in loader:
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.float32)
            optimizer.zero_grad()
            logits = _prepare_logits(net(x), y, task)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

    metrics = evaluate_model(net, val_loader, criterion, device, task)
    return metrics["auc"], metrics["loss"], metrics["apr"], net


# Backwards-readable aliases
def train_and_evaluate_multilabel(*args, **kwargs):
    kwargs["task"] = "multilabel"
    return train_and_evaluate(*args, **kwargs)


# ---------------------------------------------------------------------------
# BTM geometry / optimisation
# ---------------------------------------------------------------------------

def compute_trajectory_loss(
    student_params: torch.Tensor,
    starting_params: torch.Tensor,
    target_params: torch.Tensor,
    *,
    eps: float = 1e-12,
) -> torch.Tensor:
    param_loss = F.mse_loss(student_params, target_params, reduction="mean")
    param_dist = F.mse_loss(starting_params, target_params, reduction="mean")
    return param_loss / (param_dist + eps)


def get_top_principal_components(span_matrix: torch.Tensor, k: int = 1) -> torch.Tensor:
    if span_matrix.ndim == 1:
        return span_matrix / (torch.norm(span_matrix) + 1e-8)

    n, d = span_matrix.shape
    if n > d:
        gram = span_matrix.T @ span_matrix
        _, eigenvectors = torch.linalg.eigh(gram)
        pcs = (span_matrix @ eigenvectors[:, -k:]).T
    else:
        gram = span_matrix @ span_matrix.T
        _, eigenvectors = torch.linalg.eigh(gram)
        pcs = eigenvectors[:, -k:].T @ span_matrix
    pcs = pcs / (torch.norm(pcs, dim=1, keepdim=True) + 1e-8)
    return torch.flip(pcs, dims=[0])


def compute_subspace_alignment(
    student_span: torch.Tensor,
    teacher_displacement: torch.Tensor,
    k: int = 1,
):
    teacher = teacher_displacement.to(student_span.device)
    student_pcs = get_top_principal_components(student_span, k=k)
    teacher_pcs = get_top_principal_components(teacher, k=k)
    similarity = torch.abs(student_pcs @ teacher_pcs.T)
    return similarity.item() if k == 1 else similarity.detach().cpu().numpy()


def compute_projection_residual(
    synthetic_span: torch.Tensor,
    teacher_displacement: torch.Tensor,
) -> float:
    teacher = teacher_displacement.to(synthetic_span.device)
    q, _ = torch.linalg.qr(synthetic_span, mode="reduced")
    projection = q @ (q.T @ teacher)
    return float(torch.sum((teacher - projection) ** 2).detach().cpu())


@dataclass
class BTMConfig:
    synthetic_budget: int
    iterations: int
    expert_window: float
    syn_steps: int
    batch_syn: int
    eval_every: int = 10
    lr_x: float = 100.0
    lr_y: float = 0.5
    lr_lr: float = 1e-6
    mom_x: float = 0.9
    mom_y: float = 0.9
    mom_lr: float = 0.5
    min_start_t: float = 0.0
    max_start_t: float | None = None
    learn_y: bool = False
    init: str = "real"
    sequential_generation: bool = False
    current_max_start_t: float = 0.0
    expansion_end_iteration: int | None = None
    compute_residual: bool = False
    update_loss_threshold: float = 2.0

    def __post_init__(self):
        if self.max_start_t is None:
            self.max_start_t = 1.0 - self.expert_window
        if self.expansion_end_iteration is None:
            self.expansion_end_iteration = max(1, self.iterations // 2)


@dataclass
class BTMResult:
    best_x_syn: torch.Tensor
    best_y_syn: torch.Tensor
    syn_lr: torch.Tensor
    best_val_apr: float
    history: list[dict[str, float]]
    save_path: Path


def run_btm(
    *,
    model_factory: Callable[[], nn.Module],
    train_loader: Iterable,
    val_loader: Iterable,
    modes: Sequence[Sequence[object]],
    task: str,
    device: str | torch.device,
    config: BTMConfig,
    eval_lr: float,
    eval_momentum: float,
    eval_epochs: int,
    optim: str = "sgd",
    save_path: str | os.PathLike,
    seed: int = 42,
    pretrained_model: nn.Module | None = None,
    logger: Callable[[dict[str, float]], None] | None = None,
    show_progress: bool = True,
) -> BTMResult:
    """Run the BTM synthetic-data optimisation used by the three notebooks.

    The update sequence follows the original notebooks closely; this refactor
    mainly centralises the repeated experiment plumbing.
    """
    set_seed(seed)
    device = torch.device(device)
    criterion = nn.BCEWithLogitsLoss()

    if task == "multilabel":
        x_syn, y_syn, syn_lr = initialize_synthetic_multilabel(
            train_loader,
            config.synthetic_budget,
            initial_lr=eval_lr,
            device=device,
            seed=seed,
        )
    else:
        x_syn, y_syn, syn_lr = initialize_synthetic_binary(
            train_loader,
            config.synthetic_budget,
            device=device,
            initial_lr=eval_lr,
            init=config.init,
            seed=seed,
            pretrained_model=pretrained_model,
        )

    x_syn = x_syn.to(device).detach().requires_grad_(True)
    y_syn = y_syn.to(device).detach().requires_grad_(config.learn_y)
    syn_lr = syn_lr.to(device).detach().requires_grad_(True)

    optimizer_x = torch.optim.SGD(
        [x_syn], lr=config.lr_x, momentum=config.mom_x, nesterov=True
    )
    optimizer_lr = torch.optim.SGD([syn_lr], lr=config.lr_lr, momentum=config.mom_lr)
    optimizer_y = None
    if config.learn_y:
        optimizer_y = torch.optim.SGD(
            [y_syn], lr=config.lr_y, momentum=config.mom_y, nesterov=True
        )

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    best_score = float("-inf")
    best_x_syn = x_syn.detach().cpu().clone()
    best_y_syn = y_syn.detach().cpu().clone()
    history = []
    running_tm_loss = 0.0
    span_similarity = 0.0
    residual = 0.0

    iterator = range(config.iterations)
    if show_progress:
        iterator = tqdm(iterator)

    for it in iterator:
        if it % config.eval_every == 0:
            eval_net = model_factory().to(device)
            val_auc, val_loss, val_apr, _ = train_and_evaluate(
                eval_net,
                x_syn,
                y_syn,
                val_loader,
                device,
                criterion,
                task=task,
                lr=eval_lr,
                epochs=eval_epochs,
                optim=optim,
                momentum=eval_momentum,
            )
            metrics = {
                "iteration": float(it),
                "val_auc": float(val_auc),
                "val_apr": float(val_apr),
                "val_loss": float(val_loss),
                "tm_loss": float(running_tm_loss / max(config.eval_every, 1)),
            }
            if config.compute_residual:
                metrics.update(
                    span_similarity=float(span_similarity),
                    residual=float(residual),
                )
            history.append(metrics)
            if logger is not None:
                logger(metrics)

            print(
                f"Itr {it + 1} | Val AUC={val_auc:.4f} | Val APR={val_apr:.4f} | "
                f"TM={metrics['tm_loss']:.5f}"
            )
            running_tm_loss = 0.0

            if np.isfinite(val_apr) and val_apr > best_score:
                best_score = float(val_apr)
                best_x_syn = x_syn.detach().cpu().clone()
                best_y_syn = y_syn.detach().cpu().clone()
                torch.save({"x_syn": best_x_syn, "y_syn": best_y_syn}, save_path)

        if config.sequential_generation:
            upper = config.current_max_start_t + (
                config.max_start_t - config.current_max_start_t
            ) * it / config.expansion_end_iteration
            upper = min(upper, config.max_start_t)
        else:
            upper = config.max_start_t
        start_t = float(np.random.uniform(config.min_start_t, upper))

        theta_a, theta_b, phi = modes[np.random.randint(len(modes))]
        surrogate_model = model_factory().to(device)

        starting_state = state_from_curve(
            surrogate_model, theta_a, theta_b, phi, start_t, device, curve_type="bezier"
        )
        target_state = state_from_curve(
            surrogate_model,
            theta_a,
            theta_b,
            phi,
            start_t + config.expert_window,
            device,
            curve_type="bezier",
        )

        # load_state_dict requires detached tensors; the start/target states are
        # fixed teacher geometry and do not need gradients.
        surrogate_model.load_state_dict(
            OrderedDict((k, v.detach()) for k, v in starting_state.items()), strict=True
        )

        indices = torch.randperm(len(x_syn), device=device)[: min(config.batch_syn, len(x_syn))]
        x_syn_batch = x_syn[indices]
        y_syn_batch = y_syn[indices]

        param_map = OrderedDict(surrogate_model.named_parameters())
        step_gradients = []
        updated_params = OrderedDict((k, p) for k, p in param_map.items())

        for _ in range(config.syn_steps):
            pred = surrogate_model(x_syn_batch)
            pred = _prepare_logits(pred, y_syn_batch, task)
            ce_loss = criterion(pred, y_syn_batch)
            grads = torch.autograd.grad(
                ce_loss, tuple(param_map.values()), create_graph=True
            )

            if config.compute_residual and it % config.eval_every == 0:
                step_gradients.append(
                    torch.cat([g.detach().cpu().reshape(-1) for g in grads])
                )

            updated_params = OrderedDict(
                (name, param - syn_lr * grad)
                for (name, param), grad in zip(param_map.items(), grads)
            )

            # Preserve the update semantics of the original notebooks.
            for name, param in param_map.items():
                param.data.copy_(updated_params[name].data)

        param_keys = list(param_map)
        student_flat = torch.cat([updated_params[k].reshape(-1) for k in param_keys])
        start_flat = torch.cat([starting_state[k].reshape(-1) for k in param_keys])
        target_flat = torch.cat([target_state[k].reshape(-1) for k in param_keys])

        if config.compute_residual and it % config.eval_every == 0 and step_gradients:
            teacher_displacement = target_flat - start_flat
            student_span = torch.stack(step_gradients, dim=1).to(device)
            span_similarity = compute_subspace_alignment(
                student_span, teacher_displacement
            )
            residual = compute_projection_residual(student_span, teacher_displacement)

        traj_loss = compute_trajectory_loss(student_flat, start_flat, target_flat)

        optimizer_x.zero_grad()
        optimizer_lr.zero_grad()
        if optimizer_y is not None:
            optimizer_y.zero_grad()

        traj_loss.backward()
        running_tm_loss += float(traj_loss.detach().cpu())

        if not torch.isfinite(traj_loss):
            print("Non-finite trajectory loss encountered; stopping.")
            break

        clip_tensors = [x_syn] + ([y_syn] if config.learn_y else [])
        torch.nn.utils.clip_grad_norm_(clip_tensors, max_norm=1.0)

        if float(traj_loss.detach()) <= config.update_loss_threshold:
            optimizer_x.step()
            optimizer_lr.step()
            if optimizer_y is not None:
                optimizer_y.step()

    # Ensure there is always a checkpoint, even if validation APR was NaN.
    torch.save({"x_syn": best_x_syn, "y_syn": best_y_syn}, save_path)
    return BTMResult(
        best_x_syn=best_x_syn,
        best_y_syn=best_y_syn,
        syn_lr=syn_lr.detach().cpu(),
        best_val_apr=best_score,
        history=history,
        save_path=save_path,
    )


def evaluate_synthetic_repeated(
    *,
    model_factory: Callable[[], nn.Module],
    x_syn: torch.Tensor,
    y_syn: torch.Tensor,
    val_loader: Iterable,
    test_loader: Iterable,
    task: str,
    device: str | torch.device,
    optim: str,
    lr: float,
    momentum: float,
    epochs: int,
    repeats: int = 10,
) -> dict[str, tuple[float, float]]:
    """Repeatedly train fresh networks on the synthetic set."""
    criterion = nn.BCEWithLogitsLoss()
    val_auc, val_apr, test_auc, test_apr = [], [], [], []

    for _ in range(repeats):
        net = model_factory().to(device)
        va, _, vp, net = train_and_evaluate(
            net,
            x_syn,
            y_syn,
            val_loader,
            device,
            criterion,
            task=task,
            optim=optim,
            lr=lr,
            momentum=momentum,
            epochs=epochs,
        )
        test = evaluate_model(net, test_loader, criterion, device, task)
        val_auc.append(va)
        val_apr.append(vp)
        test_auc.append(test["auc"])
        test_apr.append(test["apr"])

    def mean_std(values):
        return float(np.mean(values)), float(np.std(values))

    return {
        "val_auc": mean_std(val_auc),
        "val_apr": mean_std(val_apr),
        "test_auc": mean_std(test_auc),
        "test_apr": mean_std(test_apr),
    }
