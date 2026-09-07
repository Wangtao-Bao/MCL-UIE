from __future__ import annotations
import argparse
import json
import math
import random
import shutil
import time
from pathlib import Path
from typing import Dict, Iterable, Tuple
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from evaluation import (
    LPIPSMetric,
    full_reference_metrics,
    resize_for_evaluation,
    uciqe,
    uiqm,
)
from mcluie_model import MCLUIE, MCLUIEConfig, MCLUIELoss, LossWeights
from runtime import select_torch_device
from dataset import (
    PairedImageDataset,
    discover_pairs,
    seed_worker,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("Traindata/LSUI400/train"))
    parser.add_argument("--reference-dir", type=Path, default=Path("Traindata/LSUI400/trainGT"))
    parser.add_argument("--test-input-dir", type=Path, default=Path("Testdata/LSUI400/testimage"))
    parser.add_argument("--test-reference-dir", type=Path, default=Path("TestdatasetGT/LSUI400"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs1/MCL-UIE_LSUI400"))
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--blocks-per-stage", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="cuda")
    parser.add_argument("--lambda-ssim", type=float, default=0.2)
    parser.add_argument("--lambda-replay", type=float, default=0.1)
    parser.add_argument("--lambda-calibration", type=float, default=0.1)
    parser.add_argument("--resume", type=Path, default=None)
    return parser.parse_args()


def validate_arguments(args: argparse.Namespace) -> None:
    positive = (
        "epochs",
        "batch_size",
        "crop_size",
        "learning_rate",
        "weight_decay",
        "base_channels",
        "blocks_per_stage",
    )
    for name in positive:
        value = getattr(args, name)
        if value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    non_negative = ("lambda_ssim", "lambda_replay", "lambda_calibration")
    for name in non_negative:
        value = getattr(args, name)
        if value < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be non-negative")


def set_reproducibility(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_training_pairs(args: argparse.Namespace):
    training_pairs = discover_pairs(args.input_dir, args.reference_dir)
    test_pairs = discover_pairs(args.test_input_dir, args.test_reference_dir)
    training_inputs = {path.resolve() for path, _ in training_pairs}
    test_inputs = {path.resolve() for path, _ in test_pairs}
    overlap = training_inputs & test_inputs
    if overlap:
        preview = ", ".join(sorted(path.name for path in overlap)[:5])
        raise RuntimeError("Training and test-selection input directories overlap: " + preview)
    return training_pairs, test_pairs


def make_data_loaders(args: argparse.Namespace) -> Tuple[DataLoader, DataLoader]:
    train_pairs, test_pairs = resolve_training_pairs(args)
    train_dataset = PairedImageDataset(
        train_pairs,
        crop_size=args.crop_size,
        training=True,
    )
    test_dataset = PairedImageDataset(
        test_pairs,
        crop_size=None,
        training=False,
    )
    generator = torch.Generator().manual_seed(args.seed)
    common = {
        "num_workers": 0,
        "pin_memory": torch.cuda.is_available(),
        "worker_init_fn": seed_worker,
    }
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        drop_last=False,
        **common,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        **common,
    )
    print(
        f"Paired data: {len(train_dataset)} train / {len(test_dataset)} test-selection"
    )
    return train_loader, test_loader


def resize_to_max_side(
    image: torch.Tensor, target: torch.Tensor, maximum: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    height, width = image.shape[-2:]
    scale = min(1.0, maximum / max(height, width))
    if scale == 1.0:
        return image, target
    size = (max(8, round(height * scale)), max(8, round(width * scale)))
    return (
        F.interpolate(image, size=size, mode="bilinear", align_corners=False),
        F.interpolate(target, size=size, mode="bilinear", align_corners=False),
    )


def auxiliary_schedule(epoch: int, warmup_epochs: int, ramp_epochs: int) -> float:
    if epoch < warmup_epochs:
        return 0.0
    if ramp_epochs <= 0:
        return 1.0
    return min(1.0, (epoch - warmup_epochs + 1) / ramp_epochs)


def tensor_to_bgr_uint8(image: torch.Tensor) -> np.ndarray:
    if image.ndim == 4:
        if image.shape[0] != 1:
            raise ValueError("Selection metrics require batch size one")
        image = image[0]
    rgb = (
        image.detach()
        .float()
        .cpu()
        .clamp(0.0, 1.0)
        .permute(1, 2, 0)
        .numpy()
    )
    rgb = (rgb * 255.0).round().astype(np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


@torch.no_grad()
def evaluate_selection_set(
    model: MCLUIE,
    loader: Iterable[Dict[str, object]],
    device: torch.device,
    maximum_side: int,
    lpips_metric: LPIPSMetric,
) -> Dict[str, float]:
    model.eval()
    psnr_values, ssim_values, lpips_values = [], [], []
    uciqe_values, uiqm_values = [], []
    target_uciqe_values, target_uiqm_values = [], []
    for batch in loader:
        image = batch["input"].to(device)
        target = batch["target"].to(device)
        image, target = resize_to_max_side(image, target, maximum_side)
        prediction = model(image)["enhanced"]

        prediction_bgr = resize_for_evaluation(tensor_to_bgr_uint8(prediction))
        target_bgr = resize_for_evaluation(tensor_to_bgr_uint8(target))
        psnr, ssim = full_reference_metrics(target_bgr, prediction_bgr)
        psnr_values.append(psnr)
        ssim_values.append(ssim)
        lpips_values.append(lpips_metric(target_bgr, prediction_bgr))
        uciqe_values.append(uciqe(prediction_bgr))
        uiqm_values.append(uiqm(prediction_bgr))
        target_uciqe_values.append(uciqe(target_bgr))
        target_uiqm_values.append(uiqm(target_bgr))
    mean_uciqe = float(np.mean(uciqe_values))
    mean_uiqm = float(np.mean(uiqm_values))
    mean_target_uciqe = float(np.mean(target_uciqe_values))
    mean_target_uiqm = float(np.mean(target_uiqm_values))
    return {
        "psnr": float(np.mean(psnr_values)),
        "ssim": float(np.mean(ssim_values)),
        "lpips": float(np.mean(lpips_values)),
        "uciqe": mean_uciqe,
        "uiqm": mean_uiqm,
        "target_uciqe": mean_target_uciqe,
        "target_uiqm": mean_target_uiqm,
        "uciqe_gap": abs(mean_uciqe - mean_target_uciqe),
        "uiqm_gap": abs(mean_uiqm - mean_target_uiqm),
    }


def save_checkpoint(
    path: Path,
    model: MCLUIE,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    best_selection_score: float,
    best_psnr: float,
    args: argparse.Namespace,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "best_selection_score": best_selection_score,
            "best_psnr": best_psnr,
            "model_name": "MCL-UIE",
            "model_config": model.config.to_dict(),
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "training_arguments": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
        },
        path,
    )


def main() -> None:
    args = parse_arguments()
    validate_arguments(args)
    set_reproducibility(args.seed)
    device = select_torch_device(torch, args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "training_arguments.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )

    train_loader, selection_loader = make_data_loaders(args)
    config = MCLUIEConfig(
        base_channels=args.base_channels,
        blocks_per_stage=args.blocks_per_stage,
    )
    model = MCLUIE(config).to(device)
    selection_lpips = LPIPSMetric("alex", device.type)
    print(f"Test-selection LPIPS: {selection_lpips.description}")
    criterion = MCLUIELoss(
        LossWeights(
            ssim=args.lambda_ssim,
            replay=args.lambda_replay,
            calibration=args.lambda_calibration,
        )
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.epochs), eta_min=args.learning_rate * 0.01
    )
    start_epoch, best_selection_score, best_psnr = 0, -math.inf, -math.inf
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_selection_score = float(
            checkpoint.get(
                "best_selection_score", checkpoint.get("best_psnr", -math.inf)
            )
        )
        best_psnr = float(checkpoint.get("best_psnr", -math.inf))

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"Device: {device} | trainable parameters: {parameter_count:,}")
    history = []
    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_start = time.time()
        auxiliary_scale = auxiliary_schedule(epoch, 1, 2)
        running: Dict[str, float] = {}
        steps = 0
        for batch in train_loader:
            image = batch["input"].to(device)
            target = batch["target"].to(device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(image, target)
            loss, components = criterion(outputs, image, target, auxiliary_scale)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at epoch {epoch + 1}: {components}")
            if loss.detach().item() < -1e-8:
                raise FloatingPointError(
                    f"Negative loss at epoch {epoch + 1}: {components}"
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            for key, value in components.items():
                running[key] = running.get(key, 0.0) + value
            steps += 1
        scheduler.step()
        train_metrics = {key: value / steps for key, value in running.items()}
        selection_metrics = evaluate_selection_set(
            model,
            selection_loader,
            device,
            768,
            selection_lpips,
        )
        selection_score = (
            selection_metrics["psnr"]
            + 5.0 * selection_metrics["ssim"]
            - 0.05
            * (10.0 * selection_metrics["uciqe_gap"] + selection_metrics["uiqm_gap"])
        )
        elapsed = time.time() - epoch_start
        record = {
            "epoch": epoch + 1,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "auxiliary_scale": auxiliary_scale,
            "selection_score": selection_score,
            "seconds": elapsed,
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"test_selection_{key}": value for key, value in selection_metrics.items()},
        }
        history.append(record)
        print(
            f"Epoch {epoch + 1:03d}/{args.epochs:03d} | "
            f"loss {train_metrics['total']:.4f} | "
            f"test PSNR {selection_metrics['psnr']:.3f} | "
            f"test SSIM {selection_metrics['ssim']:.4f} | "
            f"test LPIPS {selection_metrics['lpips']:.4f} | "
            f"test UCIQE {selection_metrics['uciqe']:.4f} | "
            f"test UIQM {selection_metrics['uiqm']:.4f} | {elapsed:.1f}s"
        )
        next_best_psnr = max(best_psnr, selection_metrics["psnr"])
        next_best_selection_score = max(best_selection_score, selection_score)
        save_checkpoint(
            args.output_dir / "latest_checkpoint.pt",
            model,
            optimizer,
            scheduler,
            epoch,
            next_best_selection_score,
            next_best_psnr,
            args,
        )
        if selection_score > best_selection_score:
            best_selection_score = selection_score
            best_psnr = next_best_psnr
            best_history_dir = args.output_dir / "best_checkpoints"
            best_history_dir.mkdir(parents=True, exist_ok=True)
            archived_best_path = (
                best_history_dir / f"epoch_{epoch + 1:03d}_checkpoint.pt"
            )
            save_checkpoint(
                archived_best_path,
                model,
                optimizer,
                scheduler,
                epoch,
                best_selection_score,
                best_psnr,
                args,
            )
            shutil.copy2(
                archived_best_path,
                args.output_dir / "best_checkpoint.pt",
            )
            print(f"New best checkpoint saved: {archived_best_path}")
        else:
            best_psnr = next_best_psnr
        with (args.output_dir / "training_history.json").open("w", encoding="utf-8") as handle:
            json.dump(history, handle, indent=2)
    print(
        "Training completed. "
        f"Best test-selection PSNR: {best_psnr:.3f} dB | "
        f"best selection score: {best_selection_score:.4f}"
    )


if __name__ == "__main__":
    main()
