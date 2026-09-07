from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
from typing import List
import torch
import torch.nn.functional as F
from PIL import Image
from evaluation import dataset_name_from_path, evaluate_directory
from mcluie_model import MCLUIE, MCLUIEConfig
from runtime import select_torch_device
from dataset import image_to_tensor, list_images, tensor_to_image


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/MCL-UIE_LSUI400/lsui_checkpoint.pt"))
    parser.add_argument("--input-dir", type=Path, default=Path("Testdata/LSUI400/testimage"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/LSUI400"))
    parser.add_argument("--dataset-name", default="LSUI400")
    parser.add_argument("--reference-dir", type=Path, default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="cuda")
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--max-side",type=int,default=0)
    parser.add_argument("--restore-original-size",action=argparse.BooleanOptionalAction,default=True,)
    parser.add_argument("--metrics",choices=("auto", "full-reference", "no-reference", "none"),
        default="no-reference",
        help="Use no-reference for datasets without GT; then --reference-dir is not needed.",
    )
    return parser.parse_args()


def resolve_input_paths(input_dir: Path, max_images: int | None) -> List[Path]:
    paths = list_images(input_dir)
    if max_images:
        paths = paths[:max_images]
    if not paths:
        raise RuntimeError("No input images were selected")
    return paths


def resize_for_inference(image: torch.Tensor, max_side: int) -> torch.Tensor:
    if max_side <= 0 or max(image.shape[-2:]) <= max_side:
        return image
    height, width = image.shape[-2:]
    scale = max_side / max(height, width)
    size = (max(8, round(height * scale)), max(8, round(width * scale)))
    return F.interpolate(image, size=size, mode="bilinear", align_corners=False)


def load_model(checkpoint_path: Path, device: torch.device) -> MCLUIE:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "model_config" not in checkpoint or "model_state" not in checkpoint:
        raise RuntimeError(f"Invalid MCL-UIE checkpoint: {checkpoint_path}")
    model_config = dict(checkpoint["model_config"])
    model_config.pop("condition_limit", None)
    model = MCLUIE(MCLUIEConfig(**model_config))
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.to(device).eval()
    return model


def main() -> None:
    args = parse_arguments()
    dataset_name = args.dataset_name or dataset_name_from_path(args.input_dir)
    device = select_torch_device(torch, args.device)
    model = load_model(args.checkpoint, device)
    input_paths = resolve_input_paths(args.input_dir, args.max_images)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    timings = []
    records = []
    with torch.inference_mode():
        for index, path in enumerate(input_paths, start=1):
            with Image.open(path) as handle:
                original = handle.convert("RGB")
            image = image_to_tensor(original).unsqueeze(0).to(device)
            original_size = image.shape[-2:]
            network_input = resize_for_inference(image, args.max_side)
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            prediction = model(network_input)["enhanced"]
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            if args.restore_original_size and prediction.shape[-2:] != original_size:
                prediction = F.interpolate(
                    prediction, size=original_size, mode="bilinear", align_corners=False
                )
            output_path = args.output_dir / f"{path.stem}.png"
            tensor_to_image(prediction).save(output_path)
            timings.append(elapsed)
            records.append(
                {
                    "input": str(path),
                    "output": str(output_path),
                    "original_size": list(original_size),
                    "network_size": list(network_input.shape[-2:]),
                    "seconds": elapsed,
                }
            )
            print(f"[{index:04d}/{len(input_paths):04d}] {path.name} -> {output_path.name}")
    summary = {
        "checkpoint": str(args.checkpoint),
        "dataset": dataset_name,
        "device": str(device),
        "images": len(records),
        "mean_seconds": sum(timings) / len(timings),
        "records": records,
    }
    if args.metrics != "none":
        metric_summary = evaluate_directory(
            args.output_dir,
            dataset_name=dataset_name,
            reference_dir=args.reference_dir,
            metrics=args.metrics,
            lpips_net="alex",
            lpips_device=device.type,
            fid_device=device.type,
            fid_batch_size=32,
            max_images=args.max_images,
            no_lpips=False,
            no_fid=False,
        )
        summary["metrics"] = metric_summary["metrics"]
        summary["metrics_mode"] = metric_summary["mode"]
    with (args.output_dir / "inference_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(
        f"Inference completed: {len(records)} images, "
        f"mean {summary['mean_seconds']:.4f} s/image"
    )


if __name__ == "__main__":
    main()
