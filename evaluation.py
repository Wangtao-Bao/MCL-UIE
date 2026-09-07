"""Self-contained metric backend for MCL-UIE.

All images are uint8 BGR arrays resized to 256 x 256 with cv2.INTER_AREA.
Full-reference datasets use PSNR, SSIM, LPIPS, and FID. No-reference datasets
use UIQM and UCIQE.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy import ndimage
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from runtime import select_torch_device
from dataset import discover_pairs, list_images
EVALUATION_SIZE = (256, 256)
FULL_REFERENCE_PREFIXES = ("lsui", "ufo", "uieb", "euvp")


def resize_for_evaluation(image_bgr: np.ndarray) -> np.ndarray:
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("Cannot resize an empty image")
    if image_bgr.shape[:2] == EVALUATION_SIZE[::-1]:
        return image_bgr
    return cv2.resize(image_bgr, EVALUATION_SIZE, interpolation=cv2.INTER_AREA)


def full_reference_metrics(
    reference_bgr: np.ndarray, prediction_bgr: np.ndarray
) -> Tuple[float, float]:
    if reference_bgr.shape != prediction_bgr.shape:
        raise ValueError(
            f"Reference/prediction shape mismatch: "
            f"{reference_bgr.shape} vs {prediction_bgr.shape}"
        )
    psnr = peak_signal_noise_ratio(reference_bgr, prediction_bgr, data_range=255)
    ssim = structural_similarity(
        reference_bgr, prediction_bgr, channel_axis=2, data_range=255
    )
    return float(psnr), float(ssim)


def uciqe(image_bgr: np.ndarray) -> float:

    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    luminance = lab[..., 0] / 255
    channel_a = lab[..., 1] / 255
    channel_b = lab[..., 2] / 255
    chroma = np.sqrt(np.square(channel_a) + np.square(channel_b))
    saturation = chroma / np.sqrt(np.square(chroma) + np.square(luminance))
    mean_saturation = np.mean(saturation)
    mean_chroma = np.mean(chroma)
    chroma_variance = np.sqrt(
        np.mean(abs(1 - np.square(mean_chroma / (chroma + 1e-8))))
    )

    bins = 256 if luminance.dtype == "uint8" else 65536
    histogram, _ = np.histogram(luminance, bins)
    cdf = np.cumsum(histogram) / np.sum(histogram)
    low = np.where(cdf > 0.01)[0]
    high = np.where(cdf >= 0.99)[0]
    contrast = 0.5 if len(low) == 0 or len(high) == 0 else (
        high[0] - low[0]
    ) / (bins - 1)
    return float(
        0.4680 * chroma_variance
        + 0.2745 * contrast
        + 0.2576 * mean_saturation
    )


def _trimmed_mean(values: np.ndarray, left: float = 0.1, right: float = 0.1) -> float:
    values = sorted(values)
    count = len(values)
    left_count = math.ceil(left * count)
    right_count = math.floor(right * count)
    weight = 1 / (count - left_count - right_count)
    return weight * sum(values[left_count + 1 : count - right_count])


def _variance(values: np.ndarray, mean: float) -> float:
    return sum(math.pow(value - mean, 2) for value in values) / len(values)


def _uicm(image_rgb: np.ndarray) -> float:
    red = image_rgb[:, :, 0].flatten()
    green = image_rgb[:, :, 1].flatten()
    blue = image_rgb[:, :, 2].flatten()
    red_green = red - green
    yellow_blue = (red + green) / 2 - blue
    mean_rg = _trimmed_mean(red_green)
    mean_yb = _trimmed_mean(yellow_blue)
    spread_rg = _variance(red_green, mean_rg)
    spread_yb = _variance(yellow_blue, mean_yb)
    color_mean = math.sqrt(mean_rg**2 + mean_yb**2)
    color_spread = math.sqrt(spread_rg + spread_yb)
    return -0.0268 * color_mean + 0.1586 * color_spread


def _sobel(channel: np.ndarray) -> np.ndarray:
    dx = ndimage.sobel(channel, 0)
    dy = ndimage.sobel(channel, 1)
    magnitude = np.hypot(dx, dy)
    magnitude *= 255.0 / np.max(magnitude)
    return magnitude


def _eme(channel: np.ndarray, window: int) -> float:
    column_blocks = channel.shape[1] / window
    row_blocks = channel.shape[0] / window
    weight = 2.0 / (column_blocks * row_blocks)
    columns = int(column_blocks)
    rows = int(row_blocks)
    channel = channel[: rows * window, : columns * window]
    value = 0.0
    for column in range(columns):
        for row in range(rows):
            block = channel[
                row * window : (row + 1) * window,
                column * window : (column + 1) * window,
            ]
            maximum = np.max(block)
            minimum = np.min(block)
            if minimum != 0.0 and maximum != 0.0:
                value += math.log(maximum / minimum)
    return weight * value


def _uism(image_rgb: np.ndarray) -> float:
    red, green, blue = (image_rgb[:, :, index] for index in range(3))
    red_eme = _eme(_sobel(red) * red, 10)
    green_eme = _eme(_sobel(green) * green, 10)
    blue_eme = _eme(_sobel(blue) * blue, 10)
    return 0.299 * red_eme + 0.587 * green_eme + 0.144 * blue_eme


def _uiconm(image_rgb: np.ndarray, window: int = 10) -> float:
    column_blocks = image_rgb.shape[1] / window
    row_blocks = image_rgb.shape[0] / window
    weight = -1.0 / (column_blocks * row_blocks)
    columns = int(column_blocks)
    rows = int(row_blocks)
    image_rgb = image_rgb[: rows * window, : columns * window]
    value = 0.0
    for column in range(columns):
        for row in range(rows):
            block = image_rgb[
                row * window : (row + 1) * window,
                column * window : (column + 1) * window,
                :,
            ]
            maximum = np.max(block)
            minimum = np.min(block)
            numerator = maximum - minimum
            denominator = maximum + minimum
            if (
                not math.isnan(numerator)
                and not math.isnan(denominator)
                and numerator != 0.0
                and denominator != 0.0
            ):
                ratio = numerator / denominator
                value += ratio * math.log(ratio)
    return weight * value


def uiqm(image_bgr: np.ndarray) -> float:
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    return float(
        0.0282 * _uicm(image_rgb)
        + 0.2953 * _uism(image_rgb)
        + 3.5753 * _uiconm(image_rgb)
    )


class LPIPSMetric:

    def __init__(self, network: str = "alex", device: str = "auto") -> None:
        try:
            import lpips
            import torch
        except ImportError as error:
            raise RuntimeError(
                "LPIPS is required for paired evaluation. Install it with "
                "`python -m pip install lpips==0.1.4`."
            ) from error
        self.torch = torch
        self.device = select_torch_device(torch, device)
        self.network = network
        self.model = lpips.LPIPS(net=network, version="0.1").to(self.device).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def _to_tensor(self, image_bgr: np.ndarray):
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_rgb = np.ascontiguousarray(image_rgb.transpose(2, 0, 1))
        tensor = self.torch.from_numpy(image_rgb).to(
            device=self.device, dtype=self.torch.float32
        )
        return tensor.unsqueeze(0).div(127.5).sub(1.0)

    def __call__(
        self, reference_bgr: np.ndarray, prediction_bgr: np.ndarray
    ) -> float:
        if reference_bgr.shape != prediction_bgr.shape:
            raise ValueError("LPIPS requires reference and prediction of equal size")
        reference = self._to_tensor(reference_bgr)
        prediction = self._to_tensor(prediction_bgr)
        with self.torch.inference_mode():
            distance = self.model(reference, prediction, normalize=False)
        return float(distance.reshape(-1).mean().cpu().item())

    @property
    def description(self) -> str:
        return f"official LPIPS v0.1, net={self.network}, device={self.device}"


class FIDMetric:

    def __init__(self, device: str = "auto", batch_size: int = 32) -> None:
        if batch_size < 1:
            raise ValueError("--fid-batch-size must be positive")
        try:
            import torch
            from cleanfid import fid
        except ImportError as error:
            raise RuntimeError(
                "FID requires clean-fid. Install it with "
                "`python -m pip install clean-fid==0.1.35`."
            ) from error
        self.fid = fid
        self.device = select_torch_device(torch, device)
        self.batch_size = batch_size
        self.mode = "clean"
        self.model = fid.build_feature_extractor(
            self.mode,
            self.device,
            use_dataparallel=False,
        )

    def __call__(self, prediction_dir: Path, reference_dir: Path) -> float:
        prediction_features = self.fid.get_folder_features(
            str(prediction_dir),
            model=self.model,
            num_workers=0,
            batch_size=self.batch_size,
            device=self.device,
            mode=self.mode,
            description="FID prediction: ",
            verbose=False,
        )
        reference_features = self.fid.get_folder_features(
            str(reference_dir),
            model=self.model,
            num_workers=0,
            batch_size=self.batch_size,
            device=self.device,
            mode=self.mode,
            description="FID GT: ",
            verbose=False,
        )
        if len(prediction_features) != len(reference_features):
            raise ValueError(
                "FID feature count mismatch: "
                f"{len(prediction_features)} prediction vs {len(reference_features)} GT"
            )
        if len(prediction_features) < 2:
            raise ValueError("FID needs at least two matched image pairs")
        return float(
            self.fid.frechet_distance(
                np.mean(prediction_features, axis=0),
                np.cov(prediction_features, rowvar=False),
                np.mean(reference_features, axis=0),
                np.cov(reference_features, rowvar=False),
            )
        )

    @property
    def description(self) -> str:
        return (
            "clean-fid 0.1.35, mode=clean, model=inception_v3, "
            f"device={self.device}, batch_size={self.batch_size}"
        )


def dataset_name_from_path(path: Path) -> str:
    path = Path(path)
    return path.parent.name if path.name.lower() == "testimage" else path.name


def uses_full_reference_metrics(dataset_name: str) -> bool:
    normalized = dataset_name.lower().replace("-", "").replace("_", "")
    if "challenging" in normalized:
        return False
    return (
        normalized in FULL_REFERENCE_PREFIXES
        or normalized.startswith(("lsui", "ufo", "uieb90", "euvp"))
    )


def _read_bgr(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return resize_for_evaluation(image)


def _mean(records: List[Dict[str, object]], key: str) -> Optional[float]:
    values = [record[key] for record in records if record.get(key) is not None]
    if not values:
        return None
    return float(np.mean(values))


def _write_metric_files(
    output_json: Path,
    output_csv: Path,
    summary: Dict[str, object],
    records: List[Dict[str, object]],
    fields: Tuple[str, ...],
) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as handle:
        json.dump({"summary": summary, "records": records}, handle, indent=2)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def _print_metric_summary(summary: Dict[str, object]) -> None:
    metrics = summary["metrics"]
    compact_metrics = ", ".join(
        f"{key}={value:.4f}"
        for key, value in metrics.items()
        if value is not None
    )
    print(f"Metrics completed: {summary['images']} images | {compact_metrics}")
    print(f"Saved: {summary['output_json']}")
    print(f"Saved: {summary['output_csv']}")


def evaluate_full_reference_directory(
    prediction_dir: Path,
    reference_dir: Path,
    *,
    dataset_name: str,
    output_json: Optional[Path] = None,
    output_csv: Optional[Path] = None,
    lpips_net: str = "alex",
    lpips_device: str = "auto",
    fid_device: str = "auto",
    fid_batch_size: int = 32,
    max_images: Optional[int] = None,
    no_lpips: bool = False,
    no_fid: bool = False,
) -> Dict[str, object]:
    prediction_dir = Path(prediction_dir).expanduser()
    reference_dir = Path(reference_dir).expanduser()
    pairs = discover_pairs(prediction_dir, reference_dir)
    if max_images:
        pairs = pairs[:max_images]
    if not pairs:
        raise RuntimeError("No prediction images were selected")

    lpips_metric = None if no_lpips else LPIPSMetric(lpips_net, lpips_device)
    fid_metric = None if no_fid else FIDMetric(fid_device, fid_batch_size)
    if lpips_metric is not None:
        print(f"LPIPS configuration: {lpips_metric.description}")
    if fid_metric is not None:
        print(f"FID configuration: {fid_metric.description}")

    records: List[Dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix=f"mcluie_fid_{dataset_name}_") as temporary:
        fid_prediction_dir = Path(temporary) / "prediction"
        fid_reference_dir = Path(temporary) / "reference"
        fid_prediction_dir.mkdir()
        fid_reference_dir.mkdir()

        for index, (prediction_path, reference_path) in enumerate(pairs, start=1):
            prediction_bgr = _read_bgr(prediction_path)
            reference_bgr = _read_bgr(reference_path)
            psnr, ssim = full_reference_metrics(reference_bgr, prediction_bgr)
            record: Dict[str, object] = {
                "image": prediction_path.name,
                "prediction": str(prediction_path),
                "reference": str(reference_path),
                "PSNR": psnr,
                "SSIM": ssim,
                "LPIPS": (
                    lpips_metric(reference_bgr, prediction_bgr)
                    if lpips_metric is not None
                    else None
                ),
            }
            records.append(record)
            cv2.imwrite(str(fid_prediction_dir / f"{index:06d}.png"), prediction_bgr)
            cv2.imwrite(str(fid_reference_dir / f"{index:06d}.png"), reference_bgr)
            print(f"[metrics {index:04d}/{len(pairs):04d}] {prediction_path.name}")

        fid_value = None
        if fid_metric is not None:
            fid_value = fid_metric(fid_prediction_dir, fid_reference_dir)

    output_json = output_json or prediction_dir / "metrics_summary.json"
    output_csv = output_csv or prediction_dir / "metrics_records.csv"
    metrics = {
        "PSNR": _mean(records, "PSNR"),
        "SSIM": _mean(records, "SSIM"),
        "LPIPS": _mean(records, "LPIPS"),
        "FID": fid_value,
    }
    summary = {
        "mode": "full-reference",
        "dataset": dataset_name,
        "prediction_dir": str(prediction_dir),
        "reference_dir": str(reference_dir),
        "images": len(records),
        "metrics": metrics,
        "output_json": str(output_json),
        "output_csv": str(output_csv),
    }
    _write_metric_files(
        output_json,
        output_csv,
        summary,
        records,
        ("image", "prediction", "reference", "PSNR", "SSIM", "LPIPS"),
    )
    _print_metric_summary(summary)
    return summary


def evaluate_no_reference_directory(
    prediction_dir: Path,
    *,
    dataset_name: str,
    output_json: Optional[Path] = None,
    output_csv: Optional[Path] = None,
    max_images: Optional[int] = None,
) -> Dict[str, object]:
    prediction_dir = Path(prediction_dir).expanduser()
    image_paths = list_images(prediction_dir)
    if max_images:
        image_paths = image_paths[:max_images]
    if not image_paths:
        raise RuntimeError("No prediction images were selected")

    records: List[Dict[str, object]] = []
    for index, image_path in enumerate(image_paths, start=1):
        image_bgr = _read_bgr(image_path)
        record = {
            "image": image_path.name,
            "prediction": str(image_path),
            "UIQM": uiqm(image_bgr),
            "UCIQE": uciqe(image_bgr),
        }
        records.append(record)
        print(f"[metrics {index:04d}/{len(image_paths):04d}] {image_path.name}")

    output_json = output_json or prediction_dir / "metrics_summary.json"
    output_csv = output_csv or prediction_dir / "metrics_records.csv"
    summary = {
        "mode": "no-reference",
        "dataset": dataset_name,
        "prediction_dir": str(prediction_dir),
        "reference_dir": None,
        "images": len(records),
        "metrics": {
            "UIQM": _mean(records, "UIQM"),
            "UCIQE": _mean(records, "UCIQE"),
        },
        "output_json": str(output_json),
        "output_csv": str(output_csv),
    }
    _write_metric_files(
        output_json,
        output_csv,
        summary,
        records,
        ("image", "prediction", "UIQM", "UCIQE"),
    )
    _print_metric_summary(summary)
    return summary


def evaluate_directory(
    prediction_dir: Path,
    *,
    dataset_name: Optional[str] = None,
    reference_dir: Optional[Path] = None,
    metrics: str = "auto",
    output_json: Optional[Path] = None,
    output_csv: Optional[Path] = None,
    lpips_net: str = "alex",
    lpips_device: str = "auto",
    fid_device: str = "auto",
    fid_batch_size: int = 32,
    max_images: Optional[int] = None,
    no_lpips: bool = False,
    no_fid: bool = False,
) -> Dict[str, object]:
    dataset = dataset_name or dataset_name_from_path(Path(prediction_dir))
    mode = metrics
    if mode == "auto":
        mode = "full-reference" if uses_full_reference_metrics(dataset) else "no-reference"
    if mode == "full-reference":
        if reference_dir is None:
            raise ValueError(
                f"Dataset {dataset} uses PSNR/SSIM/LPIPS/FID and requires --reference-dir"
            )
        return evaluate_full_reference_directory(
            prediction_dir,
            reference_dir,
            dataset_name=dataset,
            output_json=output_json,
            output_csv=output_csv,
            lpips_net=lpips_net,
            lpips_device=lpips_device,
            fid_device=fid_device,
            fid_batch_size=fid_batch_size,
            max_images=max_images,
            no_lpips=no_lpips,
            no_fid=no_fid,
        )
    if mode == "no-reference":
        return evaluate_no_reference_directory(
            prediction_dir,
            dataset_name=dataset,
            output_json=output_json,
            output_csv=output_csv,
            max_images=max_images,
        )
    raise ValueError(f"Unknown metrics mode: {metrics}")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate MCL-UIE outputs with the built-in metric backend."
    )
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--reference-dir", type=Path, default=None)
    parser.add_argument(
        "--metrics",
        choices=("auto", "full-reference", "no-reference"),
        default="auto",
    )
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--lpips-net", choices=("alex", "vgg", "squeeze"), default="alex")
    parser.add_argument("--lpips-device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--fid-device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--fid-batch-size", type=int, default=32)
    parser.add_argument("--no-lpips", action="store_true")
    parser.add_argument("--no-fid", action="store_true")
    parser.add_argument("--max-images", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    evaluate_directory(
        args.prediction_dir,
        dataset_name=args.dataset_name,
        reference_dir=args.reference_dir,
        metrics=args.metrics,
        output_json=args.output_json,
        output_csv=args.output_csv,
        lpips_net=args.lpips_net,
        lpips_device=args.lpips_device,
        fid_device=args.fid_device,
        fid_batch_size=args.fid_batch_size,
        max_images=args.max_images,
        no_lpips=args.no_lpips,
        no_fid=args.no_fid,
    )


if __name__ == "__main__":
    main()
