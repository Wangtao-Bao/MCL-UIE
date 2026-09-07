"""Dataset discovery, aligned augmentation, and image I/O for MCL-UIE."""

from __future__ import annotations

import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
ImagePair = Tuple[Path, Path]


def list_images(directory: Path) -> List[Path]:
    directory = Path(directory).expanduser()
    if not directory.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {directory}")
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def normalized_image_key(path: Path) -> str:
    key = re.sub(r"[^a-z0-9]+", "_", path.stem.lower()).strip("_")
    removable = ("_reference", "_target", "_groundtruth", "_ground_truth", "_gt", "_raw", "_input", "_img")
    changed = True
    while changed:
        changed = False
        for suffix in removable:
            if key.endswith(suffix):
                key = key[: -len(suffix)].rstrip("_")
                changed = True
                break
    return key


def discover_pairs(input_dir: Path, reference_dir: Path) -> List[ImagePair]:
    inputs = list_images(input_dir)
    references = list_images(reference_dir)
    exact_reference_map = {path.name.lower(): path for path in references}
    normalized_reference_map: Dict[str, List[Path]] = {}
    for path in references:
        key = normalized_image_key(path)
        normalized_reference_map.setdefault(key, []).append(path)
    pairs = []
    missing = []
    for input_path in inputs:
        key = normalized_image_key(input_path)
        exact_match = exact_reference_map.get(input_path.name.lower())
        fallback_matches = normalized_reference_map.get(key, [])
        if exact_match is not None:
            pairs.append((input_path, exact_match))
        elif len(fallback_matches) == 1:
            pairs.append((input_path, fallback_matches[0]))
        else:
            missing.append(input_path.name)
    if missing:
        preview = ", ".join(missing[:5])
        raise RuntimeError(
            f"Found {len(missing)} inputs without references in {input_dir}: {preview}"
        )
    if not pairs:
        raise RuntimeError(f"No matched image pairs found in {input_dir} and {reference_dir}")
    return pairs


def image_to_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    tensor = tensor.detach().float().cpu().clamp(0.0, 1.0)
    if tensor.ndim == 4:
        if tensor.shape[0] != 1:
            raise ValueError("A batched tensor must contain exactly one image")
        tensor = tensor[0]
    array = (tensor.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(array, mode="RGB")


def _resize_pair_for_crop(
    image: Image.Image, reference: Image.Image, crop_size: int
) -> Tuple[Image.Image, Image.Image]:
    if image.size != reference.size:
        reference = reference.resize(image.size, Image.Resampling.BICUBIC)
    width, height = image.size
    if min(width, height) >= crop_size:
        return image, reference
    scale = crop_size / min(width, height)
    resized = (max(crop_size, round(width * scale)), max(crop_size, round(height * scale)))
    return (
        image.resize(resized, Image.Resampling.BICUBIC),
        reference.resize(resized, Image.Resampling.BICUBIC),
    )


def _aligned_random_augmentation(
    image: Image.Image, reference: Image.Image, crop_size: int
) -> Tuple[Image.Image, Image.Image]:
    image, reference = _resize_pair_for_crop(image, reference, crop_size)
    width, height = image.size
    left = random.randint(0, width - crop_size)
    top = random.randint(0, height - crop_size)
    box = (left, top, left + crop_size, top + crop_size)
    image, reference = image.crop(box), reference.crop(box)
    if random.random() < 0.5:
        image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        reference = reference.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    if random.random() < 0.5:
        image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        reference = reference.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    rotation = random.randint(0, 3)
    if rotation:
        transpose = {
            1: Image.Transpose.ROTATE_90,
            2: Image.Transpose.ROTATE_180,
            3: Image.Transpose.ROTATE_270,
        }[rotation]
        image = image.transpose(transpose)
        reference = reference.transpose(transpose)
    return image, reference


class PairedImageDataset(Dataset):
    def __init__(
        self,
        pairs: Sequence[ImagePair],
        crop_size: Optional[int] = 256,
        training: bool = True,
        max_samples: Optional[int] = None,
    ) -> None:
        self.pairs = list(pairs[:max_samples] if max_samples else pairs)
        self.crop_size = crop_size
        self.training = training
        if not self.pairs:
            raise ValueError("PairedImageDataset received no image pairs")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> Dict[str, object]:
        input_path, reference_path = self.pairs[index]
        with Image.open(input_path) as handle:
            image = handle.convert("RGB")
        with Image.open(reference_path) as handle:
            reference = handle.convert("RGB")
        if self.training and self.crop_size:
            image, reference = _aligned_random_augmentation(
                image, reference, self.crop_size
            )
        elif image.size != reference.size:
            reference = reference.resize(image.size, Image.Resampling.BICUBIC)
        return {
            "input": image_to_tensor(image),
            "target": image_to_tensor(reference),
            "name": input_path.stem,
            "input_path": str(input_path),
            "reference_path": str(reference_path),
        }


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)
