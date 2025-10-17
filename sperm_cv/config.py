from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Dict, Sequence, Tuple


if TYPE_CHECKING:  # pragma: no cover - type hints only
    from torchvision import transforms


TransformFactory = Callable[[int], 'transforms.Compose']


@dataclass(frozen=True)
class ExperimentConfig:
    """Configuration describing dataset-specific defaults and behaviour."""

    name: str
    num_classes: int
    default_data_path: str
    default_save_dir: str
    default_epochs: int
    image_extensions: Tuple[str, ...]
    ignore_hidden: bool
    train_transform: TransformFactory
    eval_transform: TransformFactory
    model_specs: Sequence[Tuple[str, Dict[str, object]]]
    default_batch_size: int = 16
    default_kfolds: int = 5
    oversample_seed: int = 2024


def _smids_train_transform(img_size: int) -> 'transforms.Compose':
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize((img_size + 32, img_size + 32)),
        transforms.RandomCrop(img_size),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.3),
        transforms.RandomApply([
            transforms.RandomPerspective(distortion_scale=0.08),
        ], p=0.25),
        transforms.RandomRotation(20),
        transforms.RandomAutocontrast(p=0.3),
        transforms.RandomAdjustSharpness(sharpness_factor=1.6, p=0.35),
        transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.18, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)),
    ])


def _smids_eval_transform(img_size: int) -> 'transforms.Compose':
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def _mhsma_train_transform(img_size: int) -> 'transforms.Compose':
    from torchvision import transforms
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((img_size + 64, img_size + 64)),
        transforms.RandomAffine(
            degrees=12,
            translate=(0.04, 0.04),
            scale=(0.92, 1.08),
            shear=(-6, 6, -3, 3),
        ),
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        ], p=0.3),
        transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.4),
        transforms.RandomEqualize(p=0.35),
        transforms.RandomCrop(img_size),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.3),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.28, contrast=0.32, saturation=0.08, hue=0.015),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.2, 0.2, 0.2]),
        transforms.RandomErasing(p=0.1, scale=(0.01, 0.05), ratio=(0.3, 3.3)),
    ])


def _mhsma_eval_transform(img_size: int) -> 'transforms.Compose':
    from torchvision import transforms
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.2, 0.2, 0.2]),
    ])


SMIDS_CONFIG = ExperimentConfig(
    name="SMIDS",
    num_classes=3,
    default_data_path="SMIDS-others",
    default_save_dir="result_SMIDS_160",
    default_epochs=160,
    image_extensions=(".bmp",),
    ignore_hidden=False,
    train_transform=_smids_train_transform,
    eval_transform=_smids_eval_transform,
    model_specs=[
        ("MADRNet_Full", dict(use_rev_blocks=True, use_bilinear=True)),
    ],
)


MHSMA_CONFIG = ExperimentConfig(
    name="MHSMA",
    num_classes=2,
    default_data_path="mhsma/vacuole",
    default_save_dir="result_mhsma/vacuole",
    default_epochs=100,
    image_extensions=(".jpg",),
    ignore_hidden=True,
    train_transform=_mhsma_train_transform,
    eval_transform=_mhsma_eval_transform,
    model_specs=[
        ("MADRNet_Full", dict(use_rev_blocks=True, use_bilinear=False)),
    ],
)
