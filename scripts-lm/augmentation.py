#!/usr/bin/env python3
"""
CLI script to augment the raw training data with a single augmentation strategy.

# Exapmle Usage
## Rotation
python scripts-lm/augmentation.py \
  --strategy rotation \
  --output data/augmented/train-rotation.pkl

## Scaling
python scripts-lm/augmentation.py \
  --strategy scaling \
  --output data/augmented/train-scaling.pkl

## Sheering
python scripts-lm/augmentation.py \
  --strategy sheering \
  --output data/augmented/train-sheering.pkl

## Translation
python scripts-lm/augmentation.py \
  --strategy translation \
  --output data/augmented/train-translation.pkl

## Deformation (elastic)
python scripts-lm/augmentation.py \
  --strategy deformation \
  --output data/augmented/train-deformation.pkl

--------------------------------

It:
- loads `data/raw/train.pkl` (gzipped pickle, same format as in the notebooks),
- applies ONE of the 5 augmentation strategies to each sample,
- creates N augmented versions per original sample (default: 1),
- writes only the augmented samples (without the original ones) to a gzipped pickle file at the given output path.

Augmentation strategies (single-choice via CLI):
- rotation
- scaling
- sheering
- translation
- deformation  (elastic deformation on CPU)

For every video:
- the SAME type of transform is applied to all frames
- the parameters (angle, zoom factor, etc.) are randomized per video

Reproducibility:
- Seeds are set for `random`, `numpy` and `torch` (incl. CUDA) from a CLI argument.

GPU usage:
- If a CUDA GPU is available, affine transforms run on GPU via PyTorch + Kornia.
- Elastic deformation runs on CPU using `elasticdeform`.
"""

import argparse
import gzip
import pickle
import random
from pathlib import Path
from typing import Dict, Any, List

import numpy as np

import torch
import kornia.geometry.transform as KT
from tqdm import tqdm

import elasticdeform


# ------------------------------
# Reproducibility utilities
# ------------------------------

def set_global_seed(seed: int) -> None:
    """Set seeds for Python, NumPy and PyTorch (CPU & CUDA)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ------------------------------
# IO helpers (compatible with notebooks)
# ------------------------------

def load_zipped_pickle(filename: Path) -> Any:
    """Load a gzipped pickle file."""
    with gzip.open(str(filename), "rb") as f:
        return pickle.load(f)


def save_zipped_pickle(obj: Any, filename: Path) -> None:
    """Save an object to a gzipped pickle file."""
    filename.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(str(filename), "wb") as f:
        pickle.dump(obj, f, protocol=2)


# ------------------------------
# GPU-based affine augmentations (adapted from notebook)
# ------------------------------

def _ensure_4d(t: torch.Tensor) -> torch.Tensor:
    if t.dim() == 3:
        return t.unsqueeze(1)
    return t


def apply_rotation_gpu(
    image_tensor: torch.Tensor,
    mask_tensor: torch.Tensor,
    angle_range=(-15, 15),
    device: torch.device = torch.device("cuda"),
) -> (torch.Tensor, torch.Tensor):
    image_tensor = _ensure_4d(image_tensor)
    mask_tensor = _ensure_4d(mask_tensor)

    B, C, H, W = image_tensor.shape
    angle = random.uniform(angle_range[0], angle_range[1])

    center = torch.tensor([[W / 2, H / 2]], device=device, dtype=torch.float32).expand(B, -1)
    angle_tensor = torch.tensor([angle], device=device, dtype=torch.float32).expand(B)
    # Kornia expects scale as (B, 2) tensor
    scale = torch.ones(B, 2, device=device, dtype=torch.float32)

    M = KT.get_rotation_matrix2d(center, angle_tensor, scale)

    img_rot = KT.warp_affine(
        image_tensor,
        M,
        dsize=(H, W),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )
    img_rot = torch.clamp(img_rot, 0, 255)

    mask_rot = KT.warp_affine(
        mask_tensor.float(),
        M,
        dsize=(H, W),
        mode="nearest",
        padding_mode="zeros",
        align_corners=False,
    )
    mask_rot = (mask_rot > 0.5).float()

    return img_rot, mask_rot


def apply_zooming_gpu(
    image_tensor: torch.Tensor,
    mask_tensor: torch.Tensor,
    zoom_range=(0.9, 1.1),
    device: torch.device = torch.device("cuda"),
) -> (torch.Tensor, torch.Tensor):
    image_tensor = _ensure_4d(image_tensor)
    mask_tensor = _ensure_4d(mask_tensor)

    B, C, H, W = image_tensor.shape
    zoom_factor = random.uniform(zoom_range[0], zoom_range[1])

    center = torch.tensor([[W / 2, H / 2]], device=device, dtype=torch.float32).expand(B, -1)
    angle = torch.zeros(B, device=device, dtype=torch.float32)
    # Isotropic scaling: same factor in x and y -> (B, 2)
    scale = torch.full((B, 2), zoom_factor, device=device, dtype=torch.float32)

    M = KT.get_rotation_matrix2d(center, angle, scale)

    img_zoom = KT.warp_affine(
        image_tensor,
        M,
        dsize=(H, W),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )
    img_zoom = torch.clamp(img_zoom, 0, 255)

    mask_zoom = KT.warp_affine(
        mask_tensor.float(),
        M,
        dsize=(H, W),
        mode="nearest",
        padding_mode="zeros",
        align_corners=False,
    )
    mask_zoom = (mask_zoom > 0.5).float()

    return img_zoom, mask_zoom


def apply_sheering_gpu(
    image_tensor: torch.Tensor,
    mask_tensor: torch.Tensor,
    shear_range=(-0.2, 0.2),
    device: torch.device = torch.device("cuda"),
) -> (torch.Tensor, torch.Tensor):
    image_tensor = _ensure_4d(image_tensor)
    mask_tensor = _ensure_4d(mask_tensor)

    B, C, H, W = image_tensor.shape
    shear = random.uniform(shear_range[0], shear_range[1])

    if random.random() > 0.5:
        M_single = torch.tensor(
            [[1.0, shear, 0.0], [0.0, 1.0, 0.0]],
            device=device,
            dtype=torch.float32,
        )
    else:
        M_single = torch.tensor(
            [[1.0, 0.0, 0.0], [shear, 1.0, 0.0]],
            device=device,
            dtype=torch.float32,
        )

    M = M_single.unsqueeze(0).expand(B, -1, -1)

    img_shear = KT.warp_affine(
        image_tensor,
        M,
        dsize=(H, W),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )
    img_shear = torch.clamp(img_shear, 0, 255)

    mask_shear = KT.warp_affine(
        mask_tensor.float(),
        M,
        dsize=(H, W),
        mode="nearest",
        padding_mode="zeros",
        align_corners=False,
    )
    mask_shear = (mask_shear > 0.5).float()

    return img_shear, mask_shear


def apply_translation_gpu(
    image_tensor: torch.Tensor,
    mask_tensor: torch.Tensor,
    translate_range=(-10, 10),
    device: torch.device = torch.device("cuda"),
) -> (torch.Tensor, torch.Tensor):
    image_tensor = _ensure_4d(image_tensor)
    mask_tensor = _ensure_4d(mask_tensor)

    B, C, H, W = image_tensor.shape
    tx = random.randint(translate_range[0], translate_range[1])
    ty = random.randint(translate_range[0], translate_range[1])

    M_single = torch.tensor(
        [[1.0, 0.0, float(tx)], [0.0, 1.0, float(ty)]],
        device=device,
        dtype=torch.float32,
    )
    M = M_single.unsqueeze(0).expand(B, -1, -1)

    img_trans = KT.warp_affine(
        image_tensor,
        M,
        dsize=(H, W),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )
    img_trans = torch.clamp(img_trans, 0, 255)

    mask_trans = KT.warp_affine(
        mask_tensor.float(),
        M,
        dsize=(H, W),
        mode="nearest",
        padding_mode="zeros",
        align_corners=False,
    )
    mask_trans = (mask_trans > 0.5).float()

    return img_trans, mask_trans


# ------------------------------
# CPU elastic deformation (per frame)
# ------------------------------

def apply_deformation_grid(
    image: np.ndarray,
    mask: np.ndarray,
    sigma: float = 5,
    points: int = 3,
    alpha: float = 50,
) -> (np.ndarray, np.ndarray):
    """
    Apply elastic deformation to one frame + mask using the SAME deformation field.

    We pass [image, mask] as a list of inputs to elasticdeform so that both share
    one random grid. Interpolation order is set separately for image (1) and
    mask (0) to preserve binary labels.
    """
    img_f = image.astype(np.float32)
    mask_f = mask.astype(np.float32)

    deformed_img, deformed_mask = elasticdeform.deform_random_grid(
        [img_f, mask_f],
        sigma=sigma,
        points=points,
        mode="constant",
        cval=0,
        order=[1, 0],  # bilinear for image, nearest for mask
    )

    deformed_img = np.clip(deformed_img, 0, 255).astype(np.uint8)
    deformed_mask = (deformed_mask > 0.5).astype(bool)

    return deformed_img, deformed_mask

# ------------------------------
# Per-sample augmentation (ONE strategy)
# ------------------------------

def augment_sample_single_strategy(
    sample: Dict[str, Any],
    strategy: str,
    device: torch.device,
) -> Dict[str, Any]:
    """
    Apply exactly ONE augmentation strategy to an entire sample (video + label + box).

    - Same parameters for all frames of a given video.
    - Parameters are re-drawn for each call (i.e., per augmented sample).
    """
    augmented = sample.copy()

    video = sample["video"].copy()  # (H, W, T), uint8
    label = sample["label"].copy()  # (H, W, T), bool
    box = sample["box"].copy()      # (H, W), bool

    H, W, T = video.shape

    # Affine strategies can be done in batch on GPU (or CPU via torch if no CUDA)
    if strategy in {"rotation", "scaling", "sheering", "translation"}:
        # video: (H, W, T) -> (T, 1, H, W)
        video_tensor = torch.from_numpy(video.transpose(2, 0, 1)).float().unsqueeze(1).to(device)
        label_tensor = torch.from_numpy(label.transpose(2, 0, 1).astype(np.float32)).unsqueeze(1).to(device)

        if strategy == "rotation":
            video_tensor, label_tensor = apply_rotation_gpu(video_tensor, label_tensor, device=device)
        elif strategy == "scaling":
            video_tensor, label_tensor = apply_zooming_gpu(video_tensor, label_tensor, device=device)
        elif strategy == "sheering":
            video_tensor, label_tensor = apply_sheering_gpu(video_tensor, label_tensor, device=device)
        elif strategy == "translation":
            video_tensor, label_tensor = apply_translation_gpu(video_tensor, label_tensor, device=device)

        video_aug = video_tensor.squeeze(1).cpu().numpy().transpose(1, 2, 0).astype(np.uint8)
        label_aug = label_tensor.squeeze(1).cpu().numpy().transpose(1, 2, 0)
        label_aug = (label_aug > 0.5).astype(bool)

        # Apply same transform to 2D box
        box_tensor = torch.from_numpy(box.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
        box_mask_tensor = box_tensor.clone()

        if strategy == "rotation":
            _, box_mask_tensor = apply_rotation_gpu(box_tensor, box_mask_tensor, device=device)
        elif strategy == "scaling":
            _, box_mask_tensor = apply_zooming_gpu(box_tensor, box_mask_tensor, device=device)
        elif strategy == "sheering":
            _, box_mask_tensor = apply_sheering_gpu(box_tensor, box_mask_tensor, device=device)
        elif strategy == "translation":
            _, box_mask_tensor = apply_translation_gpu(box_tensor, box_mask_tensor, device=device)

        box_aug = (box_mask_tensor.squeeze().cpu().numpy() > 0.5).astype(bool)

        augmented["video"] = video_aug
        augmented["label"] = label_aug
        augmented["box"] = box_aug
        return augmented

    elif strategy == "deformation":
        # Elastic deformation on CPU, frame by frame
        video_aug = video.copy()
        label_aug = label.copy()
        for t in range(T):
            frame = video[:, :, t]
            mask = label[:, :, t]
            frame_def, mask_def = apply_deformation_grid(frame, mask)
            video_aug[:, :, t] = frame_def
            label_aug[:, :, t] = mask_def

        # Deform box as well
        box_img, box_mask = apply_deformation_grid(box.astype(np.uint8) * 255, box)
        box_aug = box_mask

        augmented["video"] = video_aug
        augmented["label"] = label_aug
        augmented["box"] = box_aug
        return augmented

    else:
        raise ValueError(f"Unknown strategy: {strategy}")


# ------------------------------
# Main CLI
# ------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Augment raw/train.pkl with a single augmentation strategy."
    )
    parser.add_argument(
        "--strategy",
        "-s",
        type=str,
        choices=["rotation", "scaling", "sheering", "translation", "deformation"],
        required=True,
        help="Augmentation strategy to apply.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        required=True,
        help="Output path for augmented gzipped pickle (e.g. data/augmented/train_rotation.pkl).",
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default=None,
        help="Path to input train.pkl (gzipped pickle). "
             "Defaults to BASE_PATH/data/raw/train.pkl if not set.",
    )
    parser.add_argument(
        "--num-augmentations",
        "-n",
        type=int,
        default=1,
        help="Number of augmented samples to generate per original sample (default: 1).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run augmentation without writing any output file. Useful for testing.",
    )
    parser.add_argument(
        "--cpu-only",
        action="store_true",
        help="Force CPU even if a CUDA GPU is available.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    set_global_seed(args.seed)

    # Determine base path (project root) from this script location
    script_path = Path(__file__).resolve()
    base_path = script_path.parents[1]

    if args.input is None:
        input_path = base_path / "data" / "raw" / "train.pkl"
    else:
        input_path = Path(args.input)

    output_path = Path(args.output)

    use_gpu = torch.cuda.is_available() and not args.cpu_only
    device = torch.device("cuda" if use_gpu else "cpu")

    print(f"Base path: {base_path}")
    print(f"Input path: {input_path}")
    print(f"Output path: {output_path}")
    print(f"Strategy: {args.strategy}")
    print(f"Num augmentations per sample: {args.num_augmentations}")
    print(f"Seed: {args.seed}")
    print(f"Using device: {device}")
    print(f"Dry run: {args.dry_run}")
    print()

    # Basic input existence check (even in dry-run) to catch obvious issues
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # In dry-run mode, we do not load data or perform any augmentation.
    if args.dry_run:
        print("Dry-run requested:")
        print("  - No training data will be loaded")
        print("  - No augmentations will be computed")
        print("  - No output file will be written")
        print()
        print("This is what would happen in a normal run:")
        print(f"  - Load: {input_path}")
        print(f"  - Strategy: {args.strategy}")
        print(f"  - Augmentations per original sample: {args.num_augmentations}")
        print(f"  - Expected augmented samples: <len(train_data)> * {args.num_augmentations}")
        print(f"  - Save augmented data to: {output_path}")
        return

    print(f"Loading training data from {input_path} ...")
    train_data: List[Dict[str, Any]] = load_zipped_pickle(input_path)
    print(f"Loaded {len(train_data)} samples.")

    augmented_data: List[Dict[str, Any]] = []

    # Generate augmented samples (no raw copies, for easier merging with raw/train.pkl)
    total_to_add = len(train_data) * args.num_augmentations
    print(f"Generating {args.num_augmentations} augmented samples per original "
          f"({total_to_add} augmented samples in total)...")

    for sample in tqdm(train_data, desc=f"Augment {args.strategy}"):
        for k in range(args.num_augmentations):
            aug_sample = augment_sample_single_strategy(sample, args.strategy, device=device)
            # Name augmentation for bookkeeping; user can also encode this in the output filename
            base_name = sample.get("name", "sample")
            aug_sample["name"] = f"{base_name}_{args.strategy}_aug{k + 1}"
            augmented_data.append(aug_sample)

    print(f"Final augmented dataset size: {len(augmented_data)} "
          f"(augmented only, originals not included)")

    print(f"Saving augmented data to {output_path} ...")
    save_zipped_pickle(augmented_data, output_path)
    print("Done.")


if __name__ == "__main__":
    main()


