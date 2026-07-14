"""Image preprocessing for Vintern-1B (InternVL2 family, dynamic tiling).

Mirrors the reference preprocessing from the 5CD-AI/Vintern model cards:
dynamic aspect-ratio tiling into 448×448 crops + a thumbnail tile.
"""

from __future__ import annotations

from PIL import Image

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _build_transform(input_size: int):
    import torchvision.transforms as T
    from torchvision.transforms.functional import InterpolationMode

    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def _closest_ratio(aspect_ratio: float, ratios, width: int, height: int, image_size: int):
    best_diff, best = float("inf"), (1, 1)
    area = width * height
    for r in ratios:
        target = r[0] / r[1]
        diff = abs(aspect_ratio - target)
        if diff < best_diff or (diff == best_diff and area > 0.5 * image_size * image_size * r[0] * r[1]):
            best_diff, best = diff, r
    return best


def dynamic_preprocess(image: Image.Image, min_num: int = 1, max_num: int = 6,
                       image_size: int = 448, use_thumbnail: bool = True) -> list[Image.Image]:
    orig_w, orig_h = image.size
    aspect_ratio = orig_w / orig_h
    ratios = sorted(
        {(i, j) for n in range(min_num, max_num + 1)
         for i in range(1, n + 1) for j in range(1, n + 1)
         if min_num <= i * j <= max_num},
        key=lambda x: x[0] * x[1],
    )
    ar = _closest_ratio(aspect_ratio, ratios, orig_w, orig_h, image_size)
    target_w, target_h = image_size * ar[0], image_size * ar[1]
    blocks = ar[0] * ar[1]
    resized = image.resize((target_w, target_h))
    tiles = []
    cols = target_w // image_size
    for i in range(blocks):
        box = (
            (i % cols) * image_size,
            (i // cols) * image_size,
            ((i % cols) + 1) * image_size,
            ((i // cols) + 1) * image_size,
        )
        tiles.append(resized.crop(box))
    if use_thumbnail and len(tiles) != 1:
        tiles.append(image.resize((image_size, image_size)))
    return tiles


def load_image_tiles(image_path: str, input_size: int = 448, max_num: int = 6):
    """Path → stacked pixel_values tensor (tiles, 3, H, W)."""
    import torch

    image = Image.open(image_path).convert("RGB")
    transform = _build_transform(input_size)
    tiles = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    return torch.stack([transform(t) for t in tiles])
