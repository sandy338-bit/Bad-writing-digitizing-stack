from __future__ import annotations

from io import BytesIO
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageOps, UnidentifiedImageError

from .schemas import Box

Image.MAX_IMAGE_PIXELS = 25_000_000


def load_image(raw: bytes) -> Image.Image:
    try:
        image = Image.open(BytesIO(raw))
        if image.width * image.height > 25_000_000:
            raise ValueError("Image exceeds 25 megapixels")
        if getattr(image, "n_frames", 1) > 1:
            raise ValueError("Upload one page at a time; animated or multi-page images are unsupported")
        image.verify()
        image = Image.open(BytesIO(raw))
        return ImageOps.exif_transpose(image).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as exc:
        raise ValueError(str(exc) if isinstance(exc, ValueError) else "File is not a valid image") from exc


def enhance(image: Image.Image) -> Image.Image:
    """Light local contrast only; preserve the original for the vision reader."""
    rgb = np.asarray(image)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8)).apply(l)
    result = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2RGB)
    return ImageEnhance.Sharpness(Image.fromarray(result)).enhance(1.2)


def crop(image: Image.Image, box: Box, padding: int = 8) -> Image.Image:
    return image.crop((max(0, box.x1 - padding), max(0, box.y1 - padding),
                       min(image.width, box.x2 + padding), min(image.height, box.y2 + padding)))


def quality_score(image: Image.Image) -> float:
    gray = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2GRAY)
    if min(gray.shape) < 8:
        return 0.0
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
    return float(min(1.0, sharpness / 120.0))


def detect_lines(image: Image.Image) -> list[Box]:
    """Geometry fallback only: locate ink clusters, without inventing OCR text."""
    gray = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2GRAY)
    _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if np.count_nonzero(ink) > ink.size * 0.6:
        ink = cv2.bitwise_not(ink)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(3, image.width // 60), 3))
    merged = cv2.dilate(ink, kernel)
    count, _, stats, _ = cv2.connectedComponentsWithStats(merged)
    boxes = []
    for x, y, width, height, area in stats[1:count]:
        if area < 25 or height < 5 or width < 6 or area > ink.size * 0.8:
            continue
        boxes.append(Box(x1=int(x), y1=int(y), x2=int(x + width), y2=int(y + height)))
    return sorted(boxes, key=lambda b: (b.y1, b.x1))


def detect_text_line_bands(image: Image.Image, limit: int = 80) -> list[Box]:
    """Find approximate horizontal handwriting lines for line-only HTR models."""
    gray = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2GRAY)
    _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if np.count_nonzero(ink) > ink.size * 0.6:
        ink = cv2.bitwise_not(ink)
    # Long notebook rules are not text and can otherwise join nearby lines.
    rule_width = max(30, image.width // 3)
    rules = cv2.morphologyEx(ink, cv2.MORPH_OPEN,
                             cv2.getStructuringElement(cv2.MORPH_RECT, (rule_width, 1)))
    ink = cv2.subtract(ink, rules)
    active = np.count_nonzero(ink, axis=1) >= max(3, image.width // 200)
    gap = max(2, min(8, image.height // 180))
    active = cv2.dilate(active.astype(np.uint8).reshape(-1, 1),
                        np.ones((gap, 1), np.uint8)).ravel().astype(bool)
    edges = np.diff(np.pad(active.astype(np.int8), (1, 1)))
    bands = []
    for y1, y2 in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        if y2 - y1 < 8 or y2 - y1 > max(60, image.height * 0.25):
            continue
        ys, xs = np.nonzero(ink[y1:y2])
        if xs.size < 25:
            continue
        x1, x2 = int(xs.min()), int(xs.max()) + 1
        if x2 - x1 < 12:
            continue
        bands.append(Box(x1=max(0, x1 - 8), y1=max(0, int(y1) - 4),
                         x2=min(image.width, x2 + 8), y2=min(image.height, int(y2) + 4)))
    if not bands and image.height > 180:
        # Cursive lines can overlap vertically, leaving no completely empty
        # rows. Split a merged band at valleys between horizontal ink peaks.
        projection = np.count_nonzero(ink, axis=1).astype(np.float32)
        smoothed = np.convolve(projection, np.ones(9, dtype=np.float32) / 9, mode="same")
        threshold = max(10.0, float(smoothed.max()) * 0.32)
        peak_indices = np.flatnonzero((smoothed >= threshold) &
                                      (smoothed >= np.roll(smoothed, 1)) &
                                      (smoothed > np.roll(smoothed, -1)))
        distance = max(20, min(70, image.height // 8))
        selected: list[int] = []
        for peak in sorted(peak_indices, key=lambda y: float(smoothed[y]), reverse=True):
            if all(abs(int(peak) - other) >= distance for other in selected):
                selected.append(int(peak))
        selected.sort()
        active_rows = np.flatnonzero(projection >= max(3, threshold * .2))
        if len(selected) >= 2 and active_rows.size:
            edges = [max(0, int(active_rows[0]) - 3)]
            edges += [(a + b) // 2 for a, b in zip(selected, selected[1:])]
            edges += [min(image.height, int(active_rows[-1]) + 4)]
            for y1, y2 in zip(edges, edges[1:]):
                ys, xs = np.nonzero(ink[y1:y2])
                if y2 - y1 < 8 or xs.size < 25:
                    continue
                bands.append(Box(x1=max(0, int(xs.min()) - 8), y1=max(0, y1 - 3),
                                 x2=min(image.width, int(xs.max()) + 9),
                                 y2=min(image.height, y2 + 3)))
    if not bands and image.height <= 180 and image.width >= image.height * 2:
        return [Box(x1=0, y1=0, x2=image.width, y2=image.height)]
    return bands[:limit]
