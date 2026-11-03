"""Image loading and preprocessing shared by training and inference."""

from io import BytesIO

import numpy as np
from PIL import Image, UnidentifiedImageError
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from waste_ml.config import IMAGE_SIZE

SUPPORTED_IMAGE_FORMATS: dict[str, str] = {
    "JPEG": "image/jpeg",
    "PNG": "image/png",
    "WEBP": "image/webp",
}


class InvalidImageError(ValueError):
    """Raised when bytes cannot be decoded as an image."""


class UnsupportedImageFormatError(ValueError):
    """Raised when the decoded image format is not supported."""


def load_image(data: bytes) -> Image.Image:
    """Decode image bytes, verify the format, and convert to RGB."""
    if not data:
        raise InvalidImageError("Empty image file")

    try:
        with Image.open(BytesIO(data)) as source:
            source.load()
            image_format = source.format
            image = source.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise InvalidImageError("File cannot be read as an image") from exc

    if image_format not in SUPPORTED_IMAGE_FORMATS:
        raise UnsupportedImageFormatError("Unsupported image format")

    return image


def preprocess_image(image: Image.Image, image_size: tuple[int, int] = IMAGE_SIZE) -> np.ndarray:
    """Resize, convert to RGB float32, and apply MobileNetV2 preprocessing."""
    resized = image.convert("RGB").resize(image_size, Image.Resampling.BILINEAR)
    array = np.asarray(resized, dtype=np.float32)
    return preprocess_input(array)


def preprocess_image_bytes(data: bytes, image_size: tuple[int, int] = IMAGE_SIZE) -> np.ndarray:
    """Decode and preprocess raw image bytes."""
    return preprocess_image(load_image(data), image_size=image_size)
