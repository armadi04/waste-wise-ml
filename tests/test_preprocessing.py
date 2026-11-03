from __future__ import annotations

from io import BytesIO

import numpy as np
import pytest
from PIL import Image

from waste_ml.config import CLASSES, IMAGE_SIZE
from waste_ml.preprocessing import (
    InvalidImageError,
    UnsupportedImageFormatError,
    load_image,
    preprocess_image,
    preprocess_image_bytes,
)


def image_bytes(mode: str = "RGB", size: tuple[int, int] = (320, 240), fmt: str = "JPEG") -> bytes:
    image = Image.new(mode, size)
    if mode == "RGB":
        image.putdata(
            [
                ((x * 7) % 256, (y * 5) % 256, ((x + y) * 3) % 256)
                for y in range(size[1])
                for x in range(size[0])
            ]
        )
    buffer = BytesIO()
    image.save(buffer, format=fmt)
    return buffer.getvalue()


def test_load_image_converts_to_rgb() -> None:
    image = load_image(image_bytes(mode="L", fmt="PNG"))
    assert image.mode == "RGB"
    assert image.size == (320, 240)


def test_preprocess_image_shape_dtype_and_range() -> None:
    image = load_image(image_bytes())
    array = preprocess_image(image)
    assert array.shape == (*IMAGE_SIZE, 3)
    assert array.dtype == np.float32
    assert float(array.min()) >= -1.0
    assert float(array.max()) <= 1.0


def test_preprocess_image_bytes_resizes() -> None:
    array = preprocess_image_bytes(image_bytes(size=(512, 180)))
    assert array.shape == (224, 224, 3)


def test_class_mapping_is_stable() -> None:
    assert CLASSES == ("organic", "plastic", "paper")


def test_invalid_image_raises() -> None:
    with pytest.raises(InvalidImageError):
        load_image(b"this is definitely not an image")


def test_empty_image_raises() -> None:
    with pytest.raises(InvalidImageError):
        load_image(b"")


def test_unsupported_format_raises() -> None:
    with pytest.raises(UnsupportedImageFormatError):
        load_image(image_bytes(fmt="GIF"))
