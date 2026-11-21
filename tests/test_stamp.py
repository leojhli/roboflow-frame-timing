"""Tests for the frame-index stamp: a tiny barcode we paint into each frame."""
import cv2
import numpy as np
import pytest

from frametiming.stamp import read_index, stamp_index


def busy_image(width: int = 640, height: int = 480) -> np.ndarray:
    """Random noise: the worst case for a barcode, because nothing is plain."""
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, (height, width, 3), dtype=np.uint8)


@pytest.mark.parametrize("idx", [0, 1, 299, 2**31 - 1])
def test_roundtrip(idx):
    assert read_index(stamp_index(busy_image(), idx)) == idx


def test_original_untouched():
    img = busy_image()
    before = img.copy()
    stamp_index(img, 12345)
    assert np.array_equal(img, before)


@pytest.mark.parametrize("idx", [0, 299, 2**31 - 1])
def test_stamp_survives_jpeg(idx):
    stamped = stamp_index(busy_image(), idx)
    ok, encoded = cv2.imencode(".jpg", stamped, [cv2.IMWRITE_JPEG_QUALITY, 70])
    assert ok
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    assert read_index(decoded) == idx


def test_rejects_narrow_image():
    with pytest.raises(ValueError):
        stamp_index(busy_image(width=200), 1)


@pytest.mark.parametrize("idx", [-1, 2**32])
def test_rejects_out_of_range_index(idx):
    with pytest.raises(ValueError):
        stamp_index(busy_image(), idx)
