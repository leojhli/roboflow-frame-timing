"""Paint a frame's number into its pixels, and read it back later.

Why: when a frame comes out the other end of InferencePipeline, we need to know
WHICH frame it is. Instead of trusting the library's own frame counter, we
write the number into the picture itself, as a tiny barcode in the top-left
corner. Whatever happens to the frame in between, the number travels with it.

Layout: 32 squares of 8x8 pixels in a row (256 px wide, 8 px tall). Each square
is one bit of the number, most significant first: white = 1, black = 0.
Squares this big survive JPEG compression, which blurs fine detail but keeps
big blocks of solid colour. JPEG also works in 8x8 tiles, so these line up.
"""
import numpy as np

BITS = 32
BLOCK = 8  # pixels per side of one square
STAMP_WIDTH = BITS * BLOCK  # 256 px


def stamp_index(img: np.ndarray, idx: int) -> np.ndarray:
    """Return a copy of `img` (BGR, uint8) with `idx` painted in the top-left."""
    if img.shape[1] < STAMP_WIDTH or img.shape[0] < BLOCK:
        raise ValueError(f"image must be at least {STAMP_WIDTH}x{BLOCK} px, got {img.shape[1]}x{img.shape[0]}")
    if not 0 <= idx < 2**BITS:
        raise ValueError(f"idx must be in [0, 2**{BITS}), got {idx}")
    out = img.copy()
    for bit in range(BITS):
        is_one = (idx >> (BITS - 1 - bit)) & 1
        x = bit * BLOCK
        out[0:BLOCK, x:x + BLOCK] = 255 if is_one else 0
    return out


def read_index(img: np.ndarray) -> int:
    """Read back the number painted by `stamp_index`."""
    idx = 0
    for bit in range(BITS):
        x = bit * BLOCK
        brightness = img[0:BLOCK, x:x + BLOCK].mean()
        idx = (idx << 1) | int(brightness > 128)
    return idx
