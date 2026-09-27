"""Numeric + ASCII verification that the rasteriser produces a real picture.

Two independent checks:
  1. adjacent-row correlation (a coherent picture scores high, garbage low),
     plus a deliberate mis-timing control to prove the metric discriminates;
  2. an ASCII rendering so the structure can be eyeballed in a terminal.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fpv_rf import dsp  # noqa: E402

RAMP = " .:-=+*#%@"


def ascii_art(img: np.ndarray, cols: int = 76) -> str:
    h, w = img.shape
    rows = max(1, int(round(cols * h / w / 2.1)))  # ~2.1:1 char aspect
    out = []
    for r in range(rows):
        a = int(r * h / rows)
        b = max(a + 1, int((r + 1) * h / rows))
        line = []
        for c in range(cols):
            x0 = int(c * w / cols)
            x1 = max(x0 + 1, int((c + 1) * w / cols))
            v = float(img[a:b, x0:x1].mean())
            line.append(RAMP[min(len(RAMP) - 1, int(v / 256.0 * len(RAMP)))])
        out.append("".join(line))
    return "\n".join(out)


def row_correlation(img: np.ndarray) -> float:
    # subtract each row's own mean first: pure horizontal banding then scores
    # 0, so this measures picture structure rather than per-row brightness
    x = img.astype(np.float32) - img.mean(axis=1, keepdims=True)
    a = x[:-1].ravel()
    b = x[1:].ravel()
    a -= a.mean()
    b -= b.mean()
    d = float(np.sqrt((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / d) if d > 0 else 0.0


def col_correlation(img: np.ndarray) -> float:
    a = img[:, :-1].astype(np.float32).ravel()
    b = img[:, 1:].astype(np.float32).ravel()
    a -= a.mean()
    b -= b.mean()
    d = float(np.sqrt((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / d) if d > 0 else 0.0


def main() -> int:
    p = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parents[1] / "_validate_out" / "frame_none.png"
    from PIL import Image

    img = np.asarray(Image.open(p).convert("L"))
    print(f"image {img.shape[1]}x{img.shape[0]}  mean={img.mean():.1f} std={img.std():.1f}")
    print(f"adjacent-ROW correlation : {row_correlation(img):+.4f}   (coherent picture -> high)")
    print(f"adjacent-COL correlation : {col_correlation(img):+.4f}")

    # control: a deliberately mis-timed decode should score much lower
    rng = np.random.default_rng(7)
    scrambled = img[rng.permutation(img.shape[0])]
    print(f"control (row-shuffled)   : {row_correlation(scrambled):+.4f}   (should be low)")
    noise = rng.integers(0, 255, img.shape, dtype=np.uint8)
    print(f"control (pure noise)     : {row_correlation(noise):+.4f}   (should be ~0)")
    print()
    print(ascii_art(img))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
