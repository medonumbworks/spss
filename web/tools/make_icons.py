"""Generate web/icon-192.png and web/icon-512.png without any imaging library.

Pure stdlib: pixels are computed from signed distances (rounded square + the same
pulse polyline as icon.svg) and written as RGBA PNG chunks by hand via zlib.
Run from anywhere: python3 web/tools/make_icons.py
"""

from __future__ import annotations

import math
import struct
import zlib
from pathlib import Path

WEB_DIR = Path(__file__).resolve().parents[1]

# colors match icon.svg / the app's dark scheme
PANEL = (31, 40, 53)      # #1f2835
LINE = (76, 195, 138)     # #4cc38a

# geometry in the 512-unit space of icon.svg (scaled per output size)
CORNER_RADIUS = 96.0
PULSE = [(72, 288), (168, 288), (208, 180), (256, 352), (296, 236), (320, 288), (440, 288)]
PULSE_WIDTH = 34.0


def rounded_rect_sdf(x: float, y: float, size: float, radius: float) -> float:
    """Signed distance to a size x size rounded square centred on the canvas."""
    half = size / 2.0
    qx = abs(x - half) - (half - radius)
    qy = abs(y - half) - (half - radius)
    outside = math.hypot(max(qx, 0.0), max(qy, 0.0))
    return outside + min(max(qx, qy), 0.0) - radius


def segment_distance(x: float, y: float, ax: float, ay: float, bx: float, by: float) -> float:
    vx, vy = bx - ax, by - ay
    wx, wy = x - ax, y - ay
    seg_len2 = vx * vx + vy * vy
    t = 0.0 if seg_len2 == 0 else max(0.0, min(1.0, (wx * vx + wy * vy) / seg_len2))
    return math.hypot(wx - t * vx, wy - t * vy)


def coverage(sdf: float) -> float:
    """1 px wide smooth edge around the zero crossing (cheap anti-aliasing)."""
    return max(0.0, min(1.0, 0.5 - sdf))


def render(size: int) -> bytes:
    scale = size / 512.0
    radius = CORNER_RADIUS * scale
    half_w = PULSE_WIDTH * scale / 2.0
    pts = [(px * scale, py * scale) for px, py in PULSE]
    segments = list(zip(pts, pts[1:]))
    rows = bytearray()
    for j in range(size):
        y = j + 0.5
        rows.append(0)  # PNG filter type 0 (None) for every row
        for i in range(size):
            x = i + 0.5
            bg_a = coverage(rounded_rect_sdf(x, y, size, radius))
            if bg_a == 0.0:
                rows.extend((0, 0, 0, 0))
                continue
            d = min(segment_distance(x, y, a[0], a[1], b[0], b[1]) for a, b in segments)
            line_a = coverage(d - half_w)
            r = LINE[0] * line_a + PANEL[0] * (1.0 - line_a)
            g = LINE[1] * line_a + PANEL[1] * (1.0 - line_a)
            b = LINE[2] * line_a + PANEL[2] * (1.0 - line_a)
            rows.extend((int(r + 0.5), int(g + 0.5), int(b + 0.5), int(bg_a * 255 + 0.5)))
    return bytes(rows)


def chunk(kind: bytes, payload: bytes) -> bytes:
    return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload))


def write_png(path: Path, size: int) -> None:
    ihdr = struct.pack(">IIBBBBB", size, size, 8, 6, 0, 0, 0)  # 8-bit RGBA
    idat = zlib.compress(render(size), 9)
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", idat) + chunk(b"IEND", b""))
    print(f"{path} ({path.stat().st_size} bytes)")


if __name__ == "__main__":
    for size in (192, 512):
        write_png(WEB_DIR / f"icon-{size}.png", size)
