#!/usr/bin/env python3
"""Generate the legacy launcher icons for CSPRNG Predictor.

Writes
    app/src/main/res/mipmap-{mdpi,hdpi,xhdpi,xxhdpi,xxxhdpi}/ic_launcher.png
        (48 / 72 / 96 / 144 / 192 px, used by Android 7.x; Android 8+ uses the
        adaptive icon in mipmap-anydpi-v26/ic_launcher.xml instead)
    docs/icon-512.png
        (512 px, for the README / store listings)

Design: a blue (#1E5EFF) rounded square, a thin white ring and 11 white dots
(the 11 picked numbers) arranged 4-3-4 in a hexagonal cluster. The same
geometry is used by res/drawable/ic_launcher_foreground.xml, so both icon
flavours look alike.

Every icon is drawn at 4x its final size and downscaled with a high quality
filter, which gives smooth (anti-aliased) edges.

Usage:  python3 scripts/gen_icons.py            (run from any directory)
Requires Pillow (pip install Pillow / apt install python3-pil).
"""

import math
import os
import sys

try:
    from PIL import Image, ImageDraw
except ImportError:  # pragma: no cover - environment problem, not a code path
    sys.stderr.write("gen_icons.py needs Pillow: pip install Pillow  (or apt install python3-pil)\n")
    sys.exit(2)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "app", "src", "main", "res")
DOCS = os.path.join(ROOT, "docs")

SUPERSAMPLE = 4

BLUE = (0x1E, 0x5E, 0xFF)
WHITE = (0xFF, 0xFF, 0xFF)

# Geometry in fractions of the icon edge length (icon centre = 0.5, 0.5).
SQUARE_INSET = 0.05     # transparent margin around the rounded square
SQUARE_RADIUS = 0.22    # corner radius of the rounded square
RING_RADIUS = 0.30      # centre line of the ring
RING_WIDTH = 0.045      # stroke width of the ring
DOT_SPACING = 0.11      # distance between neighbouring dot centres
DOT_RADIUS = 0.038

LEGACY_SIZES = [
    ("mipmap-mdpi", 48),
    ("mipmap-hdpi", 72),
    ("mipmap-xhdpi", 96),
    ("mipmap-xxhdpi", 144),
    ("mipmap-xxxhdpi", 192),
]


def dot_centres():
    """11 dot centres (unit offsets from the icon centre): rows of 4, 3 and 4,
    hexagonally packed so that the cluster is symmetric and compact."""
    s = DOT_SPACING
    h = s * math.sqrt(3.0) / 2.0
    pts = []
    for row_y, xs in ((-h, (-1.5, -0.5, 0.5, 1.5)),
                      (0.0, (-1.0, 0.0, 1.0)),
                      (h, (-1.5, -0.5, 0.5, 1.5))):
        for x in xs:
            pts.append((x * s, row_y))
    assert len(pts) == 11
    return pts


def _resample_filter():
    resampling = getattr(Image, "Resampling", None)
    if resampling is not None:
        return resampling.LANCZOS
    return Image.LANCZOS


def render(size):
    """Return an RGBA image of the icon with edge length `size` pixels."""
    big = size * SUPERSAMPLE
    # Transparent pixels carry the blue colour so filtering never darkens the edge.
    img = Image.new("RGBA", (big, big), BLUE + (0,))
    draw = ImageDraw.Draw(img)

    inset = SQUARE_INSET * big
    draw.rounded_rectangle(
        [inset, inset, big - inset - 1, big - inset - 1],
        radius=SQUARE_RADIUS * big,
        fill=BLUE + (255,),
    )

    c = big / 2.0
    ring_outer = (RING_RADIUS + RING_WIDTH / 2.0) * big
    draw.ellipse(
        [c - ring_outer, c - ring_outer, c + ring_outer, c + ring_outer],
        outline=WHITE + (255,),
        width=max(1, int(round(RING_WIDTH * big))),
    )

    r = DOT_RADIUS * big
    for dx, dy in dot_centres():
        x = c + dx * big
        y = c + dy * big
        draw.ellipse([x - r, y - r, x + r, y + r], fill=WHITE + (255,))

    # Downscale in premultiplied-alpha space for clean anti-aliased edges.
    small = img.convert("RGBa").resize((size, size), _resample_filter())
    return small.convert("RGBA")


def write_png(image, path):
    directory = os.path.dirname(path)
    if not os.path.isdir(directory):
        os.makedirs(directory)
    image.save(path, "PNG", optimize=True)
    print("wrote %s (%dx%d, %d bytes)" % (os.path.relpath(path, ROOT), image.size[0],
                                           image.size[1], os.path.getsize(path)))


def main():
    for folder, size in LEGACY_SIZES:
        write_png(render(size), os.path.join(RES, folder, "ic_launcher.png"))
    write_png(render(512), os.path.join(DOCS, "icon-512.png"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
