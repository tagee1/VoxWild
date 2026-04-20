"""Generate VoxWild logo + Windows icon — pixel-perfect at every size.

No supersampling — each size is drawn at its exact pixel count with
integer-aligned coordinates. Thin effects (specular, glow) are only
added where there are enough pixels to render them clearly.

Sizes in the .ico: 16, 20, 24, 32, 40, 48, 64, 96, 128, 256
"""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter

HERE = Path(__file__).parent

# Palette
SURFACE   = (23, 23, 23, 255)
ACCENT    = (0, 217, 139, 255)
ACCENT_H  = (30, 240, 168, 255)
ACCENT_D  = (0, 163, 106, 255)
OUTLINE   = (10, 61, 40, 255)


def _draw_bars(d, cx, cy, bar_w, gap, bar_heights, radius, use_gradient=False):
    """Draw centered vertical bars. If use_gradient, apply a simple 2-stop
    vertical gradient (ACCENT_H at top → ACCENT_D at bottom)."""
    n = len(bar_heights)
    total_w = n * bar_w + (n - 1) * gap
    x0 = cx - total_w // 2

    for i, h in enumerate(bar_heights):
        x = x0 + i * (bar_w + gap)
        y1 = cy - h // 2
        y2 = cy + h // 2

        if use_gradient and h > 8:
            # 2-stop vertical gradient bar
            for row in range(h):
                t = row / max(1, h - 1)
                r = int(ACCENT_H[0] + (ACCENT_D[0] - ACCENT_H[0]) * t)
                g = int(ACCENT_H[1] + (ACCENT_D[1] - ACCENT_H[1]) * t)
                b = int(ACCENT_H[2] + (ACCENT_D[2] - ACCENT_H[2]) * t)
                d.rectangle([x, y1 + row, x + bar_w - 1, y1 + row], fill=(r, g, b, 255))
            # Round the corners by masking with a rounded rect
            # (draw over corners with background)
            if radius > 0:
                mask = Image.new("L", (bar_w, h), 0)
                mk = ImageDraw.Draw(mask)
                mk.rounded_rectangle([0, 0, bar_w - 1, h - 1], radius=radius, fill=255)
                # Apply mask by drawing background over non-masked areas
                for row in range(h):
                    for col in range(bar_w):
                        if mask.getpixel((col, row)) == 0:
                            d.point((x + col, y1 + row), fill=(0, 0, 0, 0))
        else:
            # Flat solid bar
            if radius > 0:
                d.rounded_rectangle([x, y1, x + bar_w - 1, y2],
                                    radius=radius, fill=ACCENT)
            else:
                d.rectangle([x, y1, x + bar_w - 1, y2], fill=ACCENT)


def draw_icon(size: int) -> Image.Image:
    """Draw a single icon at the exact pixel size requested."""
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)

    # ── Background rounded square ────────────────────────────────────────
    pad = max(1, size // 16)
    r = max(2, size // 5)

    # Draw with outline for larger sizes
    if size >= 64:
        d.rounded_rectangle([pad, pad, size - pad - 1, size - pad - 1],
                            radius=r, fill=SURFACE, outline=OUTLINE, width=1)
    else:
        d.rounded_rectangle([pad, pad, size - pad - 1, size - pad - 1],
                            radius=r, fill=SURFACE)

    cx = size // 2
    cy = size // 2

    # ── Bars — size-specific tuning ──────────────────────────────────────
    if size <= 20:
        # Tiny: 3 bars, flat, no rounding
        bw = max(2, size // 7)
        gap = max(1, size // 10)
        usable = int(size * 0.55)
        heights = [int(usable * r) for r in [0.5, 1.0, 0.5]]
        _draw_bars(d, cx, cy, bw, gap, heights, radius=0, use_gradient=False)

    elif size <= 32:
        # Small: 3 bars, slight rounding, flat
        bw = max(2, size // 6)
        gap = max(1, size // 10)
        usable = int(size * 0.58)
        heights = [int(usable * r) for r in [0.45, 1.0, 0.45]]
        _draw_bars(d, cx, cy, bw, gap, heights, radius=1, use_gradient=False)

    elif size <= 48:
        # Medium: 5 bars, flat
        bw = max(2, size // 9)
        gap = max(1, size // 16)
        usable = int(size * 0.60)
        heights = [int(usable * r) for r in [0.38, 0.68, 1.0, 0.58, 0.30]]
        _draw_bars(d, cx, cy, bw, gap, heights,
                   radius=max(1, bw // 3), use_gradient=False)

    elif size <= 96:
        # Large: 5 bars with gradient
        bw = max(3, size // 10)
        gap = max(2, size // 18)
        usable = int(size * 0.58)
        heights = [int(usable * r) for r in [0.38, 0.68, 1.0, 0.58, 0.30]]
        _draw_bars(d, cx, cy, bw, gap, heights,
                   radius=max(1, bw // 3), use_gradient=True)

    else:
        # XL (128, 256, 512): 5 bars with gradient + glow
        bw = max(4, size // 12)
        gap = max(2, size // 20)
        usable = int(size * 0.56)
        heights = [int(usable * r) for r in [0.38, 0.68, 1.0, 0.58, 0.30]]
        _draw_bars(d, cx, cy, bw, gap, heights,
                   radius=max(2, bw // 3), use_gradient=True)

        # Add glow layer behind bars
        glow = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        gd = ImageDraw.Draw(glow)
        n = len(heights)
        total_w = n * bw + (n - 1) * gap
        glow_pad = size // 8
        gd.rounded_rectangle(
            [cx - total_w // 2 - glow_pad, cy - usable // 2 - glow_pad,
             cx + total_w // 2 + glow_pad, cy + usable // 2 + glow_pad],
            radius=size // 8,
            fill=(0, 217, 139, 35))
        glow = glow.filter(ImageFilter.GaussianBlur(radius=size // 16))
        # Composite: glow behind current image
        out = Image.alpha_composite(glow, img)
        return out

    return img


# ── Outputs ──────────────────────────────────────────────────────────────────
def make_icon_ico(path: Path):
    """Windows .ico with size-specific images."""
    sizes = [16, 20, 24, 32, 40, 48, 64, 96, 128, 256]
    imgs = [draw_icon(s) for s in sizes]
    imgs[0].save(
        path, format="ICO",
        sizes=[(s, s) for s in sizes],
        append_images=imgs[1:],
    )
    print(f"Saved {path.name}  (ICO: {sizes})")


def make_pngs(out_dir: Path):
    """PNG logos at sizes used in the app + website."""
    for size in (32, 64, 128, 256, 512):
        img = draw_icon(size)
        name = "logo.png" if size == 512 else f"logo-{size}.png"
        img.save(out_dir / name, "PNG", optimize=True)
        print(f"Saved {name:<16s}  ({size}x{size})")


if __name__ == "__main__":
    make_icon_ico(HERE / "icon.ico")
    make_pngs(HERE)
    site = Path("C:/tts-studio-site")
    if site.is_dir():
        draw_icon(512).save(site / "logo.png", "PNG", optimize=True)
        draw_icon(32).save(site / "favicon.png", "PNG", optimize=True)
        print("Copied to site repo")
    print("Done.")
