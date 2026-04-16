"""Generate VoxWild logo + Windows icon — with size-specific optimizations.

Every size in the .ico is hand-tuned:
  16-24px : 3 bars, pixel-aligned, no highlights (clarity at tiny sizes)
  32-48px : 5 bars, no highlights, tight proportions
  64-256px : 5 bars, glossy highlights, subtle glow

Also outputs multiple PNG sizes for web/app use.
"""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter

HERE = Path(__file__).parent

# Palette
BG        = (13, 13, 13, 255)        # page/window bg
SURFACE   = (23, 23, 23, 255)        # icon surface
ACCENT    = (0, 217, 139, 255)       # emerald
ACCENT_H  = (46, 229, 160, 255)      # emerald bright
ACCENT_D  = (0, 163, 106, 255)       # emerald dark
OUTLINE   = (10, 61, 40, 255)        # dark emerald border


# ─── Small (16-24): minimal design, 3 bars, pixel-crisp ──────────────────────
def draw_small(size: int) -> Image.Image:
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d   = ImageDraw.Draw(img)

    # Rounded square background
    pad = 1
    r   = max(2, size // 5)
    d.rounded_rectangle(
        [pad, pad, size - pad - 1, size - pad - 1],
        radius=r, fill=SURFACE,
    )

    # 3 bars — short, tall, short
    cx      = size // 2
    cy      = size // 2
    bar_w   = max(2, size // 6)
    gap     = max(1, size // 12)
    total_w = 3 * bar_w + 2 * gap
    x0      = cx - total_w // 2

    usable_h = int(size * 0.62)
    heights = [int(usable_h * 0.5), usable_h, int(usable_h * 0.5)]

    for i, h in enumerate(heights):
        x  = x0 + i * (bar_w + gap)
        y1 = cy - h // 2
        y2 = cy + h // 2
        # Solid emerald, no rounding at this scale (looks cleaner)
        d.rectangle([x, y1, x + bar_w - 1, y2], fill=ACCENT)
    return img


# ─── Medium (32-48): 5 bars, clean, still no highlights ──────────────────────
def draw_medium(size: int) -> Image.Image:
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d   = ImageDraw.Draw(img)

    pad = max(1, size // 16)
    r   = max(3, size // 5)
    d.rounded_rectangle(
        [pad, pad, size - pad - 1, size - pad - 1],
        radius=r, fill=SURFACE,
        outline=OUTLINE, width=1,
    )

    # 5 bars — sound-wave pattern
    cx      = size // 2
    cy      = size // 2
    bar_w   = max(2, size // 9)
    gap     = max(1, size // 18)
    total_w = 5 * bar_w + 4 * gap
    x0      = cx - total_w // 2

    usable_h = int(size * 0.62)
    ratios   = [0.42, 0.72, 1.0, 0.58, 0.30]

    for i, ratio in enumerate(ratios):
        h  = int(usable_h * ratio)
        x  = x0 + i * (bar_w + gap)
        y1 = cy - h // 2
        y2 = cy + h // 2
        # Slight rounding at this size looks nice
        d.rounded_rectangle(
            [x, y1, x + bar_w - 1, y2],
            radius=max(1, bar_w // 3),
            fill=ACCENT,
        )
    return img


# ─── Large (64-256): 5 bars, glossy highlights, subtle glow ──────────────────
def draw_large(size: int) -> Image.Image:
    # Build into a larger canvas then downsize? No — draw at exact size for
    # pixel fidelity on all sizes. Supersample only for 64+.
    ss = 2 if size <= 128 else 1
    s  = size * ss

    img = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    d   = ImageDraw.Draw(img)

    # Rounded background
    pad = max(2, s // 16)
    r   = s // 6
    d.rounded_rectangle(
        [pad, pad, s - pad, s - pad],
        radius=r, fill=SURFACE,
        outline=OUTLINE, width=max(1, s // 128),
    )

    # Inner top highlight (subtle)
    hl = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    hd = ImageDraw.Draw(hl)
    hd.rounded_rectangle(
        [pad + 2, pad + 2, s - pad - 2, int(s * 0.35)],
        radius=r - 2,
        fill=(255, 255, 255, 10),
    )
    img = Image.alpha_composite(img, hl)
    d   = ImageDraw.Draw(img)

    # 5 bars
    cx      = s // 2
    cy      = s // 2
    bar_w   = max(3, s // 14)
    gap     = max(2, s // 22)
    total_w = 5 * bar_w + 4 * gap
    x0      = cx - total_w // 2

    usable_h = int(s * 0.58)
    ratios   = [0.40, 0.72, 1.0, 0.60, 0.30]

    for i, ratio in enumerate(ratios):
        h  = int(usable_h * ratio)
        x  = x0 + i * (bar_w + gap)
        y1 = cy - h // 2
        y2 = cy + h // 2
        # Glossy bar — gradient fill + specular
        bar = _glossy_bar(bar_w, h)
        img.paste(bar, (x, y1), bar)

    # Slight outer emerald glow
    glow = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    gd   = ImageDraw.Draw(glow)
    gd.rounded_rectangle(
        [pad, pad, s - pad, s - pad],
        radius=r,
        fill=(0, 217, 139, 24),
    )
    glow = glow.filter(ImageFilter.GaussianBlur(radius=s // 48))
    out  = Image.alpha_composite(glow, img)

    # Downsample supersampled canvas
    if ss > 1:
        out = out.resize((size, size), Image.LANCZOS)
    return out


def _glossy_bar(width: int, height: int) -> Image.Image:
    """Single emerald bar with vertical gradient and specular highlight."""
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))

    # Rounded rect mask
    mask = Image.new("L", (width, height), 0)
    mk   = ImageDraw.Draw(mask)
    mk.rounded_rectangle(
        [0, 0, width - 1, height - 1],
        radius=max(1, width // 3), fill=255,
    )

    # Vertical gradient fill
    grad = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    gd   = ImageDraw.Draw(grad)
    for y in range(height):
        t = y / max(1, height - 1)
        if t < 0.5:
            k = t * 2
            r = int(46  + (0   - 46 ) * k)
            g = int(229 + (217 - 229) * k)
            b = int(160 + (139 - 160) * k)
        else:
            k = (t - 0.5) * 2
            r = int(0   + (0   - 0  ) * k)
            g = int(217 + (163 - 217) * k)
            b = int(139 + (106 - 139) * k)
        gd.rectangle([0, y, width, y + 1], fill=(r, g, b, 255))
    img.paste(grad, (0, 0), mask)

    # Specular highlight on top 22% of bar (white band, soft)
    spec_h = max(1, height // 5)
    spec   = Image.new("RGBA", (width, spec_h), (0, 0, 0, 0))
    sd     = ImageDraw.Draw(spec)
    for y in range(spec_h):
        alpha = int(100 * (1 - y / spec_h))
        sd.rectangle(
            [max(1, width // 5), y, width - max(1, width // 5), y + 1],
            fill=(255, 255, 255, alpha),
        )
    # Blur the specular a hair
    spec = spec.filter(ImageFilter.GaussianBlur(radius=0.5))
    img.paste(spec, (0, 2), spec)

    return img


# ─── Dispatcher ──────────────────────────────────────────────────────────────
def draw_for_size(size: int) -> Image.Image:
    if size <= 24:
        return draw_small(size)
    if size <= 48:
        return draw_medium(size)
    return draw_large(size)


# ─── Outputs ─────────────────────────────────────────────────────────────────
def make_icon_ico(path: Path):
    """Windows .ico with size-specific images — each one is hand-drawn for its
    pixel count, not just a scaled-down version of the largest."""
    sizes = [16, 20, 24, 32, 40, 48, 64, 128, 256]
    imgs  = [draw_for_size(s) for s in sizes]
    imgs[0].save(
        path, format="ICO",
        sizes=[(s, s) for s in sizes],
        append_images=imgs[1:],
    )
    print(f"Saved {path.name}  (multi-res ICO: {sizes})")


def make_pngs(out_dir: Path):
    """PNG logo at the sizes we actually render in the app + web."""
    # 512 = master / splash / about modal source
    # 256 = README / GitHub social card
    # 128 = high-DPI taskbar / large UI
    # 64  = retina header / social fallback
    # 32  = favicon
    for size in (32, 64, 128, 256, 512):
        img  = draw_for_size(size)
        path = out_dir / ("logo.png" if size == 512 else f"logo-{size}.png")
        img.save(path, "PNG", optimize=True)
        print(f"Saved {path.name:<16s}  ({size}x{size})")


if __name__ == "__main__":
    make_icon_ico(HERE / "icon.ico")
    make_pngs(HERE)
    # Also copy 512 logo + 128 favicon to the website repo
    site = Path("C:/tts-studio-site")
    if site.is_dir():
        draw_for_size(512).save(site / "logo.png", "PNG", optimize=True)
        draw_for_size(32).save(site / "favicon.png", "PNG", optimize=True)
        print(f"Copied logo.png, favicon.png → {site}")
    print("Done.")
