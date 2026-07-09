"""Generate DEV-badged icon + logo for running from source.

Overlays a bright orange "DEV" banner on the bottom of the production
artwork (drawn by _make_logo.draw_icon), so a from-source run is
instantly distinguishable from the installed app in the taskbar,
window chrome, and splash.

Outputs (committed to the repo — regenerate only when the logo changes):
    icon_dev.ico   16/20/24/32/40/48/64/96/128/256
    logo_dev.png   512x512

Sizes below 24 px get a plain banner with no text — "DEV" isn't legible
at that scale, but the orange stripe still reads clearly.
"""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

from _make_logo import draw_icon

HERE = Path(__file__).parent

BADGE       = (255, 149, 0, 255)    # orange — high contrast vs. dark/green logo
BADGE_TEXT  = (20, 12, 0, 255)      # near-black text on orange

_FONT_CANDIDATES = [
    "C:/Windows/Fonts/arialbd.ttf",
    "C:/Windows/Fonts/segoeuib.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
]


def _font(px: int):
    for cand in _FONT_CANDIDATES:
        try:
            return ImageFont.truetype(cand, px)
        except OSError:
            continue
    return ImageFont.load_default()


def badge_dev(img: Image.Image) -> Image.Image:
    """Return a copy of img with an orange DEV band across the bottom.

    The band is clipped to the icon's opaque region so it follows the
    rounded-square silhouette (and never bleeds into the soft glow that
    large sizes draw around the tile).
    """
    img = img.convert("RGBA")
    size = img.width

    # Solid silhouette of the tile (ignore low-alpha glow pixels)
    solid = img.getchannel("A").point(lambda a: 255 if a > 200 else 0)
    bbox = solid.getbbox()
    if bbox is None:                       # blank image — nothing to badge
        return img
    left, top, right, bottom = bbox

    tile_h = bottom - top
    band_h = max(4, round(tile_h * 0.34))
    band_top = bottom - band_h

    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)
    d.rectangle([left, band_top, right - 1, bottom - 1], fill=BADGE)

    if size >= 24:
        text_px = max(6, round(band_h * 0.78))
        f = _font(text_px)
        cx = (left + right) // 2
        cy = band_top + band_h // 2
        d.text((cx, cy), "DEV", font=f, fill=BADGE_TEXT, anchor="mm")

    # Clip the band to the tile silhouette, then composite over the icon.
    overlay.putalpha(Image.composite(
        overlay.getchannel("A"), Image.new("L", img.size, 0), solid))
    return Image.alpha_composite(img, overlay)


def make_dev_icon_ico(path: Path):
    sizes = [16, 20, 24, 32, 40, 48, 64, 96, 128, 256]
    imgs = [badge_dev(draw_icon(s)) for s in sizes]
    # Pillow drops any ICO size larger than the base image, so the largest
    # frame must be the one .save() is called on.
    imgs[-1].save(
        path, format="ICO",
        sizes=[(s, s) for s in sizes],
        append_images=imgs[:-1],
    )
    print(f"Saved {path.name}  (ICO: {sizes})")


def make_dev_logo(path: Path):
    badge_dev(draw_icon(512)).save(path, "PNG", optimize=True)
    print(f"Saved {path.name}  (512x512)")


if __name__ == "__main__":
    make_dev_icon_ico(HERE / "icon_dev.ico")
    make_dev_logo(HERE / "logo_dev.png")
    print("Done.")
