from PIL import Image, ImageDraw
import os

sizes = [16, 32, 48, 64, 128, 256]
images = []

for size in sizes:
    img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    margin = size // 10
    draw.ellipse([margin, margin, size-margin, size-margin],
                 fill=(30, 80, 160, 255))

    inner_margin = size // 5
    draw.ellipse([inner_margin, inner_margin, size-inner_margin, size-inner_margin],
                 fill=(40, 100, 200, 255))

    mic_w = size // 5
    mic_h = size // 3
    mic_x = size // 2 - mic_w // 2
    mic_y = size // 6
    draw.rounded_rectangle([mic_x, mic_y, mic_x + mic_w, mic_y + mic_h],
                            radius=mic_w // 2, fill=(255, 255, 255, 255))

    arc_margin = size // 4
    arc_top = size // 3
    arc_bottom = size * 2 // 3
    draw.arc([arc_margin, arc_top, size-arc_margin, arc_bottom],
             start=0, end=180, fill=(255, 255, 255, 220), width=max(1, size//20))

    center_x = size // 2
    draw.line([center_x, arc_bottom - size//20, center_x, size*3//4],
              fill=(255, 255, 255, 220), width=max(1, size//24))

    base_w = size // 3
    draw.line([center_x - base_w//2, size*3//4, center_x + base_w//2, size*3//4],
              fill=(255, 255, 255, 220), width=max(1, size//24))

    images.append(img)

images[0].save('icon.ico', format='ICO', sizes=[(s, s) for s in sizes], append_images=images[1:])
print('Icon created!')