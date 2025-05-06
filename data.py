from PIL import Image, ImageOps
import os

import matplotlib.pyplot as plt
import random

from tqdm import tqdm



def apply_solarize(img):
    # Lower the threshold to solarize more pixels (darker areas)
    return ImageOps.solarize(img, threshold=64)

def apply_sepia(img):
    sepia = img.convert("RGB")
    width, height = sepia.size
    pixels = sepia.load()
    for x in range(width):
        for y in range(height):
            r, g, b = pixels[x, y]
            tr = int(0.6 * r + 1.0 * g + 0.4 * b)
            tg = int(0.5 * r + 0.8 * g + 0.3 * b)
            tb = int(0.2 * r + 0.6 * g + 0.3 * b)
            pixels[x, y] = (min(tr, 255), min(tg, 255), min(tb, 255))
    return sepia

def apply_cool(img):
    # Drastically reduce red, increase blue
    r, g, b = img.split()
    r = r.point(lambda i: i * 0.6)
    b = b.point(lambda i: min(255, i * 1.4))
    return Image.merge('RGB', (r, g, b))

def apply_warm(img):
    # Drastically increase red, decrease blue
    r, g, b = img.split()
    r = r.point(lambda i: min(255, i * 100))
    b = b.point(lambda i: i * 0)
    return Image.merge('RGB', (r, g, b))

def apply_gamma(img, gamma=2.5):
    # Stronger gamma correction (darker image)
    inv_gamma = 1.0 / gamma
    table = [((i / 255.0) ** inv_gamma) * 255 for i in range(256)]
    table = table * 3  # For all RGB channels
    return img.point(table)


RAW_DOWNLOAD_DIR = "/Users/rohan/fiftyone/coco-2017/train/data"
RAW_DRIVE_DIR = "raw"
OUT_ROOT    = "filters"
FILTERS     = ["solarize","sepia","cool","warm","gamma"]


for fname in tqdm(os.listdir(RAW_DOWNLOAD_DIR)):
    img_path = os.path.join(RAW_DOWNLOAD_DIR, fname)
    try:
        img = Image.open(img_path).convert('RGB')
    except:
        continue

    for f in FILTERS:
        if f == 'solarize':
            filt_img = apply_solarize(img)
        elif f == 'sepia':
            filt_img = apply_sepia(img)
        elif f == 'cool':
            filt_img = apply_cool(img)
        elif f == 'warm':
            filt_img = apply_warm(img)
        elif f == 'gamma':
            filt_img = apply_gamma(img, gamma=1.5)

        # save filtered image as input
        in_path = os.path.join(OUT_ROOT, f, fname)
        filt_img.save(in_path, format='JPEG')

        out_path = os.path.join(RAW_DRIVE_DIR, fname)
        img.save(out_path, format='JPEG')

def show_samples():
    all_files = [f for f in os.listdir(RAW_DRIVE_DIR) if f.lower().endswith(('.jpg','.jpeg','.png'))]
    selected = random.sample(all_files, 5)

    fig, axes = plt.subplots(nrows=5, ncols=6, figsize=(18, 15))

    for i, fname in enumerate(selected):

        img = Image.open(os.path.join(RAW_DRIVE_DIR, fname))
        axes[i,0].imshow(img)
        axes[i,0].set_title('Raw')
        axes[i,0].axis('off')

        for j, filt in enumerate(FILTERS, start=1):
            filt_path = os.path.join(OUT_ROOT, filt, fname)
            filt_img = Image.open(filt_path)
            axes[i,j].imshow(filt_img)
            axes[i,j].set_title(filt.capitalize())
            axes[i,j].axis('off')

    plt.tight_layout()
    plt.show()