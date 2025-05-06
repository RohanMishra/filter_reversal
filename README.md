# Filter Reversal with Conditional Diffusion Models

This project implements a conditional Denoising Diffusion Probabilistic Model (DDPM) to reverse color transformations applied by photographic filters. The full pipeline includes data preparation, model training, and testing.

## How to Run

Before running the code, you must first specify a directory path where the dataset will be downloaded and processed.

**Set the data download directory:**

Open `main.py` and modify the following line to point from your home directory:

   ```python
   RAW_DOWNLOAD_DIR = "/YOUR_HOME_DIR/fiftyone/coco-2017/train/data"
```

Then simply execute ```main.py```

Please note that it begins by training all the data to a local path, and then the entire training/testing pipeline takes ~10 hours.

## Dataset:

This project utilizes the Microsoft Common Objects in Context (MS COCO) dataset, a large-scale dataset designed for object detection, segmentation, and captioning tasks. COCO contains over 330,000 images, each annotated with 80 object categories and multiple captions, providing a diverse and comprehensive resource for training and evaluating computer vision models.

https://cocodataset.org/#home
