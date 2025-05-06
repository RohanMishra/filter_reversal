import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import numpy as np

import os
import glob
from PIL import Image
from torch.utils.data import random_split, DataLoader

from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.fid import FrechetInceptionDistance


def p_losses(diffusion, model, x_start, cond_img, t, noise=None, loss_type="huber"):
    """
    Compute the denoising loss for a batch of (raw, filtered) images.
    """
    if noise is None:
        noise = torch.randn_like(x_start)
    # forward noising step
    x_noisy = diffusion.q_sample(x_start, t, noise)
    # predict the noise given the noisy image and the filter condition
    pred_noise = model(x_noisy, t, cond_img)
    # compute chosen loss
    if loss_type == "l1":
        return F.l1_loss(pred_noise, noise)
    elif loss_type == "l2":
        return F.mse_loss(pred_noise, noise)
    elif loss_type == "huber":
        return F.smooth_l1_loss(pred_noise, noise)
    else:
        raise ValueError(f"Unknown loss_type {loss_type!r}")


# --- 2) Training function ---
def train_conditional(
    train_loader,
    val_loader,
    model,
    diffusion,
    epochs: int,
    lr: float,
    device,
    loss_type: str = "huber",
    log_interval: int = 20,
    sample_every: int = 20,
):
    """
    Trains a conditional DDPM to reverse image filters.
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.to(device)

    # psnr_metric = PeakSignalNoiseRatio().to(device)
    # ssim_metric = StructuralSimilarityIndexMeasure().to(device)

    hist_loss = []
    # hist_psnr = []
    # hist_ssim = []

    for epoch in range(1, epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"[Train] Epoch {epoch}/{epochs}")

        for raw, filtered in pbar:
            raw, filtered = raw.to(device), filtered.to(device)
            # sample timesteps
            t = torch.randint(0, diffusion.timesteps, (raw.size(0),), device=device)
            loss = p_losses(diffusion, model, raw, filtered, t, loss_type=loss_type)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix(train_loss=loss.item())

            hist_loss.append(loss.item())

        torch.save(model.state_dict(), f"ddpm_filter_reversal_{epoch}.pth")
        print("Model saved to ddpm_filter_reversal.pth")

        # validation & logging

        if epoch % log_interval == 0 or epoch == epochs:

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for raw, filtered in val_loader:
                    raw, filtered = raw.to(device), filtered.to(device)
                    t = torch.randint(0, diffusion.timesteps, (raw.size(0),), device=device)
                    val_loss += p_losses(diffusion, model, raw, filtered, t, loss_type=loss_type).item()

            val_loss /= len(val_loader)
            print(f"Epoch {epoch} — Val Loss: {val_loss:.4f}")

        # sampling demonstration
        if epoch % sample_every == 0 or epoch == epochs:
            # psnr_metric.reset()
            # ssim_metric.reset()

            # take one batch of conditions
            demo_raw, demo_filtered = next(iter(val_loader))
            demo_filtered = demo_filtered.to(device)[:1]  # single example
            with torch.no_grad():
                frames = diffusion.sample(model, demo_filtered)
                # frames is a list of numpy arrays, length = timesteps
            # pick a few time indices to visualize
            time_idxs = [0, diffusion.timesteps // 4, diffusion.timesteps // 2, diffusion.timesteps - 1]
            # compute PSNR and SSIM for the reconstructed images
            # psnr_metric.update(frames[-1][0], demo_raw[0])
            # ssim_metric.update(frames[-1][0], demo_raw[0])
            fig, axes = plt.subplots(1, len(time_idxs), figsize=(len(time_idxs) * 3, 3))
            for ax, ti in zip(axes, time_idxs):
                img = frames[ti][0]  # shape (C, H, W)
                img = np.transpose(img, (1, 2, 0))  # (H, W, C)
                ax.imshow(np.clip(img, 0, 1))
                ax.set_title(f"t={ti}")
                ax.axis("off")
            plt.tight_layout()
            plt.show()
            # epoch_psnr = psnr_metric.compute().item()
            # epoch_ssim = ssim_metric.compute().item()
            # hist_psnr.append(epoch_psnr)
            # hist_ssim.append(epoch_ssim)

    # save final model

    return hist_loss


def create_dataloaders():
    img_transform = transforms.Compose(
        [transforms.Resize(IMG_SIZE), transforms.CenterCrop(IMG_SIZE), transforms.ToTensor()]
    )

    # === LOAD ALL DATA INTO MEMORY AS TUPLES ===
    def load_all_to_memory(raw_dir, filt_base, filters, image_size=128, max_images=None):
        all_data = []
        for filt_name in filters:
            raw_paths = sorted(glob.glob(os.path.join(raw_dir, "*.jpg")))
            filt_paths = {os.path.basename(p): p for p in glob.glob(os.path.join(filt_base, filt_name, "*.jpg"))}

            for raw_path in tqdm(raw_paths, desc=f"Loading {filt_name}"):
                name = os.path.basename(raw_path)
                if name not in filt_paths:
                    continue
                try:
                    raw_img = img_transform(Image.open(raw_path).convert("RGB"))
                    filt_img = img_transform(Image.open(filt_paths[name]).convert("RGB"))
                    all_data.append((filt_img, raw_img))  # (condition, target)
                except Exception as e:
                    print(f"Failed to load {name}: {e}")
                if max_images and len(all_data) >= max_images:
                    return all_data
        return all_data

    # === MAKE LOADER DIRECTLY FROM RAW LIST ===
    def make_loader_from_raw_list(raw_dir, filt_dir, filters, batch_size, max_images=None):
        all_data = load_all_to_memory(raw_dir, filt_dir, filters, IMG_SIZE, max_images)

        # Use random_split on list directly (DataLoader accepts lists)
        train_size = int(0.8 * len(all_data))
        test_size = int(0.1 * len(all_data))
        val_size = len(all_data) - train_size - test_size

        train_set, test_set, val_set = random_split(all_data, [train_size, test_size, val_size])

        return (
            DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=10),
            DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=10),
            DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=10),
        )

    # === USAGE ===
    return make_loader_from_raw_list(RAW_DRIVE_DIR, "filters", FILTERS, BATCH)
