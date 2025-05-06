import torch
from cond_unet import *
from train import *
from data import *
import fiftyone as fo
import fiftyone.zoo as foz

if __name__ == "__main__":

    RAW_DOWNLOAD_DIR = "/Users/rohan/fiftyone/coco-2017/train/data"
    RAW_DRIVE_DIR = "raw"
    OUT_ROOT = "filters"
    FILTERS = ["solarize", "sepia", "cool", "warm", "gamma"]

    foz.load_zoo_dataset(
        "coco-2017",
        split="train",
        label_types=["detections"],  # only bounding boxes, no segmentations
        classes=["person"],  # only images with at least one “person”
        max_samples=1000,  # up to 10000 samples
        only_matching=True,  # drop any images with zero “person” detections
    )

    for fname in tqdm(os.listdir(RAW_DOWNLOAD_DIR)):
        img_path = os.path.join(RAW_DOWNLOAD_DIR, fname)
        try:
            img = Image.open(img_path).convert("RGB")
        except:
            continue

        for f in FILTERS:
            if f == "solarize":
                filt_img = apply_solarize(img)
            elif f == "sepia":
                filt_img = apply_sepia(img)
            elif f == "cool":
                filt_img = apply_cool(img)
            elif f == "warm":
                filt_img = apply_warm(img)
            elif f == "gamma":
                filt_img = apply_gamma(img, gamma=1.5)

            # save filtered image as input
            in_path = os.path.join(OUT_ROOT, f, fname)
            filt_img.save(in_path, format="JPEG")

            out_path = os.path.join(RAW_DRIVE_DIR, fname)
            img.save(out_path, format="JPEG")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    in_channels = 3  # RGB images
    base_dim = 32  # U-Net “width”
    timesteps = 400

    # 2) Instantiate conditional U-Net and diffusion process
    model = ConditionalUnet(
        dim=base_dim,
        cond_channels=in_channels,  # filtered image has 3 channels
        channels=in_channels,  # noisy image also has 3 channels
    ).to(device)

    diffusion = Diffusion(timesteps=timesteps)

    train_loader, val_loader, test_loader = create_dataloaders()

    hist_loss = train_conditional(
        train_loader=train_loader,
        val_loader=val_loader,
        model=model,
        diffusion=diffusion,
        epochs=10,
        lr=1e-3,
        device=device,
        loss_type="huber",
        log_interval=1,
        sample_every=1,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    in_channels = 3  # RGB images
    base_dim = 32  # U-Net “width”
    timesteps = 400

    # 2) Instantiate conditional U-Net and diffusion process
    model = ConditionalUnet(
        dim=base_dim,
        cond_channels=in_channels,  # filtered image has 3 channels
        channels=in_channels,  # noisy image also has 3 channels
    ).to(device)

    model.load_state_dict(torch.load("ddpm_filter_reversal_10.pth"))

    diffusion = Diffusion(timesteps=timesteps)

    model.eval()
    with torch.no_grad():
        for raw, filtered in test_loader:
            raw = raw.to(device)
            filtered = filtered.to(device)

            # diffusion.sample returns a list of numpy arrays for each timestep;
            # take the last element as the final reconstruction
            samples = diffusion.sample(model, filtered)
            final_recon = torch.from_numpy(samples[-1]).to(device)
            break
