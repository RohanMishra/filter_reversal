import torch
from cond_unet import *
from train import *
from data import *

if __name__ == "__main__":
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
