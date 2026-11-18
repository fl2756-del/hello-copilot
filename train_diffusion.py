import torch
import torch.nn as nn

from helper_lib.data_loader import get_cifar_loader
from helper_lib.generator import generate_diffusion_samples
from helper_lib.model import get_model
from helper_lib.trainer import train_diffusion
from helper_lib.utils import save_model

EPOCHS = 20   # start small; raise it if you have time (the notebook used 50)

device = "mps" if torch.backends.mps.is_available() else "cpu"  # Mac GPU if available
print(f"Using device: {device}")

train_loader = get_cifar_loader("data", batch_size=64, train=True)   # pixels in [0, 1]

diffusion = get_model("Diffusion")

# per-channel normalization statistics over the training set (Module 8 notebook, cell 26)
mean = torch.zeros(3)
std = torch.zeros(3)
total_samples = 0
for imgs, _ in train_loader:
    n = imgs.size(0)
    imgs_flat = imgs.view(n, 3, -1)
    mean += imgs_flat.mean(dim=(0, 2)) * n
    std += imgs_flat.std(dim=(0, 2)) * n
    total_samples += n
mean /= total_samples
std /= total_samples
print("Normalization stats - Mean:", mean, "Std:", std)
diffusion.set_normalizer(mean.reshape(1, 3, 1, 1), std.reshape(1, 3, 1, 1))

optimizer = torch.optim.AdamW(diffusion.network.parameters(), lr=1e-3, weight_decay=1e-4)
criterion = nn.L1Loss()

trained = train_diffusion(diffusion, train_loader, criterion, optimizer, device=device, epochs=EPOCHS)
save_model(trained, "models/diffusion.pth")

generate_diffusion_samples(trained, device=device, num_samples=10, diffusion_steps=100)
