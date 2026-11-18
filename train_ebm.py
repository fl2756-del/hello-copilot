import torch

from helper_lib.data_loader import get_cifar_loader
from helper_lib.generator import generate_ebm_samples
from helper_lib.model import get_model
from helper_lib.trainer import train_ebm
from helper_lib.utils import save_model

EPOCHS = 5   # each batch runs 60 Langevin steps, so this is slow; start with 1 to time it

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

train_loader = get_cifar_loader("data", batch_size=128, train=True, scale_to_minus_one=True)  # pixels in [-1, 1]

energy_model = get_model("EBM")
optimizer = torch.optim.Adam(energy_model.parameters(), lr=0.0001, betas=(0.0, 0.999))

trained = train_ebm(energy_model, train_loader, None, optimizer, device=device, epochs=EPOCHS,
                    alpha=0.1, steps=60, step_size=10, noise=0.005)
save_model(trained, "models/ebm.pth")

generate_ebm_samples(trained, device=device, num_samples=10, steps=256, step_size=10.0, noise_std=0.01)
