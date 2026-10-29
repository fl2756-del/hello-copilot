import torch
import torch.nn as nn
import torch.optim as optim

from helper_lib.data_loader import get_mnist_loader
from helper_lib.generator import generate_samples
from helper_lib.model import get_model
from helper_lib.trainer import train_gan
from helper_lib.utils import save_model

device = "mps" if torch.backends.mps.is_available() else "cpu"  # Mac GPU if available
print(f"Using device: {device}")

train_loader = get_mnist_loader("data", batch_size=128, train=True)

gan = get_model("GAN")
criterion = nn.BCELoss()
opt_d = optim.Adam(gan.discriminator.parameters(), lr=0.0002)
opt_g = optim.Adam(gan.generator.parameters(), lr=0.0002)

trained_gan = train_gan(gan, train_loader, criterion, (opt_d, opt_g), device=device, epochs=10)
save_model(trained_gan, "models/gan.pth")

generate_samples(trained_gan, device=device, num_samples=10)