import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets

from helper_lib.data_loader import get_data_loader
from helper_lib.evaluator import evaluate_model
from helper_lib.model import get_model
from helper_lib.trainer import train_model
from helper_lib.utils import save_model

device = "mps" if torch.backends.mps.is_available() else "cpu"  # Mac GPU if available
print(f"Using device: {device}")

train_loader = get_data_loader("data", batch_size=64, train=True)
test_loader = get_data_loader("data", batch_size=64, train=False)

model = get_model("CNN")
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

trained_model = train_model(model, train_loader, criterion, optimizer, device=device, epochs=5)
evaluate_model(trained_model, test_loader, criterion, device=device)
save_model(trained_model, "models/cnn.pth")

# Save 10 test images so we can try the API with real CIFAR-10 pictures
raw_test = datasets.CIFAR10(root="data", train=False, download=True)
os.makedirs("samples", exist_ok=True)
for i in range(10):
    image, label = raw_test[i]
    image.save(f"samples/{i}_{raw_test.classes[label]}.png")
print("Saved 10 sample images to samples/")