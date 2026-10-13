import torch
from helper_lib.model import get_model

model = get_model("CNN")
print(model)

# A fake batch with one 64x64 RGB image
x = torch.randn(1, 3, 64, 64)
print("\ninput:                    ", tuple(x.shape))

x = model.pool(torch.relu(model.conv1(x)))
print("after conv1 + relu + pool: ", tuple(x.shape))

x = model.pool(torch.relu(model.conv2(x)))
print("after conv2 + relu + pool: ", tuple(x.shape))

x = x.view(x.size(0), -1)
print("after flatten:             ", tuple(x.shape))

x = model.fc2(torch.relu(model.fc1(x)))
print("output:                    ", tuple(x.shape))

total = sum(p.numel() for p in model.parameters())
print(f"\ntotal parameters: {total:,}")