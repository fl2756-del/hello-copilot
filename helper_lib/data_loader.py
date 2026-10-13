from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get_data_loader(data_dir, batch_size=32, train=True):
    # CIFAR-10 images are 32x32, but our CNN expects 64x64 -> resize first
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])
    dataset = datasets.CIFAR10(root=data_dir, train=train, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=train)
    return loader