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

def get_mnist_loader(data_dir, batch_size=128, train=True):
    # MNIST is 28x28x1. Normalize (0.5, 0.5) rescales [0, 1] -> [-1, 1],
    # matching the Tanh output of the generator (same as the Module 7 MNIST notebook).
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    dataset = datasets.MNIST(root=data_dir, train=train, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=train)
    return loader

def get_cifar_loader(data_dir, batch_size=128, train=True, scale_to_minus_one=False):
    # CIFAR-10 is 32x32x3. Keep the native 32x32 size (no Resize).
    # Diffusion: keep pixels in [0, 1] (the DiffusionModel normalizes itself, like the Module 8 notebook).
    # EBM: rescale to [-1, 1] with Normalize(0.5, 0.5) (the energy model and Langevin sampler assume [-1, 1]).
    steps = [transforms.ToTensor()]
    if scale_to_minus_one:
        steps.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
    transform = transforms.Compose(steps)
    dataset = datasets.CIFAR10(root=data_dir, train=train, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=train)
    return loader
