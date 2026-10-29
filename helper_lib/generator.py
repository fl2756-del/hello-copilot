import torch
import matplotlib.pyplot as plt


def generate_samples(model, device='cpu', num_samples=10):
    model.to(device)
    model.eval()
    with torch.no_grad():
        z = torch.randn(num_samples, model.latent_dim, device=device)  # points in the latent space
        samples = model.generator(z)                                  # num_samples x 1 x 28 x 28
        samples = (samples + 1) / 2                                   # [-1, 1] -> [0, 1] for plotting
        samples = samples.cpu().numpy()

    _fig, _axes = plt.subplots(1, num_samples, figsize=(num_samples * 1.5, 1.5))
    for _i, _ax in enumerate(_axes.flat):
        _ax.imshow(samples[_i].squeeze(), cmap='gray')
        _ax.axis('off')
    plt.tight_layout()
    plt.show()
    return samples