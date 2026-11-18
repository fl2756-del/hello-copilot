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


# ---------- EBM: Langevin dynamics (notebook's generate_samples, renamed so it doesn't clash with the GAN one) ----------
def langevin_generate(nn_energy_model, inp_imgs, steps, step_size, noise_std):
    nn_energy_model.eval()
    for _ in range(steps):
        with torch.no_grad():
            noise = torch.randn_like(inp_imgs) * noise_std
            inp_imgs = (inp_imgs + noise).clamp(-1.0, 1.0)

        inp_imgs.requires_grad_(True)
        energy = nn_energy_model(inp_imgs)
        # gradient of the energy with respect to the INPUT image (not the weights)
        grads, = torch.autograd.grad(energy, inp_imgs, grad_outputs=torch.ones_like(energy))

        with torch.no_grad():
            grads = grads.clamp(-0.03, 0.03)
            inp_imgs = (inp_imgs - step_size * grads).clamp(-1.0, 1.0)
    return inp_imgs.detach()


def _plot_row(samples):
    num_samples = samples.shape[0]
    _fig, _axes = plt.subplots(1, num_samples, figsize=(num_samples * 1.5, 1.5))
    for _i, _ax in enumerate(_axes.flat):
        _ax.imshow(samples[_i].permute(1, 2, 0).numpy())  # CHW -> HWC for colour images
        _ax.axis('off')
    plt.tight_layout()
    plt.show()


def generate_ebm_samples(model, device='cpu', num_samples=10, steps=256, step_size=10.0, noise_std=0.01):
    model.to(device)
    x = torch.rand((num_samples, model.in_channels, 32, 32), device=device) * 2 - 1  # uniform noise in [-1, 1]
    samples = langevin_generate(model, x, steps, step_size, noise_std)
    samples = ((samples + 1) / 2).clamp(0, 1).cpu()  # [-1, 1] -> [0, 1]
    _plot_row(samples)
    return samples


def generate_diffusion_samples(model, device='cpu', num_samples=10, diffusion_steps=100):
    # start from standard normal noise, run reverse diffusion, plot the samples
    model.to(device)
    model.eval()
    samples = model.generate(num_images=num_samples, diffusion_steps=diffusion_steps,
                             image_size=model.network.image_size).cpu()  # already in [0, 1]
    _plot_row(samples)
    return samples
