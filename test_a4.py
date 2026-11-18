import torch
from helper_lib.model import get_model
from helper_lib.generator import langevin_generate

ebm = get_model("EBM")
x = torch.rand(4, 3, 32, 32) * 2 - 1
print("EBM energy:", tuple(ebm(x).shape))                      # expect (4, 1)
out = langevin_generate(ebm, x, steps=3, step_size=10.0, noise_std=0.01)
print("Langevin output:", tuple(out.shape))                   # expect (4, 3, 32, 32)

diffusion = get_model("Diffusion")
noisy = torch.randn(4, 3, 32, 32)
var = torch.rand(4, 1, 1, 1)
print("UNet output:", tuple(diffusion.network(noisy, var).shape))   # expect (4, 3, 32, 32)
imgs = diffusion.generate(num_images=2, diffusion_steps=5)
print("Diffusion generate:", tuple(imgs.shape))                     # expect (2, 3, 32, 32)
