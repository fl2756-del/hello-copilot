import torch
from helper_lib.model import get_model

gan = get_model("GAN")
print(gan)

z = torch.randn(4, 100)                    # (BATCH_SIZE, 100) noise
fake = gan.generator(z)
print("generator output:    ", tuple(fake.shape))     # expect (4, 1, 28, 28)
print("discriminator output:", tuple(gan.discriminator(fake).shape))  # expect (4, 1)