import random

import numpy as np
import torch
import torch.nn as nn

from helper_lib.generator import langevin_generate

def train_model(model, data_loader, criterion, optimizer, device='cpu', epochs=10):
    model.to(device)
    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()              # 1. reset gradients
            outputs = model(inputs)            # 2. forward pass
            loss = criterion(outputs, labels)  # 3. compute loss
            loss.backward()                    # 4. backpropagation
            optimizer.step()                   # 5. update weights

            running_loss += loss.item() * labels.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
        print(f"Epoch {epoch + 1}/{epochs} - loss: {running_loss / total:.4f} - accuracy: {100 * correct / total:.2f}%")
    return model

def train_gan(model, data_loader, criterion, optimizer, device='cpu', epochs=10):
    # a GAN needs two optimizers: optimizer = (optimizer_D, optimizer_G)
    opt_d, opt_g = optimizer
    model.to(device)
    model.train()
    for epoch in range(epochs):
        for real_images, _ in data_loader:          # digit labels are not needed
            real_images = real_images.to(device)
            batch_size = real_images.size(0)
            real_labels = torch.ones(batch_size, 1, device=device)
            fake_labels = torch.zeros(batch_size, 1, device=device)

            # ---- 1. train the discriminator: real -> 1, fake -> 0 ----
            z = torch.randn(batch_size, model.latent_dim, device=device)
            fake_images = model.generator(z)
            opt_d.zero_grad()
            loss_real = criterion(model.discriminator(real_images), real_labels)
            loss_fake = criterion(model.discriminator(fake_images.detach()), fake_labels)
            loss_d = loss_real + loss_fake
            loss_d.backward()
            opt_d.step()

            # ---- 2. train the generator: fool D into saying fake -> 1 ----
            opt_g.zero_grad()
            loss_g = criterion(model.discriminator(fake_images), real_labels)
            loss_g.backward()
            opt_g.step()
        print(f"Epoch {epoch + 1}/{epochs} - D loss: {loss_d.item():.4f} - G loss: {loss_g.item():.4f}")
    return model


# ---------- EBM training (Module_7_Practical_1_EnergyBasedMethods notebook) ----------
class Buffer:
    def __init__(self, model, device, num_channels=3, image_size=32):
        self.model = model
        self.device = device
        self.shape = (num_channels, image_size, image_size)   # notebook: (1, 32, 32) for MNIST
        self.examples = [torch.rand((1, *self.shape), device=self.device) * 2 - 1 for _ in range(128)]

    def sample_new_exmps(self, steps, step_size, noise):
        n_new = np.random.binomial(128, 0.05)
        new_rand_imgs = torch.rand((n_new, *self.shape), device=self.device) * 2 - 1
        old_imgs = torch.cat(random.choices(self.examples, k=128 - n_new), dim=0)
        inp_imgs = torch.cat([new_rand_imgs, old_imgs], dim=0)

        new_imgs = langevin_generate(self.model, inp_imgs, steps, step_size, noise)

        self.examples = list(torch.split(new_imgs, 1, dim=0)) + self.examples
        self.examples = self.examples[:8192]
        return new_imgs


class EBM(nn.Module):
    def __init__(self, model, alpha, steps, step_size, noise, device):
        super().__init__()
        self.device = device
        self.model = model
        self.buffer = Buffer(self.model, device=device, num_channels=model.in_channels)
        self.alpha = alpha
        self.steps = steps
        self.step_size = step_size
        self.noise = noise

    def train_step(self, real_imgs, optimizer):
        real_imgs = real_imgs + torch.randn_like(real_imgs) * self.noise
        real_imgs = torch.clamp(real_imgs, -1.0, 1.0)

        fake_imgs = self.buffer.sample_new_exmps(steps=self.steps, step_size=self.step_size, noise=self.noise)

        inp_imgs = torch.cat([real_imgs, fake_imgs], dim=0)
        inp_imgs = inp_imgs.clone().detach().to(self.device).requires_grad_(False)

        out_scores = self.model(inp_imgs)
        real_out, fake_out = torch.split(out_scores, [real_imgs.size(0), fake_imgs.size(0)], dim=0)

        cdiv_loss = real_out.mean() - fake_out.mean()                               # contrastive divergence
        reg_loss = self.alpha * (real_out.pow(2).mean() + fake_out.pow(2).mean())  # keeps energies small
        loss = cdiv_loss + reg_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.1)
        optimizer.step()
        return loss.item(), cdiv_loss.item()


def train_ebm(model, data_loader, criterion, optimizer, device='cpu', epochs=10,
              alpha=0.1, steps=60, step_size=10, noise=0.005):
    # criterion is not used: the EBM loss (contrastive divergence + regularisation) is built inside EBM.train_step
    model.to(device)
    ebm = EBM(model, alpha=alpha, steps=steps, step_size=step_size, noise=noise, device=device)
    for epoch in range(epochs):
        total_loss, total_cdiv, n = 0.0, 0.0, 0
        for real_imgs, _ in data_loader:   # labels are not needed
            real_imgs = real_imgs.to(device)
            loss, cdiv = ebm.train_step(real_imgs, optimizer)
            total_loss += loss
            total_cdiv += cdiv
            n += 1
        print(f"Epoch {epoch + 1}/{epochs} - loss: {total_loss / n:.4f} - cdiv: {total_cdiv / n:.4f}")
    return model


# ---------- Diffusion training (Module 8 Activity + Module_7_Practical_2_DiffusionMethods) ----------
def train_diffusion(model, data_loader, criterion, optimizer, device='cpu', epochs=10):
    model.to(device)
    for epoch in range(epochs):
        model.train()
        train_losses = []
        for images, _ in data_loader:
            images = images.to(device)
            loss = model.train_step(images, optimizer, criterion)
            train_losses.append(loss)
        print(f"Epoch {epoch + 1}/{epochs} - loss: {sum(train_losses) / len(train_losses):.4f}")
    return model
