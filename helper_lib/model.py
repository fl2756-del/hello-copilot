import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18


# ---------- FCNN (from Module_3_Practical_1_FCNN notebook) ----------
class MLP(nn.Module):
    def __init__(self, num_classes=10):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 32 * 3, 200)
        self.fc2 = nn.Linear(200, 150)
        self.fc3 = nn.Linear(150, num_classes)

    def forward(self, x):
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# ---------- CNN (Assignment 2 architecture, 64x64x3 input) ----------
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(
            3, 16, kernel_size=3, stride=1, padding=1
        )  # Input channels = 3, Output channels = 16 -> 16 x 64 x 64
        self.pool = nn.MaxPool2d(
            kernel_size=2, stride=2
        )  # Pooling layer, will half the dimensions
        self.conv2 = nn.Conv2d(
            16, 32, kernel_size=3, stride=1, padding=1
        )  # Input channels = 16, Output channels = 32 -> 32 x 32 x 32
        self.fc1 = nn.Linear(32 * 16 * 16, 100)  # Fully connected layer with 100 units
        self.fc2 = nn.Linear(100, 10)  # Output layer for 10 classes

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # -> 16 x 32 x 32
        x = self.pool(F.relu(self.conv2(x)))  # -> 32 x 16 x 16
        x = x.view(-1, 32 * 16 * 16)  # Flatten -> 8192
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# ---------- EnhancedCNN (from Module_3_Practical_3_CNN notebook) ----------
class EnhancedCNN(nn.Module):
    def __init__(self):
        super(EnhancedCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)

        self.fc1 = nn.Linear(128 * 2 * 2, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # Conv -> BatchNorm -> ReLU -> Pool
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = x.view(-1, 128 * 2 * 2)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# ---------- GAN (Assignment 3 architecture, 28x28x1 MNIST) ----------
class Generator(nn.Module):
    def __init__(self, latent_dim=100):
        super(Generator, self).__init__()
        self.fc = nn.Linear(latent_dim, 7 * 7 * 128)  # noise (100) -> 7*7*128
        self.convtrans1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)  # -> 64 x 14 x 14
        self.bn1 = nn.BatchNorm2d(64)
        self.convtrans2 = nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1)    # -> 1 x 28 x 28

    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, 128, 7, 7)                    # reshape
        x = F.relu(self.bn1(self.convtrans1(x)))     # ConvTranspose -> BatchNorm -> ReLU
        x = torch.tanh(self.convtrans2(x))           # pixel values in [-1, 1]
        return x


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1)    # -> 64 x 14 x 14
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)  # -> 128 x 7 x 7
        self.bn2 = nn.BatchNorm2d(128)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(128 * 7 * 7, 1)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.2)             # Conv -> LeakyReLU(0.2)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.2)   # Conv -> BatchNorm -> LeakyReLU(0.2)
        x = self.flatten(x)
        x = torch.sigmoid(self.fc(x))                    # probability that the image is real
        return x


class GAN(nn.Module):
    # holds both networks, like VAE holds encoder + decoder in the Module 5 notebook
    def __init__(self, latent_dim=100):
        super(GAN, self).__init__()
        self.latent_dim = latent_dim
        self.generator = Generator(latent_dim)
        self.discriminator = Discriminator()


# ---------- Energy-Based Model (Module_7_Practical_1_EnergyBasedMethods notebook) ----------
def swish(x):
    return x * torch.sigmoid(x)


class EnergyModel(nn.Module):
    # notebook used MNIST (1 channel). CIFAR-10 has 3 colour channels -> in_channels=3
    def __init__(self, in_channels=3):
        super(EnergyModel, self).__init__()
        self.in_channels = in_channels
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=5, stride=2, padding=2)  # 32 -> 16
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)           # 16 -> 8
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)           # 8 -> 4
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)           # 4 -> 2

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 2 * 2, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = swish(self.conv1(x))
        x = swish(self.conv2(x))
        x = swish(self.conv3(x))
        x = swish(self.conv4(x))
        x = self.flatten(x)
        x = swish(self.fc1(x))
        return self.fc2(x)  # one energy value per image


# ---------- Diffusion Model (Module_7_Practical_2_DiffusionMethods notebook) ----------
class SinusoidalEmbedding(nn.Module):
    def __init__(self, num_frequencies=16):
        super().__init__()
        self.num_frequencies = num_frequencies
        frequencies = torch.exp(torch.linspace(math.log(1.0), math.log(1000.0), num_frequencies))
        self.register_buffer("angular_speeds", 2.0 * math.pi * frequencies.view(1, 1, 1, -1))

    def forward(self, x):
        # x: (B, 1, 1, 1) -> (B, 1, 1, 2 * num_frequencies)
        x = x.expand(-1, 1, 1, self.num_frequencies)
        sin_part = torch.sin(self.angular_speeds * x)
        cos_part = torch.cos(self.angular_speeds * x)
        return torch.cat([sin_part, cos_part], dim=-1)


def offset_cosine_diffusion_schedule(diffusion_times, min_signal_rate=0.02, max_signal_rate=0.95):
    original_shape = diffusion_times.shape
    diffusion_times_flat = diffusion_times.flatten()
    start_angle = torch.acos(torch.tensor(max_signal_rate, dtype=torch.float32, device=diffusion_times.device))
    end_angle = torch.acos(torch.tensor(min_signal_rate, dtype=torch.float32, device=diffusion_times.device))
    diffusion_angles = start_angle + diffusion_times_flat * (end_angle - start_angle)
    signal_rates = torch.cos(diffusion_angles).reshape(original_shape)
    noise_rates = torch.sin(diffusion_angles).reshape(original_shape)
    return noise_rates, signal_rates


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.needs_projection = in_channels != out_channels
        if self.needs_projection:
            self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.proj = nn.Identity()
        self.norm = nn.BatchNorm2d(in_channels, affine=False)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def swish(self, x):
        return x * torch.sigmoid(x)

    def forward(self, x):
        residual = self.proj(x)
        x = self.swish(self.conv1(x))
        x = self.conv2(x)
        return x + residual


class DownBlock(nn.Module):
    def __init__(self, width, block_depth, in_channels):
        super().__init__()
        self.blocks = nn.ModuleList()
        for i in range(block_depth):
            self.blocks.append(ResidualBlock(in_channels, width))
            in_channels = width
        self.pool = nn.AvgPool2d(kernel_size=2)

    def forward(self, x, skips):
        for block in self.blocks:
            x = block(x)
            skips.append(x)
        x = self.pool(x)
        return x


class UpBlock(nn.Module):
    def __init__(self, width, block_depth, in_channels):
        super().__init__()
        self.blocks = nn.ModuleList()
        for _ in range(block_depth):
            self.blocks.append(ResidualBlock(in_channels + width, width))
            in_channels = width

    def forward(self, x, skips):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        for block in self.blocks:
            skip = skips.pop()
            x = torch.cat([x, skip], dim=1)
            x = block(x)
        return x


class UNet(nn.Module):
    def __init__(self, image_size, num_channels, embedding_dim=32):
        super().__init__()
        self.initial = nn.Conv2d(num_channels, 32, kernel_size=1)
        self.num_channels = num_channels
        self.image_size = image_size
        self.embedding_dim = embedding_dim
        self.embedding = SinusoidalEmbedding(num_frequencies=16)
        self.embedding_proj = nn.Conv2d(embedding_dim, 32, kernel_size=1)

        self.down1 = DownBlock(32, in_channels=64, block_depth=2)
        self.down2 = DownBlock(64, in_channels=32, block_depth=2)
        self.down3 = DownBlock(96, in_channels=64, block_depth=2)

        self.mid1 = ResidualBlock(in_channels=96, out_channels=128)
        self.mid2 = ResidualBlock(in_channels=128, out_channels=128)

        self.up1 = UpBlock(96, in_channels=128, block_depth=2)
        self.up2 = UpBlock(64, block_depth=2, in_channels=96)
        self.up3 = UpBlock(32, block_depth=2, in_channels=64)

        self.final = nn.Conv2d(32, num_channels, kernel_size=1)
        nn.init.zeros_(self.final.weight)

    def forward(self, noisy_images, noise_variances):
        skips = []
        x = self.initial(noisy_images)
        noise_emb = self.embedding(noise_variances)  # (B, 1, 1, 32)
        noise_emb = F.interpolate(noise_emb.permute(0, 3, 1, 2), size=(self.image_size, self.image_size), mode='nearest')
        x = torch.cat([x, noise_emb], dim=1)

        x = self.down1(x, skips)   # 32x32 -> 16x16
        x = self.down2(x, skips)   # 16x16 -> 8x8
        x = self.down3(x, skips)   # 8x8   -> 4x4

        x = self.mid1(x)
        x = self.mid2(x)

        x = self.up1(x, skips)
        x = self.up2(x, skips)
        x = self.up3(x, skips)
        return self.final(x)


class DiffusionModel(nn.Module):
    def __init__(self, model, schedule_fn):
        super().__init__()
        self.network = model
        self.ema_network = copy.deepcopy(model)
        self.ema_network.eval()
        self.ema_decay = 0.8
        self.schedule_fn = schedule_fn
        # register_buffer (same trick SinusoidalEmbedding uses) so mean/std are saved
        # inside state_dict and our existing save_model/load_model keep working
        self.register_buffer("normalizer_mean", torch.zeros(1, model.num_channels, 1, 1))
        self.register_buffer("normalizer_std", torch.ones(1, model.num_channels, 1, 1))

    def to(self, device):
        super().to(device)
        self.ema_network.to(device)
        return self

    def set_normalizer(self, mean, std):
        self.normalizer_mean = mean
        self.normalizer_std = std

    def denormalize(self, x):
        return torch.clamp(x * self.normalizer_std + self.normalizer_mean, 0.0, 1.0)

    def denoise(self, noisy_images, noise_rates, signal_rates, training):
        if training:
            network = self.network
            network.train()
        else:
            network = self.ema_network
            network.eval()
        pred_noises = network(noisy_images, noise_rates ** 2)
        pred_images = (noisy_images - noise_rates * pred_noises) / signal_rates
        return pred_noises, pred_images

    def reverse_diffusion(self, initial_noise, diffusion_steps):
        step_size = 1.0 / diffusion_steps
        current_images = initial_noise
        for step in range(diffusion_steps):
            t = torch.ones((initial_noise.shape[0], 1, 1, 1), device=initial_noise.device) * (1 - step * step_size)
            noise_rates, signal_rates = self.schedule_fn(t)
            pred_noises, pred_images = self.denoise(current_images, noise_rates, signal_rates, training=False)
            next_diffusion_times = t - step_size
            next_noise_rates, next_signal_rates = self.schedule_fn(next_diffusion_times)
            current_images = next_signal_rates * pred_images + next_noise_rates * pred_noises
        return pred_images

    def generate(self, num_images, diffusion_steps, image_size=32, initial_noise=None):
        if initial_noise is None:
            initial_noise = torch.randn((num_images, self.network.num_channels, image_size, image_size),
                                        device=next(self.parameters()).device)
        with torch.no_grad():
            return self.denormalize(self.reverse_diffusion(initial_noise, diffusion_steps))

    def train_step(self, images, optimizer, loss_fn):
        images = (images - self.normalizer_mean) / self.normalizer_std
        noises = torch.randn_like(images)

        diffusion_times = torch.rand((images.size(0), 1, 1, 1), device=images.device)
        noise_rates, signal_rates = self.schedule_fn(diffusion_times)
        noisy_images = signal_rates * images + noise_rates * noises

        pred_noises, _ = self.denoise(noisy_images, noise_rates, signal_rates, training=True)
        loss = loss_fn(pred_noises, noises)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)  # keep training stable
        optimizer.step()

        # update the EMA copy (generate() uses ema_network)
        with torch.no_grad():
            for ema_param, param in zip(self.ema_network.parameters(), self.network.parameters()):
                ema_param.copy_(self.ema_decay * ema_param + (1.0 - self.ema_decay) * param)
            for ema_buf, buf in zip(self.ema_network.buffers(), self.network.buffers()):
                ema_buf.copy_(buf)
        return loss.item()


def get_model(model_name):
    # define and return the appropriate model_name - one of: FCNN, CNN, EnhancedCNN, GAN, EBM, Diffusion
    if model_name == "FCNN":
        model = MLP()
    elif model_name == "CNN":
        model = SimpleCNN()
    elif model_name == "EnhancedCNN":
        model = EnhancedCNN()
    elif model_name == "GAN":
        model = GAN()
    elif model_name == "EBM":
        model = EnergyModel(in_channels=3)                       # CIFAR-10: 3 x 32 x 32
    elif model_name == "Diffusion":
        unet = UNet(image_size=32, num_channels=3, embedding_dim=32)
        model = DiffusionModel(unet, offset_cosine_diffusion_schedule)
    else:
        raise ValueError(f"Invalid model name: {model_name}")
    return model
