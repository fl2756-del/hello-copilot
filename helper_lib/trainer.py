import torch

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