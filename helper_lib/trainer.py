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