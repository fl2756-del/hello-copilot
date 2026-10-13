import torch

def evaluate_model(model, data_loader, criterion, device='cpu'):
    model.to(device)
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * labels.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    avg_loss = total_loss / total
    accuracy = 100 * correct / total
    print(f"Test loss: {avg_loss:.4f} - Test accuracy: {accuracy:.2f}%")
    return avg_loss, accuracy