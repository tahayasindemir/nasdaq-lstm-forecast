import torch
from torch.utils.data import TensorDataset, DataLoader


def prepare_dataloader(X, y, batch_size, device):
    """
    Convert numpy arrays to a PyTorch DataLoader.

    Args:
        X (np.ndarray): Input features.
        y (np.ndarray): Target values.
        batch_size (int): Batch size.
        device (str): 'cuda' or 'cpu'.

    Returns:
        DataLoader
    """
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1).to(device)
    dataset = TensorDataset(X_tensor, y_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def train_one_epoch(model, dataloader, criterion, optimizer):
    model.train()
    total_loss = 0.0

    for X_batch, y_batch in dataloader:
        optimizer.zero_grad()
        output = model(X_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate(model, dataloader, criterion):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            output = model(X_batch)
            loss = criterion(output, y_batch)
            total_loss += loss.item()

    return total_loss / len(dataloader)
