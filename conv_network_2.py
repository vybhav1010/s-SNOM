import torch
import matplotlib.pyplot as plt
from torch import nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path

from data_gen import (
    NUM_PARAMETERS,
    PARAMETER_NAMES,
    load_or_create_training_data,
    n_wav,
)


learning_rate = 1e-4
batch_size = 4
epochs = 200
weight_decay = 1e-5
MODEL_PATH = Path("conv_network_2.pth")

num_var = NUM_PARAMETERS
loss_fn = nn.MSELoss()
mae_fn = nn.L1Loss()


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def cpu_state_dict(model):
    return {name: value.detach().cpu() for name, value in model.state_dict().items()}


def normalize_labels(y, label_mean, label_std):
    return (y - label_mean) / label_std


def denormalize_labels(y_normalized, label_mean, label_std):
    return y_normalized * label_std + label_mean


class ConvNetwork(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 4, 25)
        self.pool = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(4, 4, 15)
        self.conv3 = nn.Conv1d(4, 4, 5)
        self.input_dim = input_dim
        flattened_dim = self._compute_flattened_dim(input_dim)
        self.fc1 = nn.Linear(flattened_dim, 300)
        self.fc2 = nn.Linear(300, 300)
        self.fc3 = nn.Linear(300, num_var)

    def _compute_flattened_dim(self, input_dim):
        with torch.no_grad():
            example = torch.zeros(1, 1, input_dim)
            example = self.pool(F.relu(self.conv1(example)))
            example = self.pool(F.relu(self.conv2(example)))
            return int(torch.flatten(example, 1).shape[1])

    def forward(self, x):
        x = x.unsqueeze(1)

        x = self.pool(F.relu(self.conv1(x)))

        x = self.pool(F.relu(self.conv2(x)))

        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train_loop(dataloader, model, loss_fn, optimizer, label_mean, label_std):
    size = len(dataloader.dataset)
    device = next(model.parameters()).device
    model.train()
    total_loss = 0.0
    num_batches = len(dataloader)

    for batch, (X, y) in enumerate(dataloader):
        X = X.to(device)
        y = y.to(device)
        pred = model(X)
        y_normalized = normalize_labels(y, label_mean, label_std)
        loss = loss_fn(pred, y_normalized)
        total_loss += loss.item()

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            print(f"loss: {loss.item():>7f}  [{batch * batch_size + len(X):>5d}/{size:>5d}]")

    return total_loss / num_batches


def test_loop(dataloader, model, loss_fn, mae, label_mean, label_std):
    device = next(model.parameters()).device
    num_batches = len(dataloader)
    test_loss, test_mae = 0.0, 0.0

    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)
            pred_normalized = model(X)
            y_normalized = normalize_labels(y, label_mean, label_std)
            test_loss += loss_fn(pred_normalized, y_normalized).item()
            pred = denormalize_labels(pred_normalized, label_mean, label_std)
            test_mae += mae(pred, y).item()

    test_loss /= num_batches
    test_mae /= num_batches
    return test_mae, test_loss


def compute_training_medians(dataloader):
    labels = []

    for _, y in dataloader:
        labels.append(y)

    return torch.cat(labels, dim=0).median(dim=0).values


def compute_label_stats(dataloader):
    labels = []

    for _, y in dataloader:
        labels.append(y)

    all_labels = torch.cat(labels, dim=0)
    label_mean = all_labels.mean(dim=0)
    label_std = all_labels.std(dim=0).clamp_min(1e-6)
    return label_mean, label_std


def evaluate_midpoint_sample(model, dataloader, training_medians, label_mean, label_std):
    device = next(model.parameters()).device
    total_squared_error = torch.zeros(num_var, device=device)
    total_absolute_error = torch.zeros(num_var, device=device)
    total_samples = 0

    model.eval()
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)
            pred_normalized = model(X)
            pred = denormalize_labels(pred_normalized, label_mean, label_std)
            total_squared_error += torch.sum((pred - y) ** 2, dim=0)
            total_absolute_error += torch.sum(torch.abs(pred - y), dim=0)
            total_samples += y.shape[0]

    mean_mse = total_squared_error / total_samples
    mean_mae = total_absolute_error / total_samples
    #medians = torch.clamp(torch.abs(training_medians), min=1e-12)
    medians = torch.abs(training_medians)
    median_scaled_percent_error = 100.0 * mean_mae / medians

    print("Full evaluation mean MSE by quantity:")
    for index, name in enumerate(PARAMETER_NAMES):
        print(f"  {name}: {mean_mse[index].item():.6f}")
    print("Full evaluation mean MAE by quantity:")
    for index, name in enumerate(PARAMETER_NAMES):
        print(f"  {name}: {mean_mae[index].item():.6f}")
    print("Median-scaled percent error by quantity:")
    for index, name in enumerate(PARAMETER_NAMES):
        print(f"  {name}: {median_scaled_percent_error[index].item():.2f}%")


def plot_losses(train_losses, val_losses):
    epochs_axis = range(1, len(train_losses) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs_axis, train_losses, label="Train Loss")
    plt.plot(epochs_axis, val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Convolutional Network Loss vs Epoch")
    plt.legend()
    plt.tight_layout()
    plt.savefig("conv_loss_2_vs_epoch.png")
    plt.close()


def run_experiment(dataloaders):
    train_dataloader, val_dataloader, test_dataloader = dataloaders
    device = get_device()
    print(f"Using device: {device}")

    training_medians_cpu = compute_training_medians(train_dataloader)
    label_mean_cpu, label_std_cpu = compute_label_stats(train_dataloader)
    training_medians = training_medians_cpu.to(device)
    label_mean = label_mean_cpu.to(device)
    label_std = label_std_cpu.to(device)

    model = ConvNetwork(n_wav * 2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    final_mae, final_loss = None, None
    train_losses = []
    val_losses = []
    for e in range(epochs):
        print(f"Epoch {e + 1}")
        train_loss = train_loop(train_dataloader, model, loss_fn, optimizer, label_mean, label_std)
        _, val_loss = test_loop(val_dataloader, model, loss_fn, mae_fn, label_mean, label_std)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

    final_mae, final_loss = test_loop(test_dataloader, model, loss_fn, mae_fn, label_mean, label_std)
    evaluate_midpoint_sample(model, test_dataloader, training_medians, label_mean, label_std)
    plot_losses(train_losses, val_losses)
    torch.save(
        {
            "model_state_dict": cpu_state_dict(model),
            "label_mean": label_mean_cpu,
            "label_std": label_std_cpu,
        },
        MODEL_PATH,
    )
    print(f"Saved model checkpoint to {MODEL_PATH}")
    return final_mae, final_loss


if __name__ == "__main__":
    run_experiment(load_or_create_training_data())
