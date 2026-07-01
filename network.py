import torch
import matplotlib.pyplot as plt
from torch import nn

from fdmDataGen import (
    dataset,
    n_wav,
)


learning_rate = 1e-6
batch_size = 4
epochs = 200
weight_decay = 1e-5
num_var = 4
loss_weights = torch.tensor([1.0, 1e-2, 1e-5, 1e-8], dtype=torch.float32)


def loss(pred, y):
    weights = loss_weights.to(pred.device)
    squared_error = (pred - y) ** 2
    weighted_squared_error = squared_error * weights
    return weighted_squared_error.mean()

loss_fn = loss
mae_fn = nn.L1Loss()



class NeuralNetwork(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_dim, 400),
            nn.ReLU(),
            nn.Linear(400, 400),
            nn.ReLU(),
            nn.Linear(400, 4),
        )

    def forward(self, x):
        return self.linear_relu_stack(x)


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    total_loss = 0.0
    num_batches = len(dataloader)

    for batch, (X, y) in enumerate(dataloader):
        pred = model(X)
        loss = loss_fn(pred, y)
        total_loss += loss.item()

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            print(f"loss: {loss.item():>7f}  [{batch * batch_size + len(X):>5d}/{size:>5d}]")

    return total_loss / num_batches


def test_loop(dataloader, model, loss_fn, mae):
    num_batches = len(dataloader)
    test_loss, test_mae = 0.0, 0.0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            test_mae += mae(pred, y).item()

    test_loss /= num_batches
    test_mae /= num_batches
    return test_mae, test_loss


def compute_training_medians(dataloader):
    labels = []

    for _, y in dataloader:
        labels.append(y)

    return torch.cat(labels, dim=0).median(dim=0).values


def evaluate_midpoint_sample(model, dataloader, training_medians):
    total_squared_error = torch.zeros(4)
    total_absolute_error = torch.zeros(4)
    total_samples = 0

    model.eval()
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            total_squared_error += torch.sum((pred - y) ** 2, dim=0)
            total_absolute_error += torch.sum(torch.abs(pred - y), dim=0)
            total_samples += y.shape[0]

    mean_mse = total_squared_error / total_samples
    mean_mae = total_absolute_error / total_samples
    #medians = torch.clamp(torch.abs(training_medians), min=1e-12)
    medians = torch.abs(training_medians)
    median_scaled_percent_error = 100.0 * mean_mae / medians

    print("Full evaluation mean MSE by quantity:")
    print(f"  eps_inf: {mean_mse[0].item():.6f}")
    print(f"  gamma: {mean_mse[1].item():.6f}")
    print(f"  trans_phon_frequency: {mean_mse[2].item():.6f}")
    print(f"  strength_multiple: {mean_mse[3].item():.6f}")
    print("Full evaluation mean MAE by quantity:")
    print(f"  eps_inf: {mean_mae[0].item():.6f}")
    print(f"  gamma: {mean_mae[1].item():.6f}")
    print(f"  trans_phon_frequency: {mean_mae[2].item():.6f}")
    print(f"  strength_multiple: {mean_mae[3].item():.6f}")
    print("Median-scaled percent error by quantity:")
    print(f"  eps_inf: {median_scaled_percent_error[0].item():.2f}%")
    print(f"  gamma: {median_scaled_percent_error[1].item():.2f}%")
    print(f"  trans_phon_frequency: {median_scaled_percent_error[2].item():.2f}%")
    print(f"  strength_multiple: {median_scaled_percent_error[3].item():.2f}%")


def plot_losses(train_losses, val_losses):
    epochs_axis = range(1, len(train_losses) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs_axis, train_losses, label="Train Loss")
    plt.plot(epochs_axis, val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Feedforward Network Loss vs Epoch")
    plt.legend()
    plt.tight_layout()
    plt.savefig("feedforward_loss_vs_epoch.png")
    plt.close()


def run_experiment(dataloaders):

    train_dataloader, val_dataloader, test_dataloader = dataloaders

    training_medians = compute_training_medians(train_dataloader)

    model = NeuralNetwork(n_wav * 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    final_mae, final_loss = None, None
    train_losses = []
    val_losses = []
    for e in range(epochs):
        print(f"Epoch {e + 1}")
        train_loss = train_loop(train_dataloader, model, loss_fn, optimizer)
        _, val_loss = test_loop(val_dataloader, model, loss_fn, mae_fn)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

    final_mae, final_loss = test_loop(test_dataloader, model, loss_fn, mae_fn)
    evaluate_midpoint_sample(model, test_dataloader, training_medians)
    plot_losses(train_losses, val_losses)
    return final_mae, final_loss

run_experiment(dataset)
