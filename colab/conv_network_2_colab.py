import argparse
from potato import potato
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split


DEFAULT_DATA_PATH = Path("training_data.npz")
DEFAULT_OUTPUT_DIR = Path(".")
DEFAULT_MODEL_NAME = "conv_network_2.pth"
DEFAULT_LOSS_PLOT_NAME = "conv_loss_2_vs_epoch.png"


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def cpu_state_dict(model):
    return {name: value.detach().cpu() for name, value in model.state_dict().items()}


def normalize_labels(y, label_mean, label_std):
    return (y - label_mean) / label_std


def denormalize_labels(y_normalized, label_mean, label_std):
    return y_normalized * label_std + label_mean


class ConvNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 4, 25)
        self.pool = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(4, 4, 15)
        self.conv3 = nn.Conv1d(4, 4, 5)
        self.input_dim = input_dim
        self.output_dim = output_dim
        flattened_dim = self._compute_flattened_dim(input_dim)
        self.fc1 = nn.Linear(flattened_dim, 300)
        self.fc2 = nn.Linear(300, 300)
        self.fc3 = nn.Linear(300, output_dim)

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
        return self.fc3(x)


def load_training_archive(data_path):
    archive = np.load(data_path)
    data = np.asarray(archive["data"], dtype=np.float32)
    labels = np.asarray(archive["labels"], dtype=np.float32)
    wavenumber = np.asarray(archive["wavenumber"], dtype=np.float32)

    if "parameter_names" in archive:
        parameter_names = archive["parameter_names"].astype(str).tolist()
    else:
        parameter_names = [f"parameter_{index}" for index in range(labels.shape[1])]

    expected_input_dim = wavenumber.shape[0] * 2
    if data.shape[1] != expected_input_dim:
        raise ValueError(
            f"Expected data vectors of length {expected_input_dim}, got {data.shape[1]}."
        )

    return data, labels, wavenumber, parameter_names


def build_dataloaders(data, labels, batch_size, seed):
    dataset = TensorDataset(torch.from_numpy(data), torch.from_numpy(labels))
    train_length = int(len(dataset) * 0.7)
    val_length = int(len(dataset) * 0.15)
    test_length = len(dataset) - train_length - val_length
    generator = torch.Generator().manual_seed(seed)

    train, val, test = random_split(
        dataset,
        [train_length, val_length, test_length],
        generator=generator,
    )

    return (
        DataLoader(train, batch_size=batch_size, shuffle=True, pin_memory=True),
        DataLoader(val, batch_size=batch_size, shuffle=False, pin_memory=True),
        DataLoader(test, batch_size=batch_size, shuffle=False, pin_memory=True),
    )


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


def train_loop(dataloader, model, loss_fn, optimizer, label_mean, label_std):
    device = next(model.parameters()).device
    size = len(dataloader.dataset)
    total_loss = 0.0
    model.train()

    for batch, (X, y) in enumerate(dataloader):
        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        pred = model(X)
        loss = loss_fn(pred, normalize_labels(y, label_mean, label_std))
        total_loss += loss.item()

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            processed = min((batch + 1) * dataloader.batch_size, size)
            print(f"loss: {loss.item():>7f}  [{processed:>5d}/{size:>5d}]")

    return total_loss / len(dataloader)


def test_loop(dataloader, model, loss_fn, mae_fn, label_mean, label_std):
    device = next(model.parameters()).device
    test_loss = 0.0
    test_mae = 0.0
    model.eval()

    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            pred_normalized = model(X)
            y_normalized = normalize_labels(y, label_mean, label_std)
            test_loss += loss_fn(pred_normalized, y_normalized).item()
            pred = denormalize_labels(pred_normalized, label_mean, label_std)
            test_mae += mae_fn(pred, y).item()

    return test_mae / len(dataloader), test_loss / len(dataloader)


def evaluate_model(model, dataloader, training_medians, label_mean, label_std, parameter_names):
    device = next(model.parameters()).device
    total_squared_error = torch.zeros(len(parameter_names), device=device)
    total_absolute_error = torch.zeros(len(parameter_names), device=device)
    total_samples = 0
    model.eval()

    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            pred = denormalize_labels(model(X), label_mean, label_std)
            total_squared_error += torch.sum((pred - y) ** 2, dim=0)
            total_absolute_error += torch.sum(torch.abs(pred - y), dim=0)
            total_samples += y.shape[0]

    mean_mse = total_squared_error / total_samples
    mean_mae = total_absolute_error / total_samples
    median_scaled_percent_error = 100.0 * mean_mae / torch.abs(training_medians)

    print("Full evaluation mean MSE by quantity:")
    for index, name in enumerate(parameter_names):
        print(f"  {name}: {mean_mse[index].item():.6f}")
    print("Full evaluation mean MAE by quantity:")
    for index, name in enumerate(parameter_names):
        print(f"  {name}: {mean_mae[index].item():.6f}")
    print("Median-scaled percent error by quantity:")
    for index, name in enumerate(parameter_names):
        print(f"  {name}: {median_scaled_percent_error[index].item():.2f}%")


def plot_losses(train_losses, val_losses, output_path):
    epochs_axis = range(1, len(train_losses) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs_axis, train_losses, label="Train Loss")
    plt.plot(epochs_axis, val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Convolutional Network Loss vs Epoch")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def run_experiment(args):
    data, labels, wavenumber, parameter_names = load_training_archive(args.data_path)
    train_dataloader, val_dataloader, test_dataloader = build_dataloaders(
        data,
        labels,
        args.batch_size,
        args.seed,
    )

    device = get_device()
    print(f"Using device: {device}")
    print(f"Loaded data: {data.shape[0]} samples, {data.shape[1]} inputs")
    print(f"Label parameters: {parameter_names}")

    training_medians_cpu = compute_training_medians(train_dataloader)
    label_mean_cpu, label_std_cpu = compute_label_stats(train_dataloader)
    training_medians = training_medians_cpu.to(device)
    label_mean = label_mean_cpu.to(device)
    label_std = label_std_cpu.to(device)

    model = ConvNetwork(input_dim=data.shape[1], output_dim=labels.shape[1]).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    loss_fn = nn.MSELoss()
    mae_fn = nn.L1Loss()

    train_losses = []
    val_losses = []
    for epoch in range(args.epochs):
        print(f"Epoch {epoch + 1}")
        train_loss = train_loop(
            train_dataloader,
            model,
            loss_fn,
            optimizer,
            label_mean,
            label_std,
        )
        _, val_loss = test_loop(
            val_dataloader,
            model,
            loss_fn,
            mae_fn,
            label_mean,
            label_std,
        )
        train_losses.append(train_loss)
        val_losses.append(val_loss)

    final_mae, final_loss = test_loop(
        test_dataloader,
        model,
        loss_fn,
        mae_fn,
        label_mean,
        label_std,
    )
    print(f"Final test MAE: {final_mae:.6f}")
    print(f"Final normalized test loss: {final_loss:.6f}")
    evaluate_model(
        model,
        test_dataloader,
        training_medians,
        label_mean,
        label_std,
        parameter_names,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / args.model_name
    loss_plot_path = args.output_dir / args.loss_plot_name

    plot_losses(train_losses, val_losses, loss_plot_path)
    torch.save(
        {
            "model_state_dict": cpu_state_dict(model),
            "label_mean": label_mean_cpu,
            "label_std": label_std_cpu,
            "parameter_names": parameter_names,
            "wavenumber": torch.from_numpy(wavenumber.copy()),
            "input_dim": data.shape[1],
            "output_dim": labels.shape[1],
            "train_losses": train_losses,
            "val_losses": val_losses,
        },
        model_path,
    )
    print(f"Saved model checkpoint to {model_path}")
    print(f"Saved loss plot to {loss_plot_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train conv_network_2 on Colab GPU.")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--loss-plot-name", default=DEFAULT_LOSS_PLOT_NAME)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    run_experiment(parse_args())
