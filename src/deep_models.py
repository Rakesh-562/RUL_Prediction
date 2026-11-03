"""Deep learning models for RUL: 1D-CNN, LSTM and GRU on 30-cycle windows."""
import numpy as np
import torch
from torch import nn

from src.features import CAP


class CNN(nn.Module):
    """Two 1D convolutions slide along time and pick up local degradation patterns."""

    def __init__(self, n_sensors, window=30):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_sensors, 32, kernel_size=5, padding=2), nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2), nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Flatten(), nn.Dropout(0.2),
            nn.Linear(64 * window, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):                  # x: (batch, window, sensors)
        x = x.transpose(1, 2)              # Conv1d wants (batch, sensors, window)
        return self.head(self.conv(x)).squeeze(1)


class RNN(nn.Module):
    """LSTM or GRU: reads the window cycle by cycle and keeps a memory of what it has seen."""

    def __init__(self, n_sensors, cell="lstm", hidden=64):
        super().__init__()
        layer = nn.LSTM if cell == "lstm" else nn.GRU
        self.rnn = layer(n_sensors, hidden, num_layers=2, batch_first=True, dropout=0.2)
        self.head = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):                  # x: (batch, window, sensors)
        out, _ = self.rnn(x)
        return self.head(out[:, -1]).squeeze(1)   # use the output at the last cycle


def make_model(name, n_sensors):
    if name == "CNN":
        return CNN(n_sensors)
    return RNN(n_sensors, cell=name.lower())


def predict(model, X):
    model.eval()
    with torch.no_grad():
        y = model(torch.tensor(X)).numpy() * CAP
    return np.clip(y, 0, None)


def train(model, X_train, y_train, X_val, y_val, epochs=50, patience=8, lr=1e-3, batch_size=512, verbose=False):
    """Train with MSE loss; stop when validation RMSE hasn't improved for `patience` epochs.

    Targets are divided by CAP so the network learns values between 0 and 1.
    Returns the model with the weights from its best validation epoch.
    """
    X_train, y_train = torch.tensor(X_train), torch.tensor(y_train / CAP)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    best_rmse, best_state, bad_epochs = np.inf, None, 0
    for epoch in range(epochs):
        model.train()
        order = torch.randperm(len(X_train))
        for i in range(0, len(order), batch_size):
            idx = order[i:i + batch_size]
            optimizer.zero_grad()
            loss = loss_fn(model(X_train[idx]), y_train[idx])
            loss.backward()
            optimizer.step()

        val_rmse = np.sqrt(np.mean((predict(model, X_val) - y_val) ** 2))
        if verbose:
            print(f"epoch {epoch + 1:2d}  val RMSE {val_rmse:.2f}")
        if val_rmse < best_rmse:
            best_rmse, bad_epochs = val_rmse, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    model.load_state_dict(best_state)
    return model
