"""Autoencoder that learns to turn raw (noisy) hand features into their smoothed versions.

Unfinished experiment: no dataset is wired in yet. X_raw and X_smooth are arrays of
shape (n_samples, 6): raw features and the matching smoothed features.
"""
import torch
import torch.nn as nn


class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(6, 4),  # input: 6 features (example), compress to 4
            nn.ReLU(),
            nn.Linear(4, 2),  # bottleneck layer
        )
        self.decoder = nn.Sequential(
            nn.Linear(2, 4),
            nn.ReLU(),
            nn.Linear(4, 6),  # reconstruct to original size
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


def train(X_raw, X_smooth, epochs=1000, lr=0.001):
    """Train an Autoencoder to map X_raw onto X_smooth and return it."""
    model = Autoencoder()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    X_raw_tensor = torch.tensor(X_raw, dtype=torch.float32)
    X_smooth_tensor = torch.tensor(X_smooth, dtype=torch.float32)

    for epoch in range(epochs):
        output = model(X_raw_tensor)
        loss = criterion(output, X_smooth_tensor)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")

    return model


def correct(model, noisy_data):
    """Run noisy features through a trained model and return the corrected values."""
    with torch.no_grad():
        return model(torch.tensor(noisy_data, dtype=torch.float32))


def main():
    print(torch.__version__)


if __name__ == "__main__":
    main()
