# tc04/train_lstm.py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from pod_a_pipeline.burst_extractor import extract_intervals  # already in repo

WINDOW = 8
HORIZON = 3

class BurstDataset(Dataset):
    def __init__(self, series, window=WINDOW, horizon=HORIZON):
        self.series = series
        self.window = window
        self.horizon = horizon

    def __len__(self):
        return len(self.series) - self.window - self.horizon

    def __getitem__(self, idx):
        x = self.series[idx:idx+self.window]
        y = self.series[idx+self.window:idx+self.window+self.horizon]
        return torch.tensor(x).float().unsqueeze(-1), torch.tensor(y).float()

class BurstLSTM(nn.Module):
    def __init__(self, hidden=32, horizon=HORIZON):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, horizon)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

def load_series():
    base = Path("pod_a_pipeline/workloads")
    files = sorted(base.glob("*.et"))
    series = []
    for f in files:
        intervals = extract_intervals(str(f))
        series.extend(intervals)
    return np.array(series)

def main():
    series = load_series()
    split = int(0.8 * len(series))
    train, val = series[:split], series[split:]

    train_ds = BurstDataset(train)
    val_ds = BurstDataset(val)

    train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=64)

    model = BurstLSTM()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    for epoch in range(20):
        model.train()
        for x, y in train_dl:
            pred = model(x)
            loss = loss_fn(pred, y)
            opt.zero_grad()
            loss.backward()
            opt.step()

        # validation MAE
        model.eval()
        maes = []
        with torch.no_grad():
            for x, y in val_dl:
                pred = model(x)
                mae = torch.mean(torch.abs(pred - y))
                maes.append(mae.item())

        print(f"Epoch {epoch+1}: val MAE={np.mean(maes):.4f}")

    torch.save(model.state_dict(), "tc04/lstm_burst_predictor.pt")
    print("Saved model → tc04/lstm_burst_predictor.pt")

if __name__ == "__main__":
    main()
