import numpy as np
import torch
import pandas as pd
from torch import nn
import matplotlib.pyplot as plt

# Quantify periodicity and jitter (baseline vs failure)
df = pd.read_csv("analysis/bursts_4npu_256MB.csv")
df = df.sort_values("t_start_cycles")
df["interval"] = df["t_start_cycles"].diff()
intervals = df["interval"].dropna()
print(intervals.describe())

#plotting script for visualization
plt.figure()
plt.plot(intervals.values, marker="o")
plt.xlabel("Burst index")
plt.ylabel("Inter-burst interval (cycles)")
plt.title("Ring AllReduce inter-burst intervals (4 NPU baseline)")
plt.tight_layout()
plt.savefig("analysis/intervals_4npu_baseline.png")

# data preparation
def make_windows(series, L=8):
    x, y = [], []
    for i in range(len(series) - L):
        x.append(series[i:i+L])
        y.append(series[i+L])
    x = np.array(x, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    return torch.from_numpy(x), torch.from_numpy(y)

#LTSM predictor model
class IntervalLSTM(nn.Module):
    def __init__(self, hidden=32):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden, batch_first=True)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
        # x: (B, L)
        x = x.unsqueeze(-1)  # (B, L, 1)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(-1)


#training loop
series = intervals.values  # from pandas
x, y = make_windows(series, L=8)
train_size = int(0.8 * len(x))
x_train, y_train = x[:train_size], y[:train_size]
x_val, y_val = x[train_size:], y[train_size:]

model = IntervalLSTM()
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

for epoch in range(200):
    model.train()
    opt.zero_grad()
    pred = model(x_train)
    loss = loss_fn(pred, y_train)
    loss.backward()
    opt.step()

model.eval()
with torch.no_grad():
    val_pred = model(x_val)
    mse = loss_fn(val_pred, y_val).item()
    mae = (val_pred - y_val).abs().mean().item()
print("Val MSE:", mse, "Val MAE:", mae)
