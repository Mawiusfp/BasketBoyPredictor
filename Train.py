import torch
import pandas as pd
import numpy as np

n_epochs = 50000

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print("Device: " , device)

df = pd.read_csv('hoops.csv', header=0)

X = torch.tensor(df[['x1', 'y1']].values, dtype=torch.float32).to(device) / 1000.0
y = torch.tensor(df[['x2', 'y2']].values, dtype=torch.float32).to(device) / 1000.0

print(f"Number of stored hoop coords: {len(X)}")
print(f"Number of stored aim points: {len(y)}")

model = torch.nn.Sequential(
    torch.nn.Linear(2, 64),
    torch.nn.ReLU(),
    torch.nn.Linear(64, 64),
    torch.nn.ReLU(),
    torch.nn.Linear(64, 2)
).to(device)

loss_fn = torch.nn.MSELoss().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

for epoch in range(n_epochs):

    prediction = model(X)
    loss = loss_fn(prediction, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 1000 == 0:
        print(f"{loss} -> {epoch}")

torch.save(model.state_dict(), 'BasketBoyPredictor.pth')
print("Model saved")