import torch
import pandas as pd
import cv2
import time
import numpy as np
import pygetwindow as gw
import mss
from PIL import Image


n_epochs = 50000

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print("Device: " , device)

df = pd.read_csv('hoops.csv', header=0)

# Scale by 1000 to keep numbers small for the math
X = torch.tensor(df[['x1', 'y1']].values, dtype=torch.float32).to(device) / 1000.0
y = torch.tensor(df[['x2', 'y2']].values, dtype=torch.float32).to(device) / 1000.0

print(f"Number of stored hoop coords: {len(X)}")
print(f"Number of stored aim points: {len(y)}")

# Model: 2 Inputs (x1, y1) -> 2 Outputs (x2, y2)
model = torch.nn.Linear(2, 2).to(device)

loss_fn = torch.nn.MSELoss().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1) # Higher LR because we scaled

for epoch in range(n_epochs):

    prediction = model(X)
    loss = loss_fn(prediction, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f"Epoch: {epoch} ; Loss: {loss.item():.6f}")

# Load hoop template image
template = cv2.imread('Images/Hoop_Bigger.png', cv2.IMREAD_COLOR)
template_h, template_w = template.shape[:2]

# Get and resize game window
window = gw.getWindowsWithTitle("Mini App: ")[0]
window.resizeTo(820, 1000)
time.sleep(0.5)

with mss.mss() as sct:
    while True:

        monitor = {
            "top": window.top,
            "left": window.left,
            "width": window.width,
            "height": window.height
        }

        # Take ss
        img = sct.grab(monitor)
        img_np = np.array(img)  # BGRA
        frame = cv2.cvtColor(img_np, cv2.COLOR_BGRA2BGR)

        # Detect hoop using template matching
        result = cv2.matchTemplate(frame, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        threshold = 0.8

        if max_val >= threshold:
            hoop_x, hoop_y = max_loc
            cv2.rectangle(frame, max_loc, (hoop_x + template_w, hoop_y + template_h), (0, 255, 0), 2)

            test_input = torch.tensor([[hoop_x / 1000.0, hoop_y / 1000.0]], dtype=torch.float32).to(device)

            model.eval()
            with torch.no_grad():
                prediction = model(test_input)

            aim_x = int(prediction[0][0].cpu().item() * 1000)
            aim_y = int(prediction[0][1].cpu().item() * 1000)

            cv2.circle(frame, (aim_x, aim_y), 5, (0, 0, 255), -1)
            cv2.putText(frame, "Aim Here", (aim_x + 10, aim_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            # debugging
            print(f"Hoop: ({hoop_x}, {hoop_y}) -> Aim: ({aim_x}, {aim_y})")

        # Show the frame
        cv2.imshow('Tracking Hoop + Predicted Aim', frame)

        # Break on ESC key
        if cv2.waitKey(1) & 0xFF == 27:
            break

        time.sleep(0.10)

cv2.destroyAllWindows()
