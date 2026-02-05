import torch
import pandas as pd
import cv2
import time
import numpy as np
import pygetwindow as gw
import mss
from PIL import Image

def is_roughly_the_same(a, b, tolerance=5):
    return abs(a - b) <= tolerance

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using: ", device)

template = cv2.imread('Images/Hoop_Bigger.png', cv2.IMREAD_COLOR)
template_h, template_w = template.shape[:2]

window = gw.getWindowsWithTitle("Mini App: ")[0]
window.resizeTo(820, 1000)
time.sleep(0.5)

model = torch.nn.Sequential(
    torch.nn.Linear(2, 64),
    torch.nn.ReLU(),
    torch.nn.Linear(64, 64),
    torch.nn.ReLU(),
    torch.nn.Linear(64, 2)
).to(device)

model.load_state_dict(torch.load('BasketBoyPredictor.pth', map_location=device))
model.eval()

with mss.mss() as sct:

    current_hoop = [0,0]

    while True:
        monitor = {
            "top": window.top,
            "left": window.left,
            "width": window.width,
            "height": window.height
        }

        img = sct.grab(monitor)
        img_np = np.array(img)
        frame = cv2.cvtColor(img_np, cv2.COLOR_BGRA2BGR)

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

            x_moved = not is_roughly_the_same(current_hoop[0], hoop_x, tolerance=5)
            y_moved = not is_roughly_the_same(current_hoop[1], hoop_y, tolerance=5)

            if x_moved or y_moved:
                
                print(f"{hoop_x}, {hoop_y}, {aim_x}, {aim_y}")
                
                current_hoop = [hoop_x, hoop_y]

        cv2.imshow('Tracking Hoop + Predicted Aim', frame)

        # Break on ESC key
        if cv2.waitKey(1) & 0xFF == 27:
            break

        time.sleep(0.10)

cv2.destroyAllWindows()
