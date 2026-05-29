import mss
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from config import game_region
from PIL import Image
import keyboard
import time
import os

# 1. Must match the EXACT same architecture from your training script
class SubwaySurfersCNN(nn.Module):
    def __init__(self):
        super(SubwaySurfersCNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc1 = nn.Linear(128 * 12 * 12, 128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 5)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# 2. Setup environment and configuration

MODEL_PATH = "./models/subway_surfers_cnn.pth"

# --- AGENT CALIBRATION SETTINGS ---
CONFIDENCE_THRESHOLD = 0.98  # Requires 55% certainty to move, otherwise defaults to NOOP
ACTION_COOLDOWN = 0.5     # Increased from 0.12 to give animations more time to settle

# Action mapping index back to strings/keys
actions_map = {0: 'up', 1: 'left', 2: 'noop', 3: 'right', 4: 'down'}

# Match the exact preprocessing normalization from training
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Use GPU acceleration if available
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Loading AI brain on: {device}")

# Load the trained model weights
model = SubwaySurfersCNN().to(device)
if not os.path.exists(MODEL_PATH):
    print(f"Error: Could not find trained model file at {MODEL_PATH}. Did you train it first?")
    exit()

model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval() # Set model to evaluation mode (turns off dropout)

print("\n🤖 AI Agent is ready!")
print(f"-> Guard Parameters: Threshold={CONFIDENCE_THRESHOLD*100}%, Cooldown={ACTION_COOLDOWN}s")
print("-> Open Subway Surfers on your screen.")
print("-> Press 'P' to START the AI autopilot.")
print("-> Hold 'Q' at any time to STOP the AI.")

while True:
    if keyboard.is_pressed('p'):
        print(">> AUTOPILOT ACTIVATED! <<")
        break
    if keyboard.is_pressed('q'):
        exit()

# Small countdown so you can click back into the game window
for i in range(3, 0, -1):
    print(f"Starting in {i}...")
    time.sleep(1)

# 3. Live Inference Loop
with mss.mss() as sct:
    while True:
        # Loop break condition
        if keyboard.is_pressed('q'):
            print(">> AUTOPILOT DEACTIVATED <<")
            break
            
        # Capture the live game window frame
        screenshot = np.array(sct.grab(game_region))
        
        # Preprocess the frame to match training input layout
        gray_img = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2GRAY)
        resized_img = cv2.resize(gray_img, (96, 96))
        
        # Convert PIL Image to PyTorch Tensor format
        pil_img = Image.fromarray(resized_img)
        input_tensor = preprocess(pil_img).unsqueeze(0).to(device) # Add batch dimension: [1, 1, 96, 96]
        
        # Make the live prediction
        with torch.no_grad():
            outputs = model(input_tensor)
            
            # --- SOFTMAX CONFIDENCE FILTER ---
            # Converts the raw logits from the network into actual probabilistic percentages (0.0 to 1.0)
            probabilities = torch.softmax(outputs, dim=1)[0]
            max_prob, predicted_idx = torch.max(probabilities, 0)
            
            action = actions_map[predicted_idx.item()]
            confidence = max_prob.item()
            
        # Check if the highest predicted action passes our confidence requirement
        if confidence < CONFIDENCE_THRESHOLD:
            print(f"AI Unsure ({confidence*100:.1f}%) -> Forced NOOP")
            action = 'noop'
            
        # 4. Execute physical controls based on network output
        if action != 'noop':
            keyboard.send(action)
            print(f"AI Confident ({confidence*100:.1f}%) -> Action: {action.upper()}")
            # Hold the execution frame long enough for the lane change/jump to complete clean
            time.sleep(ACTION_COOLDOWN) 
        else:
            if confidence >= CONFIDENCE_THRESHOLD:
                print(f"AI Confident ({confidence*100:.1f}%) -> Action: NOOP (Running straight)")
            
        # Small frame rate pacing buffer
        # time.sleep(0.01)  # Adjust as needed for smoother or more responsive play
        