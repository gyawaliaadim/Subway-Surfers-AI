import mss
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import keyboard
import pyautogui
import time
import os
import random
from config import game_region  # Raw game view

# 1. Coordinate configurations given by you
DEATH_REG = {'top': 436, 'left': 850, 'width': 48, 'height': 52}
RESTART_POS = (906, 658)
MODEL_PATH = "./models/subway_surfers_rl.pth"

# Actions map
actions_map = {0: 'up', 1: 'left', 2: 'noop', 3: 'right', 4: 'down'}
NUM_ACTIONS = len(actions_map)

# Load the death template snippet for matching
TEMPLATE_PATH = "./utilities/death_template.png"
if not os.path.exists(TEMPLATE_PATH):
    print(f"⚠️ Warning: Missing {TEMPLATE_PATH}. Death detection might fail until provided.")
    death_template = None
else:
    death_template = cv2.imread(TEMPLATE_PATH, cv2.IMREAD_GRAYSCALE)

# 2. Deep Q-Network (DQN) Architecture
class SubwaySurfersDQN(nn.Module):
    def __init__(self):
        super(SubwaySurfersDQN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2), # 100x100 -> 50x50
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2), # 50x50 -> 25x25
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2) # 25x25 -> 12x12
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * 12 * 12, 256),
            nn.ReLU(),
            nn.Linear(256, NUM_ACTIONS) # Outputs expected future reward (Q-value) for each action
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# 3. Helper Functions for Environment States
def check_death(sct):
    """Grabs the tiny death sub-region and looks for the template match."""
    if death_template is None:
        return False
    
    # Grab the precise 48x52 region
    reg_img = np.array(sct.grab(DEATH_REG))
    gray_reg = cv2.cvtColor(reg_img, cv2.COLOR_BGRA2GRAY)
    
    # Match template
    res = cv2.matchTemplate(gray_reg, death_template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, _ = cv2.minMaxLoc(res)
    
    # Threshold of 0.85 means 85% visual similarity match
    return max_val > 0.85

def get_state(sct):
    """Captures and resizes game view to 100x100 normalized tensor."""
    screenshot = np.array(sct.grab(game_region))
    gray_img = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2GRAY)
    resized_img = cv2.resize(gray_img, (100, 100))
    tensor_img = torch.FloatTensor(resized_img).unsqueeze(0).unsqueeze(0) / 255.0 # Shape: [1, 1, 100, 100]
    return tensor_img

# 4. Main RL Training & Play Loop
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    
    policy_net = SubwaySurfersDQN().to(device)
    if os.path.exists(MODEL_PATH):
        policy_net.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print("Loaded existing model weights. Continuing training...")
    
    optimizer = optim.Adam(policy_net.parameters(), lr=0.00025)
    criterion = nn.MSELoss()
    
    # Hyperparameters
    GAMMA = 0.99       # Discount factor for future rewards
    EPSILON = 0.1       # 10% chance to take completely random actions to explore new paths
    
    print("\n🎮 RL Autopilot Ready. Press 'P' to engage, 'Q' to quit.")
    while True:
        if keyboard.is_pressed('p'): break
        if keyboard.is_pressed('q'): exit()
        
    time.sleep(2)
    
    with mss.mss() as sct:
        episode = 1
        while True:
            if keyboard.is_pressed('q'): break
            
            print(f"\n--- Starting Episode {episode} ---")
            state = get_state(sct).to(device)
            is_dead = False
            score = 0
            
            while not is_dead:
                if keyboard.is_pressed('q'): break
                
                # Epsilon-Greedy action choices
                if random.random() < EPSILON:
                    action_idx = random.randint(0, NUM_ACTIONS - 1)
                else:
                    with torch.no_grad():
                        action_idx = policy_net(state).argmax().item()
                
                # Execute physical choice
                action_str = actions_map[action_idx]
                if action_str != 'noop':
                    keyboard.send(action_str)
                    time.sleep(0.15) # Brief cooldown for animation
                
                # Observe new environment state
                is_dead = check_death(sct)
                next_state = get_state(sct).to(device)
                
                # --- REWARD CALCULATIONS ---
                if is_dead:
                    reward = -100.0  # Punishment for death screen
                else:
                    reward = 1.0     # Incentive reward for staying alive
                    score += 1
                
                # Train the model network inline (Temporal Difference learning)
                q_values = policy_net(state)
                next_q_values = policy_net(next_state)
                
                # Calculate target Q value
                target_q = q_values.clone().detach()
                if is_dead:
                    target_q[0][action_idx] = reward
                else:
                    target_q[0][action_idx] = reward + GAMMA * torch.max(next_q_values).item()
                
                # Backpropagation update step
                loss = criterion(q_values, target_q)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                state = next_state
                print(f"Action: {action_str.upper()} | Frame Reward: {reward} | Total Survival Frames: {score}", end="\r")
            
            # --- DEATH SCREEN DETECTED: HANDSHAKE RESTART ---
            print(f"\n💀 Dead Screen Flagged! Final Score: {score} frames survived.")
            print(f"Clicking anywhere near restart button position: {RESTART_POS}")
            
            # Use pyautogui to instantly click restart
            pyautogui.click(x=RESTART_POS[0], y=RESTART_POS[1])
            time.sleep(3.0)  # Heavy wait buffer for load screens/run animations to clear
            
            # Periodically write out weights backup files
            if episode % 5 == 0:
                os.makedirs("./models", exist_ok=True)
                torch.save(policy_net.state_dict(), MODEL_PATH)
                print(">> Auto-saved fresh weights optimization checkpoint to disk. <<")
                
            episode += 1