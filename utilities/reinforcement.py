# rl_agent.py
# Reinforcement Learning agent for Subway Surfers
# Uses Policy Gradient (REINFORCE) to improve in real-time.
# Death is detected by template-matching cropped_image.png inside DEATH_REG.
# On death: auto-clicks RESTART_POS, applies the RL update, then resumes.

import os
import time
import cv2
import mss
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from torchvision import transforms
from PIL import Image
import keyboard
import pyautogui

from config import game_region

# ---------------------------------------------------------------------------
# PATHS & REGIONS
# ---------------------------------------------------------------------------
current_dir   = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_PATH = os.path.join(current_dir, "..", "game_regions", "cropped_image.png")
MODEL_PATH    = "./models/subway_surfers_cnn.pth"
RL_MODEL_PATH = "./models/subway_surfers_rl.pth"

# Coordinate configurations given by you
DEATH_REG   = {'top': 436, 'left': 850, 'width': 48, 'height': 52}
RESTART_POS = (906, 658)

# ---------------------------------------------------------------------------
# AGENT CALIBRATION
# ---------------------------------------------------------------------------
DEATH_MATCH_THRESHOLD = 0.75   # Template-match confidence to call death (0–1)
ACTION_COOLDOWN       = 0.18   # Seconds between actions during live play
GAMMA                 = 0.99   # Discount factor for future rewards
LR                    = 1e-4   # RL fine-tune learning rate
MAX_EPISODES          = 9999   # Run until you press Q

# Reward shaping
REWARD_ALIVE   =  0.01   # Small reward each frame for surviving
REWARD_DEATH   = -1.0    # Penalty on death

# ---------------------------------------------------------------------------
# ACTION MAP  (matches training label scheme)
# ---------------------------------------------------------------------------
ACTIONS = {0: 'up', 1: 'left', 2: 'noop', 3: 'right', 4: 'down'}
N_ACTIONS = len(ACTIONS)

# ---------------------------------------------------------------------------
# PREPROCESSING  (identical to train.py so loaded weights are compatible)
# ---------------------------------------------------------------------------
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

device = torch.device(
    "cuda"  if torch.cuda.is_available()          else
    "mps"   if torch.backends.mps.is_available()  else
    "cpu"
)
print(f"RL Agent running on: {device}")


# ---------------------------------------------------------------------------
# POLICY NETWORK
# Identical CNN backbone as train.py; fc2 now outputs logits used by
# Categorical distribution for stochastic action selection.
# ---------------------------------------------------------------------------
class PolicyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2))
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2))
        self.fc1     = nn.Linear(128 * 12 * 12, 128)
        self.relu    = nn.ReLU()
        self.dropout = nn.Dropout(0.3)          # slightly relaxed for RL exploration
        self.fc2     = nn.Linear(128, N_ACTIONS)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)  # raw logits → fed into Categorical


# ---------------------------------------------------------------------------
# HELPER: load weights (RL checkpoint preferred, falls back to supervised)
# ---------------------------------------------------------------------------
def load_model():
    model = PolicyCNN().to(device)

    if os.path.exists(RL_MODEL_PATH):
        model.load_state_dict(torch.load(RL_MODEL_PATH, map_location=device))
        print(f"Loaded RL checkpoint: {RL_MODEL_PATH}")
    elif os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print(f"Loaded supervised weights: {MODEL_PATH}  (no RL checkpoint yet)")
    else:
        print("WARNING: No pre-trained weights found. Starting from random init.")

    return model


# ---------------------------------------------------------------------------
# HELPER: frame capture & preprocessing
# ---------------------------------------------------------------------------
def grab_frame(sct) -> torch.Tensor:
    """Capture game_region → grayscale 100×100 tensor ready for the model."""
    raw        = np.array(sct.grab(game_region))
    gray       = cv2.cvtColor(raw, cv2.COLOR_BGRA2GRAY)
    resized    = cv2.resize(gray, (100, 100))
    pil_img    = Image.fromarray(resized)
    tensor     = preprocess(pil_img).unsqueeze(0).to(device)
    return tensor


# ---------------------------------------------------------------------------
# HELPER: death detection via template matching
# ---------------------------------------------------------------------------
def load_death_template():
    if not os.path.exists(TEMPLATE_PATH):
        raise FileNotFoundError(
            f"Death template not found at:\n  {TEMPLATE_PATH}\n"
            "Make sure cropped_image.png exists in the game_regions folder."
        )
    tmpl = cv2.imread(TEMPLATE_PATH, cv2.IMREAD_GRAYSCALE)
    if tmpl is None:
        raise ValueError(f"cv2 could not read template: {TEMPLATE_PATH}")
    print(f"Death template loaded  ({tmpl.shape[1]}×{tmpl.shape[0]} px)")
    return tmpl


def is_dead(sct, template) -> bool:
    """
    Grabs DEATH_REG from screen, converts to grayscale, runs template match.
    Returns True when match score >= DEATH_MATCH_THRESHOLD.
    """
    region_shot = np.array(sct.grab(DEATH_REG))
    gray_region = cv2.cvtColor(region_shot, cv2.COLOR_BGRA2GRAY)

    # Template must be <= region size
    th, tw = template.shape[:2]
    rh, rw = gray_region.shape[:2]
    if tw > rw or th > rh:
        # Resize template down to fit if needed
        scale   = min(rw / tw, rh / th)
        new_w   = max(1, int(tw * scale))
        new_h   = max(1, int(th * scale))
        tmpl_rs = cv2.resize(template, (new_w, new_h))
    else:
        tmpl_rs = template

    result = cv2.matchTemplate(gray_region, tmpl_rs, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, _ = cv2.minMaxLoc(result)
    return max_val >= DEATH_MATCH_THRESHOLD


# ---------------------------------------------------------------------------
# HELPER: REINFORCE policy-gradient update
# ---------------------------------------------------------------------------
def compute_returns(rewards: list, gamma: float) -> torch.Tensor:
    """Compute discounted returns G_t = r_t + γ·r_{t+1} + … for each timestep."""
    returns = []
    G = 0.0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    returns = torch.tensor(returns, dtype=torch.float32, device=device)
    # Normalise for training stability
    if returns.std() > 1e-8:
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    return returns


def update_policy(optimizer, log_probs: list, rewards: list):
    """
    REINFORCE gradient update.
    Loss = -Σ log π(a|s) · G_t
    """
    if not log_probs:
        return 0.0

    returns = compute_returns(rewards, GAMMA)
    policy_loss = torch.stack(
        [-lp * G for lp, G in zip(log_probs, returns)]
    ).sum()

    optimizer.zero_grad()
    policy_loss.backward()
    torch.nn.utils.clip_grad_norm_(optimizer.param_groups[0]['params'], max_norm=1.0)
    optimizer.step()

    return policy_loss.item()


# ---------------------------------------------------------------------------
# HELPER: restart the game
# ---------------------------------------------------------------------------
def restart_game():
    """Click the restart button and wait for the game to reload."""
    print("  → Clicking RESTART …")
    pyautogui.click(RESTART_POS[0], RESTART_POS[1])
    time.sleep(2.0)   # Wait for the respawn animation to finish


# ---------------------------------------------------------------------------
# MAIN RL LOOP
# ---------------------------------------------------------------------------
def main():
    death_template = load_death_template()
    model          = load_model()
    model.train()   # Keep in train mode so Dropout adds exploration noise

    optimizer = optim.Adam(model.parameters(), lr=LR)

    # ── Startup prompt ──────────────────────────────────────────────────────
    print("\n🤖  RL Agent ready!")
    print(f"   Death threshold : {DEATH_MATCH_THRESHOLD}")
    print(f"   Action cooldown : {ACTION_COOLDOWN}s")
    print(f"   Discount γ      : {GAMMA}")
    print("\n→  Open Subway Surfers, then press P to START.")
    print("→  Hold Q at any time to STOP and save weights.\n")

    while True:
        if keyboard.is_pressed('p'):
            break
        if keyboard.is_pressed('q'):
            return

    print("Starting in 3 …"); time.sleep(1)
    print("Starting in 2 …"); time.sleep(1)
    print("Starting in 1 …"); time.sleep(1)
    print(">> RL AUTOPILOT ACTIVATED <<\n")

    total_steps   = 0
    best_episode  = 0

    with mss.mss() as sct:
        for episode in range(1, MAX_EPISODES + 1):

            log_probs     = []
            rewards       = []
            episode_steps = 0

            # ── Episode loop ────────────────────────────────────────────────
            while True:
                if keyboard.is_pressed('q'):
                    print(">> RL AUTOPILOT DEACTIVATED <<")
                    _save_and_exit(model, log_probs, rewards, optimizer, total_steps)
                    return

                # ── Death check ─────────────────────────────────────────────
                if is_dead(sct, death_template):
                    if rewards:
                        rewards[-1] += REWARD_DEATH

                    print(f"  ✗ DEAD  (episode {episode}, steps {episode_steps})")
                    break # end episode, trigger update + restart

                # ── Observe ─────────────────────────────────────────────────
                frame = grab_frame(sct)

                # ── Select action stochastically ────────────────────────────
                logits = model(frame)
                dist   = Categorical(logits=logits.squeeze(0))
                action_idx = dist.sample()
                log_prob   = dist.log_prob(action_idx)

                log_probs.append(log_prob)
                rewards.append(REWARD_ALIVE)

                # ── Execute action ──────────────────────────────────────────
                action_str = ACTIONS[action_idx.item()]
                if action_str != 'noop':
                    keyboard.send(action_str)

                episode_steps += 1
                total_steps   += 1
                time.sleep(ACTION_COOLDOWN)

            # ── End-of-episode: policy gradient update ──────────────────────
            loss = update_policy(optimizer, log_probs, rewards)
            survived = episode_steps
            if survived > best_episode:
                best_episode = survived

            print(
                f"Episode {episode:4d} | "
                f"Steps: {survived:5d} | "
                f"Best: {best_episode:5d} | "
                f"Loss: {loss:8.4f} | "
                f"Total steps: {total_steps}"
            )

            # Save checkpoint every 10 episodes
            if episode % 10 == 0:
                _save_weights(model)

            # ── Restart game ────────────────────────────────────────────────
            restart_game()


# ---------------------------------------------------------------------------
# SAVE HELPERS
# ---------------------------------------------------------------------------
def _save_weights(model):
    os.makedirs("./models", exist_ok=True)
    torch.save(model.state_dict(), RL_MODEL_PATH)
    print(f"  ✓ RL checkpoint saved → {RL_MODEL_PATH}")


def _save_and_exit(model, log_probs, rewards, optimizer, total_steps):
    if log_probs:
        update_policy(optimizer, log_probs, rewards)
    _save_weights(model)
    print(f"  Total steps this session: {total_steps}")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()