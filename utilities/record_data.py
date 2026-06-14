import os
import time
import threading
from collections import deque
from datetime import datetime

from pynput import keyboard
import mss
from PIL import Image

# ---------------- CONFIG ---------------- #
BASE_DIR = "dataset"
ACTIONS = {
    "w": "jump",
    "a": "left",
    "s": "roll",
    "d": "right"
}
NOOP_DIR = os.path.join(BASE_DIR, "noop")

for action in list(ACTIONS.values()) + ["noop"]:
    os.makedirs(os.path.join(BASE_DIR, action), exist_ok=True)

# ---------------- STATE ---------------- #
action_count = 0
saved_files = deque()   # track all saved images for deletion

sct = mss.mss()
monitor = sct.monitors[1]  # full screen

lock = threading.Lock()

# ---------------- SCREENSHOT ---------------- #
def take_screenshot():
    img = sct.grab(monitor)
    return Image.frombytes("RGB", img.size, img.rgb)

def save_image(label):
    global saved_files

    img = take_screenshot()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    path = os.path.join(BASE_DIR, label, f"{timestamp}.jpg")
    img.save(path, "JPEG", quality=90)

    with lock:
        saved_files.append(path)

    print(f"[SAVED] {label} -> {path}")


# ---------------- NOOP HANDLER ---------------- #
def delayed_noop():
    time.sleep(0.1)
    save_image("noop")


# ---------------- DELETE FUNCTIONS ---------------- #
def delete_last():
    with lock:
        if saved_files:
            path = saved_files.pop()
            if os.path.exists(path):
                os.remove(path)
                print(f"[DELETED] {path}")

def delete_last_three():
    for _ in range(3):
        delete_last()


# ---------------- KEY HANDLER ---------------- #
def on_press(key):
    global action_count

    try:
        k = key.char.lower()
    except:
        return

    # DELETE LAST 3
    if k == "e":
        delete_last_three()
        return

    # DELETE LAST 1
    if k == "q":
        delete_last()
        return

    # ACTION KEYS
    if k in ACTIONS:
        action = ACTIONS[k]
        save_image(action)

        action_count += 1

        # every 4 actions → schedule noop
        if action_count % 4 == 0:
            threading.Thread(target=delayed_noop, daemon=True).start()


# ---------------- MAIN LOOP ---------------- #
print("Dataset recorder running...")
print("W=jump A=roll S=left D=right | Q=delete last | E=delete last 3")

with keyboard.Listener(on_press=on_press) as listener:
    listener.join()