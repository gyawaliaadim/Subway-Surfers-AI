import mss
import cv2
import numpy as np
import keyboard # To detect your game controls
import uuid # To give images unique names
from config import game_region
# Define the region of your screen to capture (X, Y, Width, Height)
# Adjust these coordinates to match where your game window sits!

current_dir = os.path.dirname(os.path.abspath(__file__))
regions_folder = os.path.join(current_dir, "..", "dataset", "roll")

def capture_and_save(action_label):
    with mss.mss() as sct:
        # 1. Grab the screenshot
        screenshot = np.array(sct.grab(game_region))
        
        # 2. Convert to Grayscale & Resize to 150x150
        gray_img = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2GRAY)
        resized_img = cv2.resize(gray_img, (150, 150))
        
        # 3. Save to the corresponding folder
        filename = f"./dataset/{action_label}/{uuid.uuid4()}.png"
        cv2.imwrite(filename, resized_img)
        print(f"Saved: {filename}")

print("Recording started... Play the game!")

# Loop to listen for keystrokes in real-time
while True:
    if keyboard.is_pressed('up'):
        capture_and_save('jump')
    elif keyboard.is_pressed('down'):
        capture_and_save('roll')
    elif keyboard.is_pressed('left'):
        capture_and_save('left')
    elif keyboard.is_pressed('right'):
        capture_and_save('right')
    # Optional: Set a key to break the loop and stop recording
    elif keyboard.is_pressed('q'):
        break