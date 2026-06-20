"""
Interactive ROI (Region of Interest) selector for defining the game capture area.

Loads a reference image and allows the user to manually draw a bounding box
over the Subway Surfers gameplay region using OpenCV's GUI selection tool.
The selected coordinates (top, left, width, height) are then formatted into
a `game_region` dictionary compatible with MSS-based screen capture scripts.

This is used to precisely calibrate the area of the screen that should be
captured for dataset collection or real-time AI inference.
"""

import os

import mss
import cv2
import numpy as np

print("Select your game window: ")
print("1. A screenshot window will pop up.")
print("2. Click and drag a box over your Subway Surfers game window.")
print("3. Press 'ENTER' or 'SPACE' to confirm, or 'c' to cancel.")

current_dir = os.path.dirname(os.path.abspath(__file__))
# regions_folder = os.path.join(current_dir, "..", "game_regions", "left top.png")
regions_folder = os.path.join(current_dir, "..", "dataset", "roll", "0b36925c-838c-4529-8508-523af4a1f626.jpg")

with mss.mss() as sct:
    # 1. Capture the entire primary monitor
    img = cv2.imread(regions_folder)
    # Convert from BGRA to BGR for OpenCV
    screenshot = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    
    # 2. Open the interactive selection window
    # (Drag your mouse to select the game area)
    roi = cv2.selectROI("Drag a box over Subway Surfers & press ENTER", screenshot, fromCenter=False, showCrosshair=True)
    
    # 3. Extract coordinates (X, Y, Width, Height)
    x, y, w, h = roi
    
    # Close the selector window immediately
    cv2.destroyAllWindows()
    
    if w > 0 and h > 0:
        # 4. Format it exactly for your MSS script
        game_region = {
            "top": int(y),
            "left": int(x),
            "width": int(w),
            "height": int(h)
        }
        print("\n" + "="*40)
        print("COPY THIS INTO YOUR MAIN SCRIPT:")
        print("="*40)
        print(f"game_region = {game_region}")
        print("="*40)
    else:
        print("Selection cancelled.")