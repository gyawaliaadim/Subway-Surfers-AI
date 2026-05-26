import mss
import cv2
import numpy as np

print("Select your game window: ")
print("1. A screenshot window will pop up.")
print("2. Click and drag a box over your Subway Surfers game window.")
print("3. Press 'ENTER' or 'SPACE' to confirm, or 'c' to cancel.")

with mss.mss() as sct:
    # 1. Capture the entire primary monitor
    monitor = sct.monitors[1]
    screenshot = np.array(sct.grab(monitor))
    
    # Convert from BGRA to BGR for OpenCV
    screenshot = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)
    
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