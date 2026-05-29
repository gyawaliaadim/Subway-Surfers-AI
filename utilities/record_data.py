import mss
import cv2
import numpy as np
import keyboard
import time
import os 
import uuid
# game_region removed since we are doing full screen, but keeping import if needed elsewhere
# from config import game_region

FRAME_DELAY = 0.05 
NOOP_DELAY = 0.25
keypress_counter = 0

key_states = {
    'left': False,
    'right': False,
    'up': False,
    'down': False,
    'space': False  
}

saved_files_log = []

# Ensure directories exist for all actions
for folder in ['left', 'right', 'jump', 'roll', 'noop']:
    os.makedirs(f"./dataset/{folder}", exist_ok=True)

print("Full-Res Color Recorder Ready...")
print("Press 'S' to START playing, 'Q' to QUIT.")
while True:
    if keyboard.is_pressed('s'):
        print(">> RECORDING STARTED! <<")
        break
    if keyboard.is_pressed('q'):
        exit()

with mss.mss() as sct:
    # Get primary monitor details for full resolution capture
    monitor = sct.monitors[1]  # 1 is usually the main monitor

    while True:
        start_time = time.time()
        should_save = False
        action = 'noop'
        
        # 1. Read key states
        current_left = keyboard.is_pressed('left') or keyboard.is_pressed('a')
        current_right = keyboard.is_pressed('right') or keyboard.is_pressed('d')
        current_up = keyboard.is_pressed('up') or keyboard.is_pressed('w')
        current_down = keyboard.is_pressed('down') or keyboard.is_pressed('s')
        current_space = keyboard.is_pressed('q')
        
        # 2. Crash / Delete logic
        if current_space and not key_states['space']:
            print("\n⚠️ Crash detected! Deleting last 2 frames...")
            for _ in range(2):
                if saved_files_log:
                    file_to_delete = saved_files_log.pop()
                    if os.path.exists(file_to_delete):
                        os.remove(file_to_delete)
                        print(f"Removed bad data: {file_to_delete}")
                else:
                    print("No more recent frames to delete in this session.")
            print("Cleanup complete. Back to recording!\n")

        # 3. Action detection
        new_press = False
        if current_left and not key_states['left']:
            action = 'left'
            new_press = True
        elif current_right and not key_states['right']:
            action = 'right'
            new_press = True
        elif current_up and not key_states['up']:
            action = 'jump'
            new_press = True
        elif current_down and not key_states['down']:
            action = 'roll'
            new_press = True

        # 4. Handle saving logic
        if new_press:
            keypress_counter += 1
            should_save = True  
        else:
            is_anyone_holding_keys = (current_left or current_right or current_up or current_down)
            if not is_anyone_holding_keys and keypress_counter >= 4:
                time.sleep(NOOP_DELAY)
                action = 'noop'
                should_save = True
                keypress_counter = 0 

        # 5. Save state for next loop
        key_states['left'] = current_left
        key_states['right'] = current_right
        key_states['up'] = current_up
        key_states['down'] = current_down
        key_states['space'] = current_space 

        # 6. Process and save to disk (Full Screen, Color, JPG)
        if should_save:
            # Capture full monitor resolution
            screenshot = np.array(sct.grab(monitor))
            
            # mss captures BGRA. Convert to standard BGR for full color OpenCV saving
            color_img = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)
            
            unique_id = str(uuid.uuid4())
            filename = f"./dataset/{action}/{unique_id}.jpg"
            
            # Saves ALL actions now, not just 'noop'
            cv2.imwrite(filename, color_img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            
            saved_files_log.append(filename)
            total_saved = len(saved_files_log)
            print(f"Captured: {action} | Total Session Frames: {total_saved} | Next Noop Progress: {keypress_counter}/4")
        
        if keyboard.is_pressed('q'):
            print(">> RECORDING STOPPED <<")
            break
            
        elapsed = time.time() - start_time
        if elapsed < FRAME_DELAY:
            time.sleep(FRAME_DELAY - elapsed)