import cv2
import os
from config import game_region
# 1. Define the crop region we want to test

# Slicing shortcuts
y, x, h, w = game_region['top'], game_region['left'], game_region['height'], game_region['width']

# 2. Automatically point to the game_regions folder relative to this script
current_dir = os.path.dirname(os.path.abspath(__file__))
regions_folder = os.path.join(current_dir, "..", "dataset", "jump")

# Filter for files that end with image extensions
if os.path.exists(regions_folder):
    image_files = [f for f in os.listdir(regions_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    image_paths = [os.path.join(regions_folder, f) for f in image_files]
else:
    print(f"Error: Could not find folder at {regions_folder}")
    image_paths = []

if not image_paths:
    print("No images found in your game_regions folder!")
    exit()

print(f"Found {len(image_paths)} images to check in black & white.")
print("-> Click on the window and press ANY KEY to see the next frame.")
print("-> Press 'q' to exit.")

# 3. Loop through them
for idx, path in enumerate(image_paths):
    img = cv2.imread(path)
    if img is None:
        print(f"Could not load image: {path}")
        continue
        
    # Crop the raw full-size screenshot
    cropped_img = img[y:y+h, x:x+w]
    
    # --- CONVERT TO BLACK AND WHITE (GRAYSCALE) ---
    gray_cropped = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
    
    # Downscale the black and white image to 150x150
    ai_vision = cv2.resize(gray_cropped, (100,100))
    
    # Display windows
    file_name = os.path.basename(path)
    cv2.imshow(f"Testing Region (B&W): {file_name}", gray_cropped)
    cv2.imshow("Downsampled B&W (150x150)", ai_vision)
    
    key = cv2.waitKey(0) & 0xFF
    cv2.destroyAllWindows() # Cleans up the windows before showing the next one
    
    if key == ord('q'):
        print("Inspection stopped early.")
        break

print("All evaluations complete!")