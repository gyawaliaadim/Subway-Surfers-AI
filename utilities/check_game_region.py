import cv2
import os

# 1. Put the paths to your full screenshots here
image_paths = [
   
    # Add as many paths as you want
]

# Your perfect static coordinates
game_region = {'top': 360, 'left': 750, 'width': 340, 'height': 420}

# Extract values for easier cropping slices
y = game_region['top']
x = game_region['left']
h = game_region['height']
w = game_region['width']

print("Starting image loop inspection...")
print("-> Click on the image window and press ANY KEY to see the next image.")
print("-> Press 'q' to quit the loop entirely.")

for idx, path in enumerate(image_paths):
    if not os.path.exists(path):
        print(f"Skipping: File not found at {path}")
        continue
        
    # Read the original full-sized screenshot
    img = cv2.imread(path)
    
    # Crop the image using numpy slicing: img[y:y+h, x:x+w]
    cropped_img = img[y:y+h, x:x+w]
    
    # Optional: Resize to 150x150 just to see exactly what the AI will ingest
    ai_vision = cv2.resize(cropped_img, (150, 150))
    
    # Show both the raw cropped area and the downscaled version
    cv2.imshow("Cropped Game Region", cropped_img)
    cv2.imshow("What the AI Sees (150x150)", ai_vision)
    
    print(f"Showing image [{idx + 1}/{len(image_paths)}]: {path}")
    
    # Wait for a keystroke. If 'q' is pressed, exit the loop.
    key = cv2.waitKey(0) & 0xFF
    if key == ord('q'):
        print("Inspection stopped by user.")
        break

cv2.destroyAllWindows()
print("Done!")