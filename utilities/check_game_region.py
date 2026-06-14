import cv2
import os
from config import game_region

# Crop region
y = game_region['top']
x = game_region['left']
h = game_region['height']
w = game_region['width']

# Direct image path (ONLY ONE IMAGE)
current_dir = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(
    current_dir,
    "..",
    "dataset",
    "roll",
    "20260614_135023_739372.jpg"
)

# Load image
img = cv2.imread(image_path)

if img is None:
    print(f"Could not load image: {image_path}")
    exit()

print("Press any key to close.")

# Crop
cropped_img = img[y:y+h, x:x+w]

# Convert to grayscale
gray_cropped = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)

# Resize for AI input
ai_vision = cv2.resize(gray_cropped, (100, 100))

# Show
cv2.imshow("Cropped B&W", gray_cropped)
cv2.imshow("AI Input (100x100)", ai_vision)

cv2.waitKey(0)
cv2.destroyAllWindows()