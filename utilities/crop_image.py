import cv2
import os
# Load the original image
current_dir = os.path.dirname(os.path.abspath(__file__))
regions_folder = os.path.join(current_dir, "..", "game_regions", "endscreen.png")
image = cv2.imread(regions_folder)
cv2.imshow("Original Image", image)
cv2.waitKey(0)
# Region to crop
DEATH_REG = {
    'top': 436,
    'left': 850,
    'width': 48,
    'height': 52
}

# Extract values
x = DEATH_REG['left']
y = DEATH_REG['top']
w = DEATH_REG['width']
h = DEATH_REG['height']

# Crop the image
cropped = image[y:y+h, x:x+w]

# Save cropped image
cv2.imwrite("cropped_image.png", cropped)

print("Cropped image saved as cropped_image.png")