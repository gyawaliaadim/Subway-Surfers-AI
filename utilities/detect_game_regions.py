"""
End-screen detection of subway surfers using template matching.

Loads a reference image (`cropped_image.png`) and compares it against
a specific region of a game screenshot (`endscreen.png`). The selected
region is cropped, resized if necessary, and matched using OpenCV's
normalized correlation template matching. If the similarity score
exceeds the defined threshold, the script reports a successful match.
"""

import cv2
import numpy as np
import os

# =========================
# INPUTS
# =========================

current_dir = os.path.dirname(os.path.abspath(__file__))
img1_path = os.path.join(current_dir, "..", "game_regions", "cropped_image.png")
img2_path = os.path.join(current_dir, "..", "game_regions", "endscreen.png")

REGION = {
    'top': 436,
    'left': 850,
    'width': 48,
    'height': 52
}

# Similarity threshold
THRESHOLD = 0.99

# =========================
# LOAD IMAGES
# =========================

img1 = cv2.imread(img1_path)
img2 = cv2.imread(img2_path)

if img1 is None or img2 is None:
    print("Error loading images")
    exit()

# =========================
# CROP REGION FROM IMG2
# =========================

x = REGION['left']
y = REGION['top']
w = REGION['width']
h = REGION['height']

region_crop = img2[y:y+h, x:x+w]

# Resize img1 if needed
if img1.shape[:2] != region_crop.shape[:2]:
    img1 = cv2.resize(img1, (w, h))

# =========================
# TEMPLATE MATCHING
# =========================

result = cv2.matchTemplate(region_crop, img1, cv2.TM_CCOEFF_NORMED)
similarity = result[0][0]

print("Similarity:", similarity)

if similarity >= THRESHOLD:
    print(True)
else:
    print(False)