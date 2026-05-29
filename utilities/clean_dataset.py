import os
import cv2

# =========================
# CONFIG
# =========================
current_dir = os.path.dirname(os.path.abspath(__file__))
FOLDER_PATH = os.path.join(current_dir, "..", "dataset", "noop")
SUPPORTED_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")

# =========================
# LOAD IMAGES
# =========================
images = [
    os.path.join(FOLDER_PATH, file)
    for file in os.listdir(FOLDER_PATH)
    if file.lower().endswith(SUPPORTED_EXTENSIONS)
]

images.sort()

if not images:
    print("No images found.")
    exit()

index = 0

# =========================
# MAIN LOOP
# =========================
while True:

    # If all images deleted
    if not images:
        print("No images remaining.")
        break

    image_path = images[index]

    img = cv2.imread(image_path)

    if img is None:
        print(f"Could not open: {image_path}")
        images.pop(index)

        if index >= len(images):
            index = max(0, len(images) - 1)

        continue

    # Resize if too large
    screen_height = 900
    h, w = img.shape[:2]

    if h > screen_height:
        scale = screen_height / h
        img = cv2.resize(img, (int(w * scale), int(h * scale)))

    # Show image
    display = img.copy()

    filename = os.path.basename(image_path)

    cv2.putText(
        display,
        f"{index + 1}/{len(images)} - {filename}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("Image Viewer", display)

    key = cv2.waitKeyEx(0)

    # =========================
    # RIGHT ARROW -> NEXT
    # =========================
    if key == 2555904:
        index = (index + 1) % len(images)

    # =========================
    # LEFT ARROW -> PREVIOUS
    # =========================
    elif key == 2424832:
        index = (index - 1) % len(images)

    # =========================
    # SPACE -> DELETE IMAGE
    # =========================
    elif key == 32:
        try:
            os.remove(image_path)
            print(f"Deleted: {filename}")

            images.pop(index)

            if index >= len(images):
                index = max(0, len(images) - 1)

        except Exception as e:
            print(f"Failed to delete: {e}")

    # =========================
    # ESC -> EXIT
    # =========================
    elif key == 27:
        break

cv2.destroyAllWindows()