import cv2
import os
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import transforms
from config import game_region
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


DATASET_DIR = os.path.join("dataset")
DATASET_PATHS = {
    "jump": os.path.join(DATASET_DIR, "jump"),
    "roll": os.path.join(DATASET_DIR, "roll"),
    "left": os.path.join(DATASET_DIR, "left"),
    "right": os.path.join(DATASET_DIR, "right"),
    "noop": os.path.join(DATASET_DIR, "noop")
}
ACTIONS= {
    
    "jump": 1,
    "roll": 2,
    "left": 3,
    "right": 4,
    "noop": 5,
    "jump_reversed": 1,
    "roll_reversed": 2,
    "left_reversed": 4,
    "right_reversed": 3,  # Reversed actions for left and right
    "noop_reversed": 5
}


class CropRegionTransformGameRegion:
    def __call__(self, img):
        left = game_region["left"]
        top = game_region["top"]
        right = left + game_region["width"]
        bottom = top + game_region["height"]

        return img.crop((left, top, right, bottom))
    
transform = transforms.Compose([
    CropRegionTransformGameRegion(),
    transforms.Grayscale(),
    transforms.Resize((96, 96)),
    transforms.ToTensor()
])

dataset = datasets.ImageFolder(
    root="dataset",
    transform=transform
)

loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True
)


path = "dataset/jump"
for img_name in os.listdir(path):
  
    img = Image.open(os.path.join(path, img_name))
    transformed = transform(img)
    cv2.imshow("Transformed Image", transformed.permute(1, 2, 0).numpy())
    cv2.waitKey(0)
# plt.figure(figsize=(8,4))

# plt.subplot(1,2,1)
# plt.title("Original")
# plt.imshow(img)
# plt.axis("off")

# plt.subplot(1,2,2)
# plt.title("Cropped + Grayscale + Resized")
# plt.imshow(transformed, cmap="gray")
# plt.axis("off")

# plt.show()