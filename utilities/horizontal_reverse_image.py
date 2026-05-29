import cv2
import os
import uuid
print("ehllo")
# 1. Define paths relative to this script
current_dir = os.path.dirname(os.path.abspath(__file__))
dataset_dir = os.path.join(current_dir, "..", "dataset")

# The 5 base actions we are looking for
base_actions = ['jump', 'left', 'noop', 'right', 'roll']

print("Starting Horizontal Data Augmentation Pipeline...\n")

for action in base_actions:
    input_folder = os.path.join(dataset_dir, action)
    
    # Verify the input folder actually exists before trying to read it
    if not os.path.exists(input_folder):
        print(f"Skipping folder '{action}': Path does not exist.")
        continue
        
    # Determine the correct destination folder based on the directional swap rule
    if action == 'left':
        dest_action = 'right_reversed'
    elif action == 'right':
        dest_action = 'left_reversed'
    else:
        dest_action = f"{action}_reversed"
        
    output_folder = os.path.join(dataset_dir, dest_action)
    
    # Automatically build the new directory if it doesn't exist yet
    os.makedirs(output_folder, exist_ok=True)
    
    # Grab all valid source images
    images = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    print(f"Processing '{action}' -> Saving into '{dest_action}' ({len(images)} files)...")
    
    count = 0
    for img_name in images:
        img_path = os.path.join(input_folder, img_name)
        img = cv2.imread(img_path)
        
        if img is None:
            continue
            
        # cv2.flip(img, 1) mirrors the image horizontally across the Y-axis
        flipped_img = cv2.flip(img, 1)
        
        # Save using a new unique identifier string
        unique_id = str(uuid.uuid4())
        output_path = os.path.join(output_folder, f"{unique_id}.jpg")
        
        cv2.imwrite(output_path, flipped_img)
        count += 1
        
    print(f"Successfully generated {count} flipped assets for '{dest_action}'.")

print("\nAll target reversed datasets generated cleanly!")