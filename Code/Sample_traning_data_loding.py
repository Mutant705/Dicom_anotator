import numpy as np
import matplotlib.pyplot as plt

def load_training_sample(file_path):
    # Load the compressed file
    data = np.load(file_path, allow_pickle=True)
    
    # Extract the original CT/X-ray image
    image = data['image']
    
    # Extract the class mapping to know what is what
    # Structure: [[1, 'Bones'], [2, 'L_Pleural'], ...]
    class_map = dict(data['class_map'])
    
    print(f"Loaded file: {file_path}")
    print(f"Available Classes: {class_map}")

    # Example: Accessing specific masks
    # Let's say we want to train a model to find Bones (Index 1)
    bone_mask = data.get('mask_1', np.zeros_like(image))
    
    return image, bone_mask, class_map

# --- Visualization Loop ---
file_to_check = "output_frame_0.npz"
img, mask, labels = load_training_sample(file_to_check)

fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].imshow(img, cmap='gray')
ax[0].set_title("Input Image (16-bit)")

ax[1].imshow(mask, cmap='jet')
ax[1].set_title(f"Target Mask: {labels[1]}")

plt.show()
