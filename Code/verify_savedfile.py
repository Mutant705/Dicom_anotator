import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
import os

def verify_npz():
    # 1. Setup Tkinter window strictly for the file dialog
    root = tk.Tk()
    root.withdraw()  # Hide the main Tkinter window

    # 2. Ask the user to select an annotated .npz file
    file_path = filedialog.askopenfilename(
        title="Select an Annotated .npz File",
        filetypes=[("Numpy Zipped Archives", "*.npz")]
    )

    if not file_path:
        print("[-] No file selected. Exiting.")
        return

    print(f"[*] Loading {os.path.basename(file_path)} into RAM...")
    
    # 3. Load the NPZ archive
    data = np.load(file_path, allow_pickle=True)
    
    if 'raw_image' not in data:
        print("[!] Error: 'raw_image' not found in this .npz file.")
        return

    raw_image = data['raw_image']
    
    # 4. Extract the class mapping dictionary
    class_map = {}
    if 'class_map' in data:
        # Convert the numpy array of pairs back into a standard Python dictionary
        for item in data['class_map']:
            class_map[int(item[0])] = str(item[1])
    else:
        print("[!] Warning: No class_map found in the archive.")
    
    # 5. Find all keys that represent masks
    mask_keys = [k for k in data.files if k.startswith('mask_')]
    
    print(f"[*] Success! Found {len(mask_keys)} mask(s).")
    for k in mask_keys:
        class_id = int(k.split('_')[1])
        class_name = class_map.get(class_id, "Unknown Class")
        print(f"    - {k} -> {class_name}")

    # 6. Dynamically set up the Matplotlib grid based on how many masks exist
    num_plots = 1 + len(mask_keys)
    cols = 3
    rows = (num_plots + cols - 1) // cols  # Ceiling division for rows
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    
    # Flatten axes array for easy iteration, handling the edge case of a single row
    if num_plots == 1:
        axes = [axes]
    elif isinstance(axes, np.ndarray):
        axes = axes.flatten()
        
    # --- Plot 1: The Raw Image ---
    axes[0].imshow(raw_image, cmap='gray')
    axes[0].set_title(f"Raw 16-bit Image\nMin: {raw_image.min()} | Max: {raw_image.max()}")
    axes[0].axis('off')
    
    # --- Plot 2+: The Masks ---
    for i, m_key in enumerate(mask_keys):
        ax = axes[i + 1]
        mask_data = data[m_key]
        class_id = int(m_key.split('_')[1])
        class_name = class_map.get(class_id, "Unknown")
        
        # Draw the raw image in the background
        ax.imshow(raw_image, cmap='gray')
        
        # Mask out the '0' values so only the '1's show up in color
        transparent_mask = np.ma.masked_where(mask_data == 0, mask_data)
        
        # Overlay the mask in a bright color (spring = magenta/yellow)
        ax.imshow(transparent_mask, cmap='spring', alpha=0.6, interpolation='none')
        
        ax.set_title(f"{class_name} (Class {class_id})")
        ax.axis('off')
        
    # Hide any empty subplots in the grid
    for j in range(num_plots, len(axes)):
        axes[j].axis('off')
        
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    verify_npz()
