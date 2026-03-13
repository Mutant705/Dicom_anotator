# Disclamer : AI was used in making this Project
I made this to streamline my workflow and sorry for messy flake this is my firstime using direnv 

# 🛠️ DICOM GPU Annotator & Segmenter

A high-performance Python toolset for annotating 16/32-bit DICOM files using GPU-accelerated buffers. This application preserves the full dynamic range of your medical data while allowing for multi-class semantic segmentation.

## 📂 Project Structure

```text
.
├── code
│   ├── main.py                # The entry point; manages file/frame loops
│   ├── Data                   # 📥 Source: Place your .dcm files here
│   ├── training_data          # 📤 Target: Output folders for .npz files
│   └── Modules
│       ├── Data_extractor.py                      # DICOM parsing logic
│       └── Display_and_Anotator_gpu_accelrated.py # GPU UI & Drawing logic

```

---

## ⚙️ Configuration & Customization

### 1. Adding, Removing, or Renaming Classes

To change the classification labels (e.g., changing "Vessels" to "Nerves" or adding a 5th class), edit the `CLASS_CONFIG` dictionary at the top of **`Modules/Display_and_Anotator_gpu_accelrated.py`**:

```python
# Modules/Display_and_Anotator_gpu_accelrated.py

CLASS_CONFIG = {
    1: "Bone",
    2: "Tissue",
    3: "Tumor",
    4: "Vessels",
    5: "New_Class_Name",  # To add a class, simply add a new index
}

```

*The UI will automatically generate a new button for any entry added here (supports up to 15 classes).*

### 2. Changing Source and Target Folders

The folder paths are managed in **`code/main.py`**. Look for the `main()` function:

```python
# code/main.py

def main():
    data_dir = os.path.join(os.path.dirname(__file__), 'Data')          # Change source here
    train_dir = os.path.join(os.path.dirname(__file__), 'training_data') # Change target here

```

---

## 🚀 How to Use

1. **Start the App**: Run `python code/main.py`.
2. **Interact**:
* **Left Click**: Draw annotation.
* **Right Click**: Erase.
* **Window Slider**: Adjust the 8-bit view window (does not affect raw data).
* **Navigation**: Use `Next`, `Previous`, or `Enter` to move through frames.
* **Delete**: Remove a low-quality frame from the dataset entirely.


3. **Save**: The data is exported automatically to `code/training_data/` when you finish a file.

---

## 🧠 Using the Final Data for Training

The app saves data in **`.npz` (Compressed Numpy)** format. This is superior to PNG or JPG because it keeps the raw pixel intensities (Hounsfield Units) and masks in separate channels.

### How to load in your Training Script (PyTorch/TensorFlow):

```python
import numpy as np

# Load a specific frame
data = np.load("training_data/patient_x/frame_0.npz")

# 1. Access the raw, un-normalized pixels
raw_pixels = data['raw_image'] 

# 2. Apply a custom window for a specific model (e.g., Bone Window)
# This is where 'granulation' happens!
bone_window = np.clip(raw_pixels, -500, 1500) 

# 3. Access specific masks
bone_mask = data['mask_1']   # Based on your CLASS_CONFIG index
tumor_mask = data['mask_3']

```

---

## 🛡️ Key Technical Features

* **Zero-Lag Drawing**: Uses `CuPy` to handle $3043 \times 3043$ arrays directly in VRAM.
* **Granulated Normalization**: By storing `int32` raw data, you can train multiple models (Lung, Bone, Soft Tissue) using the **same** saved files by applying different windowing in your training loader.
* **Memory Efficient**: Masks are stored as `uint8` binaries, significantly reducing the storage footprint.

Would you like me to add a **troubleshooting** section to the README regarding CuPy installation or Matplotlib backends?
