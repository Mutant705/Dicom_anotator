# DISCLAIMER
In development of this project AI was used amd sorry for messy Flake.nix it was my first time using direnv to develop 

# Medical Image Annotator v2.0

A high-performance, GPU-accelerated suite for annotating and analyzing 16-bit medical DICOM images. This toolkit is designed to bridge the gap between raw medical data and machine learning training pipelines.

## 🛠 Dependencies

This project requires a Python 3.13+ environment with the following libraries:

* `pydicom`: For medical image header and pixel data parsing.
* `numpy`: For heavy-duty CPU matrix operations.
* `cupy`: (NVIDIA GPU Required) For real-time GPU-accelerated mask operations.
* `matplotlib`: For the interactive UI rendering and data visualization.
* `opencv-python (cv2)`: Used specifically for the geometric polygon filling logic.
* `tkinter`: For the host window and sidebar interface.

---

## 📂 Script Definitions

| Script | Function |
| --- | --- |
| **`main.py`** | The orchestration layer. It manages the file loop, handles the global class mapping, and connects the UI actions to the disk saving logic. |
| **`Modules/Interface.py`** | The GUI controller. Manages mouse input (drawing/erasing), sidebar controls, and the real-time interaction between the canvas and the engine. |
| **`Modules/AnnotatorEngine.py`** | The GPU core. Maintains 16 individual mask buffers on the GPU. Handles boundary-safe brush slicing and fills. |
| **`Modules/Data_extractor.py`** | The IO manager. Extracts frames from DICOMs and packages annotated data into metadata-rich `.npz` files. |
| **`Modules/NormalizationEngine.py`** | The visual processor. Maps raw 16-bit values to 8-bit space using Linear, Log, Skewed, or Z-Score algorithms. |
| **`Pixel_distribution_analyser.py`** | The calibration tool. Helps determine the optimal "View" settings by analyzing the bit-depth distribution of the dataset. |

---

## ⚙️ Configuration: Changing Classes

To add or modify classes (e.g., adding "Heart" or "Tumor"):

1. **Open `main.py**` and locate the `CLASS_MAPPING` dictionary.
2. **Add a new ID and Name**:
```python
CLASS_MAPPING = {
    1: "Bones",
    2: "L_Pleural",
    6: "Heart"  # New Class Added
}

```


3. **Update `Modules/Interface.py**`: Inside the `setup_ui` method, add the matching radio button to the sidebar so it appears in the GUI.

---

## 🔍 Pixel Distribution Analyser

Before annotating, use `Pixel_distribution_analyser.py` to calibrate your view.

* It scans your data folder and lets you manually verify which files to include (to filter out bad views or noisy frames).
* It generates a **Logarithmic Histogram** of pixel intensities.
* **Why use this?** Medical images often have values from 0 to 65535, but the "relevant" data (bones/tissue) might only exist between 1000 and 4000. This tool identifies those percentiles so you can set your `V1` and `V2` sidebar values accurately.

---

## 🧪 Advanced Extension: Reconstructing to DICOM (.dcm)

If your workflow requires saving annotations back into a DICOM format rather than `.npz`, you must modify the saving logic to handle medical headers and pixel encoding.

### Technical Requirements:

1. **Header Integrity:** You must clone the `file_meta` and the main dataset from the original source file.
2. **3D Mask Integration:** To store multiple classes, you can either:
* Create a **Multi-frame DICOM** where each frame is a binary mask.
* Use **Bit-Packing**: Assign bit 0 to Bones, bit 1 to Lungs, etc., and store them in a single 16-bit `PixelData` array.



### Sample DICOM Export Logic:

```python
def export_as_dicom(self, source_ds, save_path):
    import pydicom
    from pydicom.uid import ExplicitVRLittleEndian
    
    # 1. Create a copy of the source header
    new_ds = source_ds.copy()
    
    # 2. Flatten/Combine masks (Example: combining all into one bit-packed array)
    # Each class gets a specific bit in the 16-bit integer
    combined_mask = np.zeros(self.pixel_data.shape, dtype=np.uint16)
    for idx, mask in self.masks.items():
        if np.any(mask):
            combined_mask |= (mask.astype(np.uint16) << (idx - 1))

    # 3. Update pixel data and encoding
    new_ds.PixelData = combined_mask.tobytes()
    new_ds.BitsAllocated = 16
    new_ds.BitsStored = 16
    new_ds.HighBit = 15
    new_ds.PixelRepresentation = 0  # Unsigned integer
    
    # 4. Set Transfer Syntax to Uncompressed (standard for simple exports)
    new_ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    new_ds.fix_meta_info()
    
    new_ds.save_as(save_path)

```

---

## 🖱 UI Controls Reference

* **Left Click:** Draw (Green).
* **Right Click:** Proximity Eraser (Red Circle).
* **Yellow Line:** Indicates a closed loop is detected—releasing the mouse will trigger an automatic GPU fill.
* **Sidebar:** Use the "Apply View" button after changing V1/V2 to re-normalize the raw data.

---

## 🧠 Training Data Integration

The **`Sample_training_data_loading.py`** script serves as a bridge between your manual annotations and your Deep Learning models (like U-Net or SegNet). Because the `.npz` files are compressed and contain both the raw image and multiple binary masks, you need a specific way to extract them for training.

### How the Loader Works

1. **Bit-Depth Preservation:** It loads the `raw_image` as a 32-bit float. This is crucial because standard 8-bit images lose the subtle tissue details needed for medical AI.
2. **Metadata Mapping:** It reads the `class_map` saved inside the file. This ensures that even if you change class order later, your training script always knows which mask belongs to which organ.
3. **Binary Masks:** It extracts each mask (e.g., `mask_1` for Bones) as a clean 0/1 array.

### Sample Loader Implementation

This logic can be directly pasted into a **PyTorch Dataset** or a **TensorFlow Data Generator**:

```python
import numpy as np
import matplotlib.pyplot as plt

def load_annotation(npz_path):
    # Load the compressed file
    data = np.load(npz_path, allow_pickle=True)
    
    # 1. Get the raw image
    image = data['raw_image']
    
    # 2. Reconstruct the Class Mapping
    # Converts the saved array back into a usable dictionary
    class_map = dict(data['class_map'])
    
    # 3. Access a specific mask (e.g., Bones)
    # We use .get() to return a blank mask if that class wasn't annotated
    bone_mask = data.get('mask_1', np.zeros_like(image))
    lung_mask = data.get('mask_2', np.zeros_like(image))

    return image, bone_mask, lung_mask, class_map

# --- Quick Verification ---
img, bone, lung, labels = load_annotation("training_data/patient_001/frame_0_PA.npz")

print(f"Verified Labels: {labels}")
plt.subplot(1,2,1); plt.imshow(img, cmap='gray'); plt.title("Input")
plt.subplot(1,2,2); plt.imshow(bone, cmap='jet'); plt.title(f"Label: {labels[1]}")
plt.show()

```

### Tips for Training

* **Normalization:** Since `raw_image` is 16-bit, you should scale the pixels between `0.0` and `1.0` or use Z-Score normalization before feeding them into the model.
* **Data Augmentation:** When rotating or flipping the `image`, make sure you apply the **exact same transformation** to the `mask_n` arrays so they stay aligned.
* **Multi-Class Training:** For multi-class segmentation, stack your masks into a 3D volume (e.g., shape `[5, 1024, 1024]`) where each channel is a different class.



