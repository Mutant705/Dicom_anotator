# Disclamer : AI was used in making this Project
I made this to streamline my workflow and sorry for messy flake this is my firstime using direnv 

Here is your updated `README.md`. I have completely restructured it to match your new modular architecture, updated the data types to reflect the `uint16` memory optimization, corrected where the `CLASS_MAPPING` lives, and added a detailed mathematical breakdown of your Normalization Engine using LaTeX formulas.

***

# 🛠️ DICOM GPU Annotator

A high-performance Python toolset for annotating DICOM files using GPU-accelerated buffers (CuPy). This application strictly preserves the dynamic range of your medical data by standardizing it into memory-efficient `uint16` arrays while allowing for fast, multi-class semantic segmentation.

## 📂 Project Structure

```text
.
├── main.py                    # The entry point; configures classes and runs the app
├── Data/                      # 📥 Source: Place your .dcm files here
├── training_data/             # 📤 Target: Output folders for .npz files
└── Modules/
    ├── Data_extractor.py      # DICOM parsing, baseline shifting, and uint16 packaging
    ├── AnnotatorEngine.py     # CuPy backend for GPU-accelerated drawing and masking
    ├── NormalizationEngine.py # Dynamic floating-point windowing for the 8-bit UI
    └── Interface.py           # Tkinter/Matplotlib frontend UI
```

---

## ⚙️ Configuration & Customization

### 1. Adding, Removing, or Renaming Classes

Class configurations are now managed centrally in **`main.py`**. This mapping is saved directly into the output `.npz` files so your training scripts always know exactly what each mask represents.

```python
# main.py

    # Define the mapping to be saved into the NPZ
    CLASS_MAPPING = {
        1: "Bones",
        2: "L_Pleural",
        3: "R_Pleural",
        4: "Mediastinum",
        5: "Abdominal",
        6: "New_Class" # To add a class, simply append a new integer key
    }
```

### 2. Changing Source and Target Folders

The folder paths are also managed at the top of **`main.py`**:

```python
# main.py

def main():
    data_dir = os.path.join(os.path.dirname(__file__), 'Data')          # Source directory
    train_dir = os.path.join(os.path.dirname(__file__), 'training_data') # Target directory
```

---

## 🚀 How to Use

1. **Start the App**: Run `python main.py`.
2. **Interact**:
    * **Left Click**: Draw annotation (creates a smooth stroke).
    * **Left Click (Closed Loop)**: Automatically fills the inside of a drawn polygon.
    * **Right Click**: Erase annotations.
    * **Sidebar**: Adjust the Normalization Mode, `V1`, and `V2` parameters to change your visual contrast (does not alter the saved raw data).
    * **Navigation**: Use the `NEXT` and `PREV` buttons to move through DICOM frames.
3. **Save**: Click `SAVE & EXIT` to export the current file's frames and masks to the `training_data/` folder.

---

## 🧠 Using the Final Data for Training

The app saves data in **`.npz` (Compressed Numpy)** format. This preserves your clean `uint16` raw pixel intensities alongside `uint8` binary masks, drastically reducing storage size while maintaining absolute fidelity.

### How to load in your Training Script (PyTorch/TensorFlow):

```python
import numpy as np

# Load a specific frame
data = np.load("training_data/patient_x/frame_0_UnknownView.npz")

# 1. Access the raw uint16 pixels (0 to 65535)
raw_pixels = data['raw_image'] 

# 2. Check the class mapping to see what masks are available
print(data['class_map']) 
# Output: [[1 'Bones'] [2 'L_Pleural'] ...]

# 3. Access specific masks (strictly 0 or 1)
bone_mask = data['mask_1']   
pleural_mask = data['mask_2']
```

---

## 🧮 Normalization Engine Mathematics

To display 16-bit data (up to 65,535 shades) on an 8-bit monitor (256 shades), the `NormalizationEngine.py` temporarily casts the GPU data to `float32` and applies one of the following transformations before mapping the result to standard 0-255 pixels. You control these equations via the UI using parameters **V1** and **V2**. 

Let $x$ be the input pixel value, and $y$ be the normalized output (from $0.0$ to $1.0$).

### 1. Linear (Standard Windowing)
Standard DICOM window leveling. 
* **V1** = Minimum Bound (Level)
* **V2** = Maximum Bound (Width)

$$y = \frac{\max(V_1, \min(x, V_2)) - V_1}{V_2 - V_1}$$

### 2. Sigmoid (Non-Linear Soft Tissue Window)
Provides high contrast in a specific region of interest while smoothly compressing extreme highs and lows.
* **V1** = Center Threshold (The inflexion point)
* **V2** = Steepness/Temperature (Controls how harsh the contrast transition is)

$$y = \frac{1}{1 + e^{-(x - V_1) / V_2}}$$

### 3. Z-Score (Statistical Windowing)
Highlights variations relative to the image's overall mean ($\mu$) and standard deviation ($\sigma$).
* **V1** = Output Bound (Clips the standard deviations)
* **V2** = Mean Offset (Shifts the center point)

$$z = \max\left(-V_1, \min\left(\frac{x - (\mu + V_2)}{\sigma}, V_1\right)\right)$$
$$y = \frac{z + V_1}{2 \cdot V_1}$$

### 4. Log (Logarithmic Compression)
Brings up details in the dark regions of the image while aggressively compressing the bright regions.
* **V1** = Multiplier Scale
* **V2** = Baseline Shift

$$y_{raw} = \ln(1 + \max(0, x \cdot V_1 + V_2))$$
*(Note: $y_{raw}$ is then min-max normalized to 0.0 - 1.0).*

---

## 🛡️ Key Technical Features

* **Zero-Lag Drawing**: Uses `CuPy` to handle massive arrays directly in VRAM.
* **Smart Memory Boxing**: Draws and erases using boundary-safe slicing, so you never encounter out-of-bounds indexing crashes near the edge of an image.
* **Memory Efficient Pipeline**: Raw data is safely shifted and clipped to strictly unsigned `uint16` in the extractor, cutting the VRAM footprint in half compared to 32-bit floats.
* **Data Agnostic Normalization**: Because the raw `uint16` array is saved, you can apply entirely new windowing logic in your PyTorch/TensorFlow datasets later without having to re-export the images.

***
### Training Specialized Models (Selective Mask Loading)

Because the annotator saves every class as an independent binary mask within the `.npz` file, you don't have to train a massive, multi-class model every time. You can use the exact same dataset to train highly specialized models (e.g., a "Bone-Only" segmentation model) by selectively loading only the targets you need. 

Here is an example PyTorch `Dataset` class that uses a `target_classes` list to dynamically build your mask tensors on the fly:

**Usage Examples:**

```python
# 🦴 Example 1: Train a model ONLY on Bones (Class 1) with a Bone Window
bone_dataset = DICOMSegmentationDataset(
    data_dir="training_data/", 
    target_classes=[1], 
    vmin=1000, vmax=3500    
)
# Resulting mask tensor shape: [Batch, 1, H, W]

# 🫁 Example 2: Train a model on Left & Right Lungs (Classes 2 & 3) with a Tissue Window
lung_dataset = DICOMSegmentationDataset(
    data_dir="training_data/", 
    target_classes=[2, 3], 
    vmin=-500, vmax=1500   
)
# Resulting mask tensor shape: [Batch, 2, H, W]
```
