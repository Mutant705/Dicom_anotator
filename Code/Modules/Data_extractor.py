import pydicom
import numpy as np
import os
from dataclasses import dataclass, field
from typing import List, Dict

@dataclass
class DICOMFrame:
    frame_index: int
    pixel_data: np.ndarray  # Now properly scaled 16-bit
    view_details: str
    masks: Dict[int, np.ndarray] = field(default_factory=dict)

    def save_npz(self, base_dir, filename_no_ext, class_mapping):
        """Saves image, binary masks, and class metadata."""
        # Create a simple, flat filename for training convenience
        view_sanitized = "".join([c if c.isalnum() else "_" for c in self.view_details])
        save_folder = os.path.join(base_dir, filename_no_ext)
        os.makedirs(save_folder, exist_ok=True)

        # Payload: 16-bit raw image
        save_dict = {
            "raw_image": self.pixel_data,
            # We store the mapping so the training script knows what's what
            "class_map": np.array(list(class_mapping.items()), dtype=object)
        }
        
        # Add binary masks
        for m_idx, mask in self.masks.items():
            if np.any(mask):
                # IMPORTANT: Force strictly 0 and 1
                save_dict[f"mask_{m_idx}"] = (mask > 0).astype(np.uint8)

        # Save as one compressed file per frame
        file_name = f"frame_{self.frame_index}_{view_sanitized}.npz"
        np.savez_compressed(os.path.join(save_folder, file_name), **save_dict)
        print(f"[*] Exported {file_name}")

class DICOMProcessor:
    def __init__(self, file_path: str):
        self.ds = pydicom.dcmread(file_path)
        self.filename = os.path.basename(file_path).replace('.dcm', '')
        self.frames: List[DICOMFrame] = []
        self.__extract()

    def __extract(self):
        # Extract metadata for folder naming
        view = getattr(self.ds, 'SeriesDescription', 'Unknown_View')
        
        # 1. Start with a float
        raw_pixels = self.ds.pixel_array.astype(np.float32)
        
        # 2. Apply the rescale math (Handling CT Hounsfield Units, etc.)
        slope = getattr(self.ds, 'RescaleSlope', 1.0)
        intercept = getattr(self.ds, 'RescaleIntercept', 0.0)
        rescaled_pixels = (raw_pixels * slope) + intercept
        
        # 3. Shift the baseline (Bring negatives up to 0 safely)
        min_val = np.min(rescaled_pixels)
        if min_val < 0:
            rescaled_pixels -= min_val
            
        # 4. Handle the MONOCHROME1 inversion (X-Rays with inverted photometry)
        photometric_interp = getattr(self.ds, 'PhotometricInterpretation', '')
        if photometric_interp == 'MONOCHROME1':
            max_val = np.max(rescaled_pixels)
            rescaled_pixels = max_val - rescaled_pixels
            
        # 5. Clip and finalize
        clipped_pixels = np.clip(rescaled_pixels, 0, 65535)
        pixel_array = clipped_pixels.astype(np.uint16)
        
        # Determine number of frames
        if hasattr(self.ds, 'NumberOfFrames'):
            num_frames = int(self.ds.NumberOfFrames)
        else:
            num_frames = 1 if pixel_array.ndim == 2 else pixel_array.shape[0]

        # Populate frame objects
        if num_frames > 1 and pixel_array.ndim > 2:
            for i in range(num_frames):
                self.frames.append(DICOMFrame(i, pixel_array[i], view))
        else:
            self.frames.append(DICOMFrame(0, pixel_array, view))
#####old apprach forcing into 32 bit array 
# import pydicom
# import numpy as np
# import os
# from dataclasses import dataclass, field
# from typing import List, Dict
#
# @dataclass
# class DICOMFrame:
#     frame_index: int
#     pixel_data: np.ndarray  # Original 16-bit
#     view_details: str
#     masks: Dict[int, np.ndarray] = field(default_factory=dict)
#
#     def save_npz(self, base_dir, filename_no_ext, class_mapping):
#         """Saves image, binary masks, and class metadata."""
#         # Create a simple, flat filename for training convenience
#         view_sanitized = "".join([c if c.isalnum() else "_" for c in self.view_details])
#         save_folder = os.path.join(base_dir, filename_no_ext)
#         os.makedirs(save_folder, exist_ok=True)
#
#         # Payload: 16-bit raw image
#         save_dict = {
#             "raw_image": self.pixel_data,
#             # We store the mapping so the training script knows what's what
#             "class_map": np.array(list(class_mapping.items()), dtype=object)
#         }
#
#         # Add binary masks
#         for m_idx, mask in self.masks.items():
#             if np.any(mask):
#                 # IMPORTANT: Force strictly 0 and 1
#                 save_dict[f"mask_{m_idx}"] = (mask > 0).astype(np.uint8)
#
#         # Save as one compressed file per frame
#         file_name = f"frame_{self.frame_index}_{view_sanitized}.npz"
#         np.savez_compressed(os.path.join(save_folder, file_name), **save_dict)
#         print(f"[*] Exported {file_name}")
#
# class DICOMProcessor:
#     def __init__(self, file_path: str):
#         self.ds = pydicom.dcmread(file_path)
#         self.filename = os.path.basename(file_path).replace('.dcm', '')
#         self.frames: List[DICOMFrame] = []
#         self.__extract()
#
#     def __extract(self):
#         # Extract metadata for folder naming
#         view = getattr(self.ds, 'SeriesDescription', 'Unknown_View')
#         pixel_array = self.ds.pixel_array.astype(np.int32)
#
#         # Determine number of frames
#         if hasattr(self.ds, 'NumberOfFrames'):
#             num_frames = int(self.ds.NumberOfFrames)
#         else:
#             num_frames = 1 if pixel_array.ndim == 2 else pixel_array.shape[0]
#
#         # Populate frame objects
#         if num_frames > 1 and pixel_array.ndim > 2:
#             for i in range(num_frames):
#                 self.frames.append(DICOMFrame(i, pixel_array[i], view))
#         else:
#             self.frames.append(DICOMFrame(0, pixel_array, view))
#
#            self.frames.append(DICOMFrame(0, pixel_array, view))
