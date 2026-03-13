import pydicom
import numpy as np
import os
from dataclasses import dataclass, field
from typing import List, Dict

@dataclass
class DICOMFrame:
    frame_index: int
    pixel_data: np.ndarray  # Original 16-bit
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
        pixel_array = self.ds.pixel_array.astype(np.int32)
        
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


# import pydicom
# import numpy as np
# import os
# from dataclasses import dataclass, field
# from typing import List, Dict
#
# @dataclass
# class DICOMFrame:
#     frame_index: int
#     pixel_data: np.ndarray  # Original 16-bit (int32)
#     view_details: str
#     masks: Dict[int, np.ndarray] = field(default_factory=dict)
#
#     def save_npz(self, base_dir, filename_no_ext):
#         # Path: training_data / filename / frame_view / file.npz
#         view_sanitized = "".join([c if c.isalnum() else "_" for c in self.view_details])
#         folder_name = f"{self.frame_index}_{view_sanitized}"
#         save_path = os.path.join(base_dir, filename_no_ext, folder_name)
#
#         os.makedirs(save_path, exist_ok=True)
#
#         # Payload: 16-bit raw image + binary masks
#         save_dict = {"raw_image": self.pixel_data}
#         for m_idx, mask in self.masks.items():
#             if np.any(mask):
#                 save_dict[f"mask_{m_idx}"] = mask.astype(np.uint8)
#
#         np.savez_compressed(os.path.join(save_path, "file.npz"), **save_dict)
#
# class DICOMProcessor:
#     def __init__(self, file_path: str):
#         self.ds = pydicom.dcmread(file_path)
#         self.filename = os.path.basename(file_path).replace('.dcm', '')
#         self.frames: List[DICOMFrame] = []
#         self.__extract()
#
#     def __extract(self):
#         view = getattr(self.ds, 'SeriesDescription', 'Unknown_View')
#         pixel_array = self.ds.pixel_array.astype(np.int32)
#         num_frames = int(getattr(self.ds, 'NumberOfFrames', 1))
#
#         if num_frames > 1 and pixel_array.ndim > 2:
#             for i in range(num_frames):
#                 self.frames.append(DICOMFrame(i, pixel_array[i], view))
#         else:
#             self.frames.append(DICOMFrame(0, pixel_array, view))
