"""
MODULE: Data_extractor.py
DESCRIPTION: Extracts frames and metadata from DICOM files into structured dataclasses.

USAGE:
    from Data_extractor import DICOMProcessor

    # Initialize the class with the file path (this triggers extraction automatically)
    processor = DICOMProcessor("path/to/file.dcm")

    # Access the list of processed frames
    for frame in processor.frames:
        print(f"Frame {frame.frame_index}: {frame.pixel_data.shape}")
        print(f"Encoding: {frame.encoding} | Compression: {frame.compression}")
        
    # Access metadata summary
    print(processor.metadata)
"""
import pydicom
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict
import os

@dataclass
class DICOMFrame:
    """Enhanced data structure to hold raw data and resulting annotation masks."""
    frame_index: int
    pixel_data: np.ndarray
    view_details: str
    encoding: str
    compression: str
    # New attributes for the Annotation Module
    masks: List[np.ndarray] = field(default_factory=list) 
    class_mapping: Dict[int, str] = field(default_factory=dict)
    is_annotated: bool = False

    def save_npz(self, output_dir="dataset"):
        """Saves this specific frame and its masks for AI training."""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        filename = os.path.join(output_dir, f"frame_{self.frame_index}.npz")
        
        # Build payload: Raw Image + any masks that aren't empty
        save_dict = {"raw_image": self.pixel_data}
        for i, mask in enumerate(self.masks):
            if np.any(mask):
                save_dict[f"mask_{i+1}"] = mask.astype(np.uint8)
        
        # Metadata
        save_dict["classes"] = np.array(list(self.class_mapping.values()))
        
        np.savez_compressed(filename, **save_dict)
        print(f"[DISK] Saved {filename}")

class DICOMProcessor:
    def __init__(self, file_path: str):
        self.__ds = pydicom.dcmread(file_path)
        self.frames: List[DICOMFrame] = []
        self.metadata = self.__extract_summary()
        self.__extract_frames()

    def __extract_frames(self):
        ts = self.__ds.file_meta.TransferSyntaxUID
        encoding_name = ts.name
        compression_status = "Compressed" if ts.is_compressed else "Uncompressed"
        view_details = getattr(self.__ds, 'SeriesDescription', 'Unknown View')
        
        # Force raw pixel values to int32 for future-proof windowing
        pixel_array = self.__ds.pixel_array.astype(np.int32)
        num_frames = int(getattr(self.__ds, 'NumberOfFrames', 1))

        if num_frames > 1 and pixel_array.ndim > 2:
            for i in range(num_frames):
                self.frames.append(DICOMFrame(i, pixel_array[i], view_details, encoding_name, compression_status))
        else:
            self.frames.append(DICOMFrame(0, pixel_array, view_details, encoding_name, compression_status))

    def __extract_summary(self) -> dict:
        return {
            "PatientName": str(getattr(self.__ds, 'PatientName', 'N/A')),
            "Modality": getattr(self.__ds, 'Modality', 'N/A'),
            "SOPInstanceUID": getattr(self.__ds, 'SOPInstanceUID', 'N/A')
        }

