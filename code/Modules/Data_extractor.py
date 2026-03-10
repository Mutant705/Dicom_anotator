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
from dataclasses import dataclass
from typing import List

@dataclass
class DICOMFrame:
    """Public data structure for individual frame data and its specific metadata."""
    frame_index: int
    pixel_data: np.ndarray
    view_details: str
    encoding: str
    compression: str

class DICOMProcessor:
    def __init__(self, file_path: str):
        """
        Public constructor. 
        Loads the DICOM file and immediately runs the private extraction logic.
        """
        # Private raw dataset
        self.__ds = pydicom.dcmread(file_path)
        
        # Publicly accessible results
        self.frames: List[DICOMFrame] = []
        self.metadata = self.__extract_summary()
        
        # Trigger internal processing
        self.__extract_frames()

    def __extract_frames(self):
        """Private method: Parses pixel arrays and frame-specific metadata."""
        # Universal metadata for all frames in this file
        transfer_syntax = self.__ds.file_meta.TransferSyntaxUID
        encoding_name = transfer_syntax.name
        compression_status = "Compressed" if transfer_syntax.is_compressed else "Uncompressed"
        
        # Extract View Details (SeriesDescription or BodyPartExamined)
        view_details = getattr(self.__ds, 'SeriesDescription', 'Unknown View')
        
        # Handle Pixel Data
        pixel_array = self.__ds.pixel_array
        
        # Determine frame count (default to 1 if tag is missing)
        num_frames = int(getattr(self.__ds, 'NumberOfFrames', 1))

        if num_frames > 1 and pixel_array.ndim > 2:
            for i in range(num_frames):
                self.frames.append(DICOMFrame(
                    frame_index=i,
                    pixel_data=pixel_array[i],
                    view_details=view_details,
                    encoding=encoding_name,
                    compression=compression_status
                ))
        else:
            # Single frame case
            self.frames.append(DICOMFrame(
                frame_index=0,
                pixel_data=pixel_array,
                view_details=view_details,
                encoding=encoding_name,
                compression=compression_status
            ))

    def __extract_summary(self) -> dict:
        """Private method: Safely extracts high-level metadata."""
        return {
            "PatientName": str(getattr(self.__ds, 'PatientName', 'N/A')),
            "Modality": getattr(self.__ds, 'Modality', 'N/A'),
            "StudyDate": getattr(self.__ds, 'StudyDate', 'N/A'),
            "SOPClassUID": self.__ds.SOPClassUID
        }
