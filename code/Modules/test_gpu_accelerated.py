import numpy as np
import os

# Import specific classes to avoid the 'module object is not callable' error
from Data_extractor import DICOMProcessor
from Display_and_Anotator_gpu_accelrated import DICOMVisualizer

# 1. Initialize the Processor
dicom_path = "../Test_data/1.2.392.200036.9125.4.0.1.2.840.114257.1.1.10668.43713.2685653.1.dcm"

if not os.path.exists(dicom_path):
    print(f"Error: Could not find file at {dicom_path}")
else:
    processor = DICOMProcessor(dicom_path)

    print("--- DICOM EXTRACTION SUMMARY ---")
    print(processor.metadata)
    
    # 2. Select the first frame
    frame_to_annotate = processor.frames[0]

    # 3. Initialize Visualizer 
    view = DICOMVisualizer(frame_to_annotate)

    print("\n--- OPENING GPU ACCELERATED UI ---")
    view.show()

    # 4. Post-Annotation Logic
    if getattr(frame_to_annotate, 'is_annotated', False):
        print("\n--- SAVING DATA ---")
        save_folder = "training_output"
        frame_to_annotate.save_npz(output_dir=save_folder)
        
        # Verification
        saved_file = os.path.join(save_folder, f"frame_{frame_to_annotate.frame_index}.npz")
        loaded_data = np.load(saved_file)
        print(f"Verified: {saved_file} contains {loaded_data.files}")
    else:
        print("\nSession ended without saving.")





