import os
import sys
import glob
import numpy as np

# Add Modules directory to path so we can import them
sys.path.append(os.path.join(os.path.dirname(__file__), 'Modules'))

from Data_extractor import DICOMProcessor
from t2 import DICOMVisualizer

def main():
    data_dir = os.path.join(os.path.dirname(__file__), 'Data')
    train_dir = os.path.join(os.path.dirname(__file__), 'training_data')
    
    # Get all .dcm files
    dcm_files = glob.glob(os.path.join(data_dir, "*.dcm"))
    if not dcm_files:
        print("No DICOM files found in /code/Data/")
        return

    file_idx = 0
    while file_idx < len(dcm_files):
        current_file = dcm_files[file_idx]
        filename_no_ext = os.path.basename(current_file).replace('.dcm', '')
        
        print(f"\n[FILE {file_idx+1}/{len(dcm_files)}] Loading: {filename_no_ext}")
        processor = DICOMProcessor(current_file)
        
        frame_idx = 0
        while frame_idx < len(processor.frames):
            frame = processor.frames[frame_idx]
            print(f"  -> Displaying Frame {frame_idx + 1}/{len(processor.frames)}")
            
            viz = DICOMVisualizer(frame)
            action = viz.show()

            if action == "quit":
                print("Exiting application...")
                return
            
            elif action == "delete":
                print(f"  !! Deleting Frame {frame_idx}")
                processor.frames.pop(frame_idx)
                # Don't increment frame_idx because the next frame shifted into current position
                if frame_idx >= len(processor.frames): 
                    break # Go to next file if we deleted the last frame
            
            elif action == "next":
                frame_idx += 1
            
            elif action == "prev":
                if frame_idx > 0:
                    frame_idx -= 1
                else:
                    print("  Starting of file reached.")

        # Save all remaining frames of this file
        print(f"[SAVE] Exporting frames for {filename_no_ext}...")
        file_save_path = os.path.join(train_dir, filename_no_ext)
        if not os.path.exists(file_save_path):
            os.makedirs(file_save_path)
            
        for f in processor.frames:
            # We use frame_index from original data to keep track
            f.save_npz(output_dir=file_save_path)

        file_idx += 1

    print("\nAll files processed successfully.")

if __name__ == "__main__":
    main()
