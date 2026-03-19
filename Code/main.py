import tkinter as tk
import os, glob, shutil
from Modules.Data_extractor import DICOMProcessor
from Modules.Interface import AnnotatorUI

def filter_frames_by_view(proc, allowed_words):
    """
    Filters frames to only keep those where the view name is made entirely 
    of a combination of the allowed words (case-insensitive).
    """
    filtered_frames = []
    # Ensure our reference set is fully lowercase
    allowed_lower = {w.lower() for w in allowed_words}
    
    for frame in proc.frames:
        # Replace non-alphanumeric chars with spaces to split cleanly (e.g., "CHEST, AP" -> "CHEST  AP")
        view_clean = "".join([c if c.isalnum() else " " for c in frame.view_details])
        view_words = set(view_clean.lower().split())
        
        # Check if the words in the view name form a subset of the allowed words.
        # We also check `if view_words` to ensure we don't accidentally keep entirely blank views.
        if view_words and view_words.issubset(allowed_lower):
            filtered_frames.append(frame)
            
    # Overwrite the processor's frame list with only the surviving frames
    proc.frames = filtered_frames

def main():
    # Define your paths
    data_dir = os.path.join(os.path.dirname(__file__), 'Data')
    train_dir = os.path.join(os.path.dirname(__file__), 'training_data')
    annotated_dir = os.path.join(data_dir, 'already_annotated')
    
    # Ensure the annotated directory exists
    os.makedirs(annotated_dir, exist_ok=True)
    dcm_files = glob.glob(os.path.join(data_dir, "*.dcm"))
    
    # Define the mapping to be saved into the NPZ
    CLASS_MAPPING = {
        1: "Bones", 2: "L_Pleural", 3: "R_Pleural", 
        4: "Mediastinum", 5: "Abdominal", 6: "test"
    }
    
    # === EDIT THIS SET ===
    # Add all acceptable words here. If a DICOM's view name contains a word 
    # NOT in this list, it will be discarded.
    ALLOWED_VIEW_WORDS = {"pa", "chest", "erect",}
    
    for f_path in dcm_files:
        proc = DICOMProcessor(f_path)
        
        # Filter out unwanted views
        filter_frames_by_view(proc, ALLOWED_VIEW_WORDS)
        
        # If all frames in this DICOM were discarded, skip to the next file
        if not proc.frames:
            print(f"[-] Skipping {os.path.basename(f_path)}: No matching views found.")
            continue
            
        root = tk.Tk()
        
        def on_save():
            # The UI will call this to save, but the UI itself will handle the closing
            ui.sync_mask()
            for frame in proc.frames:
                frame.save_npz(train_dir, proc.filename, CLASS_MAPPING)
            
            # Move the source file to the completed folder
            try:
                destination = os.path.join(annotated_dir, os.path.basename(f_path))
                shutil.move(f_path, destination)
                print(f"[*] Moved {os.path.basename(f_path)} to already_annotated/\n")
            except Exception as e:
                print(f"[!] Failed to move file: {e}\n")

        ui = AnnotatorUI(root, proc, on_save)
        root.mainloop()

if __name__ == "__main__":
    main()
