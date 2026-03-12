import pydicom
import os
import sys
from main_gui import App

class DICOMManager:
    def __init__(self, file_list):
        self.file_list = file_list
        self.current_idx = 0
        self.status = "stay"

    def run(self):
        """Loop to handle navigation between DICOM frames."""
        while self.current_idx < len(self.file_list):
            file_path = self.file_list[self.current_idx]
            
            try:
                # Load the frame
                frame_data = pydicom.dcmread(file_path)
                
                # Initialize the Tkinter UI
                # We pass the frame and a callback for navigation
                app = App(frame_data)
                
                # Start UI
                app.mainloop()
                
                # After the window is closed, check navigation status
                # (You can set app.status inside main_gui.py before closing)
                nav_status = getattr(app, 'status', 'next')
                
                if nav_status == "quit":
                    break
                elif nav_status == "prev":
                    self.current_idx = max(0, self.current_idx - 1)
                else:
                    self.current_idx += 1
                    
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                self.current_idx += 1

def get_dicom_files(directory):
    """Helper to find all .dcm files in a folder."""
    files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.dcm')]
    return sorted(files)

if __name__ == "__main__":
    # CONFIGURATION: Set your folder path here
    DICOM_FOLDER = "./data" # Change this to your folder path
    
    if not os.path.exists(DICOM_FOLDER):
        print(f"Folder not found: {DICOM_FOLDER}")
        # Create a dummy folder for the example
        os.makedirs(DICOM_FOLDER, exist_ok=True)
        print("Please place your .dcm files in the /data folder.")
        sys.exit()

    files = get_dicom_files(DICOM_FOLDER)
    
    if not files:
        print("No DICOM files found in directory.")
    else:
        manager = DICOMManager(files)
        manager.run()
