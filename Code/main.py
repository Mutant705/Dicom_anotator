import tkinter as tk
import os, glob
from Modules.Data_extractor import DICOMProcessor
from Modules.Interface import AnnotatorUI

def main():
    # Define your paths
    data_dir = os.path.join(os.path.dirname(__file__), 'Data')
    train_dir = os.path.join(os.path.dirname(__file__), 'training_data')
    dcm_files = glob.glob(os.path.join(data_dir, "*.dcm"))
    
    # Define the mapping to be saved into the NPZ
    CLASS_MAPPING = {
        1: "Bones",
        2: "L_Pleural",
        3: "R_Pleural",
        4: "Mediastinum",
        5: "Abdominal",
        6: "test"
    }
    
    for f_path in dcm_files:
        root = tk.Tk()
        proc = DICOMProcessor(f_path)
        
        def on_save():
            ui.sync_mask()
            # We now pass CLASS_MAPPING as the 3rd argument to fix the TypeError
            for frame in proc.frames:
                frame.save_npz(train_dir, proc.filename, CLASS_MAPPING)
            root.destroy()

        ui = AnnotatorUI(root, proc, on_save)
        root.mainloop()

if __name__ == "__main__":
    main()


# import tkinter as tk
# import os, glob
# from Modules.Data_extractor import DICOMProcessor
# from Modules.Interface import AnnotatorUI
#
# def main():
#     data_dir = os.path.join(os.path.dirname(__file__), 'Data')
#     train_dir = os.path.join(os.path.dirname(__file__), 'training_data')
#     dcm_files = glob.glob(os.path.join(data_dir, "*.dcm"))
#
#     # Define the mapping to be saved into the NPZ
#     CLASS_MAPPING = {
#         1: "Bones",
#         2: "L_Pleural",
#         3: "R_Pleural",
#         4: "Mediastinum",
#         5: "Gastric"
#     }
#
#     for f_path in dcm_files:
#         root = tk.Tk()
#         proc = DICOMProcessor(f_path)
#
#         def on_save():
#             ui.sync_mask()
#             # We pass the mapping to the save function so it can be stored inside the NPZ
#             for frame in proc.frames:
#                 frame.save_npz(train_dir, proc.filename, CLASS_MAPPING)
#             root.destroy()
#
#         ui = AnnotatorUI(root, proc, on_save)
#         root.mainloop()
#
# if __name__ == "__main__":
#     main()
# #
# # import tkinter as tk
# # import os, glob
# # from Modules.Data_extractor import DICOMProcessor
# # from Modules.Interface import AnnotatorUI
#
# def main():
#     data_dir = os.path.join(os.path.dirname(__file__), 'Data')
#     train_dir = os.path.join(os.path.dirname(__file__), 'training_data')
#     dcm_files = glob.glob(os.path.join(data_dir, "*.dcm"))
#
#     for f_path in dcm_files:
#         root = tk.Tk()
#         proc = DICOMProcessor(f_path)
#
#         def on_save():
#             ui.sync_mask()
#             for frame in proc.frames:
#                 frame.save_npz(train_dir, proc.filename)
#             root.destroy()
#
#         ui = AnnotatorUI(root, proc, on_save)
#         root.mainloop()
#
# if __name__ == "__main__":
#     main()
