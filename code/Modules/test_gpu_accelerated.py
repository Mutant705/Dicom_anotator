from Data_extractor import DICOMProcessor
# Changed import to point to your new GPU-accelerated file
from Display_and_Anotate_gpu_accelrated import DICOMVisualizer

# Initialize the class
# Update path if necessary
dicom_path = "../Test_data/1.2.392.200036.9125.4.0.1.2.840.114257.1.1.10668.43713.2685653.1.dcm"
processor = DICOMProcessor(dicom_path)

# Access the list of processed frames
for frame in processor.frames:
    print(f"Frame {frame.frame_index}: {frame.pixel_data.shape}")
    print(f"Encoding: {frame.encoding} | Compression: {frame.compression}")

# Access metadata summary
print(processor.metadata)

# Select the first frame
frame_data = processor.frames[0]

# Initialize Visualizer 
# window_start defaults to 0 as requested, or pass "Normalize"
view = DICOMVisualizer(frame_data, window_start=0)

# Start the interactive UI
view.show()

# After you press 'q', the mask is pulled from GPU VRAM back to CPU RAM 
# and stored in frame_data.pixel_data
print("Annotation session finished.")
print(f"Resulting Mask Shape: {frame_data.pixel_data.shape}")
