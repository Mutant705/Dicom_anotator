from Data_extractor import DICOMProcessor
from Display_and_Anotate import DICOMVisualizer


# Initialize the class
processor = DICOMProcessor(
    "../Test_data/1.2.392.200036.9125.4.0.1.2.840.114257.1.1.10668.43713.2685653.1.dcm"
)

# Access the list of processed frames
for frame in processor.frames:
    print(f"Frame {frame.frame_index}: {frame.pixel_data.shape}")
    print(f"Encoding: {frame.encoding} | Compression: {frame.compression}")

# Access metadata summary
print(processor.metadata)
frame_data = processor.frames[0]

view = DICOMVisualizer(frame_data)
view.show()
