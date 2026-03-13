import pydicom
import numpy as np
import matplotlib.pyplot as plt
import os
from glob import glob

def interactive_pa_selector(data_path):
    files = glob(os.path.join(data_path, "*.dcm"))
    selected_pixels = []

    print(f"Found {len(files)} DICOM files. Starting manual verification...\n")

    for f in files:
        try:
            ds = pydicom.dcmread(f)
            
            # Extract metadata for the "Permission" prompt
            view = getattr(ds, 'ViewPosition', 'NOT SPECIFIED')
            desc = getattr(ds, 'SeriesDescription', 'No Description')
            frames = getattr(ds, 'NumberOfFrames', 1)
            rows = getattr(ds, 'Rows', 0)
            cols = getattr(ds, 'Columns', 0)

            print("-" * 50)
            print(f"FILE: {os.path.basename(f)}")
            print(f"Metadata View: {view}")
            print(f"Description:   {desc}")
            print(f"Dimensions:    {rows}x{cols} | Frames: {frames}")
            
            # Ask the user for permission
            choice = input("Include this file in the Bone Spread Analysis? (y/n): ").lower()
            
            if choice == 'y':
                arr = ds.pixel_array
                # Handle Multi-frame (3D array)
                if len(arr.shape) == 3:
                    print(f"  --> Adding all {arr.shape[0]} frames...")
                    for i in range(arr.shape[0]):
                        selected_pixels.append(arr[i].flatten())
                else:
                    selected_pixels.append(arr.flatten())
                print("  [Added]")
            else:
                print("  [Skipped]")

        except Exception as e:
            print(f"Could not read {f}: {e}")

    if not selected_pixels:
        print("\nNo frames were selected. Analysis cancelled.")
        return

    # Combine data and plot
    all_data = np.concatenate(selected_pixels)
    
    plt.figure(figsize=(12, 6))
    plt.hist(all_data, bins=500, log=True, color='midnightblue', alpha=0.8)
    
    # Calculate markers based on your previous histogram (around 3000)
    p5, p95 = np.percentile(all_data, [5, 95])
    
    plt.axvline(p5, color='red', linestyle='--', label=f'5th Pctl: {int(p5)}')
    plt.axvline(p95, color='green', linestyle='--', label=f'95th Pctl: {int(p95)}')
    
    plt.title("Filtered Pixel Value Distribution (PA Views Only)")
    plt.xlabel("16-bit Pixel Value")
    plt.ylabel("Log Count")
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.show()

    print("\n--- Final Recommendations based on Selection ---")
    print(f"Suggested Window Level (Center): {int((p95 + p5) / 2)}")
    print(f"Suggested Window Width (Spread): {int(p95 - p5)}")

# Run the interactive script
interactive_pa_selector("./Data")

