import pydicom
import numpy as np
import matplotlib.pyplot as plt
import os
from glob import glob

# =====================================================================
# CONFIGURATION
# =====================================================================
# Define the set of words you want to trigger a selection (Case-Insensitive)
# e.g., if a file's SeriesDescription is "Chest PA", it will match "chest" or "pa"
ALLOWED_KEYWORDS = {"chest", "pa", "erect"}
# =====================================================================

def automated_view_selector(data_path, target_keywords):
    """
    Automatically selects DICOM frames based on metadata keywords, 
    converts them to standardized units (HU), and plots an anatomical histogram.
    """
    files = glob(os.path.join(data_path, "*.dcm"))
    selected_pixels = []
    
    # Ensure keywords are lowercase for case-insensitive matching
    target_keywords = {word.lower() for word in target_keywords}

    print(f"[*] Found {len(files)} DICOM files. Starting automated filtering...\n")

    for f in files:
        try:
            ds = pydicom.dcmread(f)
            
            # 1. Extract and format metadata for filtering
            view = getattr(ds, 'ViewPosition', '').lower()
            desc = getattr(ds, 'SeriesDescription', '').lower()
            
            # Create a set of words found in the metadata
            metadata_words = set(view.split() + desc.split())
            
            # Check for intersection between target keywords and metadata words
            if not target_keywords.intersection(metadata_words):
                print(f"[-] Skipped: {os.path.basename(f)} (No keyword match in '{view}' or '{desc}')")
                continue
            
            print(f"[+] Selected: {os.path.basename(f)} (Matched keywords)")
            
            # 2. Extract and Rescale to Meaningful Data (Hounsfield Units)
            raw_pixels = ds.pixel_array.astype(np.float32)
            slope = getattr(ds, 'RescaleSlope', 1.0)
            intercept = getattr(ds, 'RescaleIntercept', 0.0)
            
            rescaled_pixels = (raw_pixels * slope) + intercept

            # Handle Multi-frame
            if len(rescaled_pixels.shape) == 3:
                for i in range(rescaled_pixels.shape[0]):
                    selected_pixels.append(rescaled_pixels[i].flatten())
            else:
                selected_pixels.append(rescaled_pixels.flatten())

        except Exception as e:
            print(f"[!] Could not read {f}: {e}")

    if not selected_pixels:
        print("\n[!] No frames matched the target keywords. Analysis cancelled.")
        return

    # Combine data
    all_data = np.concatenate(selected_pixels)
    
    # 3. Plotting and Highlighting Anatomical Regions
    plt.figure(figsize=(14, 7))
    plt.hist(all_data, bins=500, log=True, color='midnightblue', alpha=0.8, zorder=2)
    
    # Define standard anatomical ranges in HU
    regions = {
        "Air": (-1000, -400),
        "Soft Tissue (Fat)": (-100, -40),
        "Tissue (Muscle/Organs)": (-40, 100),
        "Bones": (300, 1000),
        "External Objects (Metal)": (1000, 3000)
    }

    # Alternate colors: Green and Yellow
    colors = ['#28a745', '#ffc107'] 
    
    for i, (name, (min_val, max_val)) in enumerate(regions.items()):
        c = colors[i % 2]
        # axvspan creates a shaded vertical region
        plt.axvspan(min_val, max_val, color=c, alpha=0.2, label=f"{name} ({min_val} to {max_val})", zorder=1)

    # Calculate global markers for the whole dataset
    p5, p95 = np.percentile(all_data, [5, 95])
    plt.axvline(p5, color='red', linestyle='--', label=f'5th Pctl: {int(p5)}', zorder=3)
    plt.axvline(p95, color='red', linestyle='--', label=f'95th Pctl: {int(p95)}', zorder=3)
    
    plt.title("Filtered Pixel Value Distribution (Hounsfield Units)")
    plt.xlabel("Pixel Value (HU)")
    plt.ylabel("Log Count")
    
    # Move legend outside the plot so it doesn't cover the histogram
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.grid(True, which="both", ls="-", alpha=0.3)
    plt.tight_layout()
    plt.show()

    print("\n--- Final Recommendations based on Selection ---")
    print(f"Suggested Window Level (Center): {int((p95 + p5) / 2)}")
    print(f"Suggested Window Width (Spread): {int(p95 - p5)}")

if __name__ == "__main__":
    # Run the automated script using the global configuration variables
    automated_view_selector("./Data", ALLOWED_KEYWORDS)
