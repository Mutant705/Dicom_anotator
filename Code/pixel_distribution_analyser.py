import pydicom
import numpy as np
import matplotlib.pyplot as plt
import os
from glob import glob

# =====================================================================
# CONFIGURATION
# =====================================================================
ALLOWED_KEYWORDS = {"chest", "pa", "ap"}
# =====================================================================

def automated_view_selector(data_path, target_keywords):
    files = glob(os.path.join(data_path, "*.dcm"))
    target_keywords = {word.lower() for word in target_keywords}

    print(f"[*] Found {len(files)} DICOM files. Starting memory-safe analysis...\n")

    # 1. PRE-DEFINE HISTOGRAM BINS (Instead of storing all pixels)
    # We create 600 bins from -2000 HU to +4000 HU to catch everything from Air to Metal
    bin_edges = np.linspace(-2000, 4000, 601)
    global_counts = np.zeros(600, dtype=np.int64)
    
    files_processed = 0

    for f in files:
        try:
            ds = pydicom.dcmread(f)
            
            view = getattr(ds, 'ViewPosition', '').lower()
            desc = getattr(ds, 'SeriesDescription', '').lower()
            metadata_words = set(view.split() + desc.split())
            
            # Check for keyword match
            if not target_keywords.intersection(metadata_words):
                continue
                
            # Read and rescale
            raw_pixels = ds.pixel_array.astype(np.float32)
            slope = getattr(ds, 'RescaleSlope', 1.0)
            intercept = getattr(ds, 'RescaleIntercept', 0.0)
            rescaled_pixels = (raw_pixels * slope) + intercept

            # 2. STREAMING COUNT & SUBSAMPLING
            # We take every 4th pixel (::4) to make the math lightning fast, 
            # sort it into our bins, and immediately throw the image out of RAM.
            if len(rescaled_pixels.shape) == 3:
                for i in range(rescaled_pixels.shape[0]):
                    frame_data = rescaled_pixels[i, ::4, ::4]
                    counts, _ = np.histogram(frame_data, bins=bin_edges)
                    global_counts += counts
            else:
                frame_data = rescaled_pixels[::4, ::4]
                counts, _ = np.histogram(frame_data, bins=bin_edges)
                global_counts += counts
                
            files_processed += 1
            
            # Print a status update so you know it is still chewing through data
            if files_processed % 50 == 0:
                print(f"  ... Processed {files_processed} valid files so far ...")

        except Exception as e:
            pass # Silently skip corrupted files so the script doesn't stop

    if files_processed == 0:
        print("\n[!] No frames matched the target keywords. Analysis cancelled.")
        return

    print(f"\n[*] Finished processing {files_processed} files. Generating plot...\n")

    # 3. CALCULATE PERCENTILES MATHEMATICALLY
    # Since we don't have the raw data anymore, we calculate the 5th and 95th 
    # percentiles using a cumulative distribution of our bin counts.
    cumulative_counts = np.cumsum(global_counts)
    total_pixels = cumulative_counts[-1]
    
    idx_p5 = np.searchsorted(cumulative_counts, total_pixels * 0.05)
    idx_p95 = np.searchsorted(cumulative_counts, total_pixels * 0.95)
    
    p5 = bin_edges[idx_p5]
    p95 = bin_edges[idx_p95]

    # 4. PLOTTING
    plt.figure(figsize=(14, 7))
    
    # We use plt.stairs because we already did the histogram math ourselves
    plt.stairs(global_counts, bin_edges, fill=True, color='midnightblue', alpha=0.8, zorder=2)
    plt.yscale('log') # Keep the Y-axis logarithmic 
    
    # Anatomical Ranges
    regions = {
        "Air": (-1000, -400),
        "Soft Tissue (Fat)": (-100, -40),
        "Tissue (Muscle/Organs)": (-40, 100),
        "Bones": (300, 1000),
        "External Objects (Metal)": (1000, 3000)
    }

    colors = ['#28a745', '#ffc107'] 
    
    for i, (name, (min_val, max_val)) in enumerate(regions.items()):
        c = colors[i % 2]
        plt.axvspan(min_val, max_val, color=c, alpha=0.2, label=f"{name} ({min_val} to {max_val})", zorder=1)

    plt.axvline(p5, color='red', linestyle='--', label=f'5th Pctl: {int(p5)}', zorder=3)
    plt.axvline(p95, color='red', linestyle='--', label=f'95th Pctl: {int(p95)}', zorder=3)
    
    plt.title(f"Filtered Pixel Distribution ({files_processed} Files - Hounsfield Units)")
    plt.xlabel("Pixel Value (HU)")
    plt.ylabel("Log Count")
    
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.grid(True, which="both", ls="-", alpha=0.3)
    plt.tight_layout()
    plt.show()

    print("\n--- Final Recommendations based on Selection ---")
    print(f"Suggested Window Level (Center): {int((p95 + p5) / 2)}")
    print(f"Suggested Window Width (Spread): {int(p95 - p5)}")

if __name__ == "__main__":
    automated_view_selector("./Data", ALLOWED_KEYWORDS)
