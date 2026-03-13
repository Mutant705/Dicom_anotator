import cupy as cp

class NormalizationEngine:
    @staticmethod
    def run(raw_data, mode, v1, v2):
        """
        Takes 16-bit GPU array and transforms it into 8-bit based on mode.
        """
        data = raw_data.astype(cp.float32)
        
        if mode == "Linear":
            # Windowing (v1: Level/Min, v2: Width/Max)
            res = cp.clip(data, v1, v2)
            res = (res - v1) / (v2 - v1 + 1e-7)
            
        elif mode == "Skewed":
            res = 1.0 / (1.0 + cp.exp(-(data - v1) / (v2 + 1e-7)))
            
        elif mode == "Z-Score":
            mean, std = cp.mean(data), cp.std(data)
            res = cp.clip((data - (mean + v2)) / (std + 1e-7), -v1, v1)
            res = (res + v1) / (2 * v1)
            
        elif mode == "Log":
            res = cp.log1p(cp.maximum(0, data * v1 + v2))
            res = (res - res.min()) / (res.max() - res.min() + 1e-7)
            
        else:
            res = (data - data.min()) / (data.max() - data.min() + 1e-7)
            
        return (res * 255).astype(cp.uint8)
