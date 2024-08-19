import os
import numpy as np
import bm3d
import tifffile
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr

def bm3d_filter_3d_image_stack(image_stack, ground_truth, sigma_psd_range, test_index, output_folder=""):
    if test_index < 0 or test_index >= image_stack.shape[0]:
        raise IndexError(f"Test index {test_index} out of range.")
    
    # Convert to float, assuming the original data range is 0-65535 for 16-bit
    test_slice = image_stack[test_index].astype(np.float64) / 65535.0
    ground_truth_slice = ground_truth[test_index].astype(np.float64) / 65535.0
    
    psnr_values = []
    best_psnr = -np.inf
    best_sigma_psd = None
    
    for sigma_psd in sigma_psd_range:
        print(f"Testing sigma_psd={sigma_psd}")
        
        # Apply BM3D filter to the test slice
        filtered_test_slice = bm3d.bm3d(test_slice, sigma_psd=sigma_psd, stage_arg=bm3d.BM3DStages.ALL_STAGES)
        
        # Calculate PSNR
        current_psnr = psnr(ground_truth_slice, filtered_test_slice)
        print(f"PSNR: {current_psnr}")

        psnr_values.append(current_psnr)

        if current_psnr > best_psnr:
            best_psnr = current_psnr
            best_sigma_psd = sigma_psd
    
    print(f"Best PSNR: {best_psnr} with sigma_psd={best_sigma_psd}")
    
    # Plot PSNR values
    plt.figure(figsize=(8, 6))
    plt.plot(sigma_psd_range, psnr_values, marker='o')
    plt.title('PSNR vs Sigma_psd')
    plt.xlabel('Sigma_psd')
    plt.ylabel('PSNR')
    plt.grid(True)
    
    # Save plot
    base_name = os.path.basename(data_path)
    name, ext = os.path.splitext(base_name)
    psnr_plot_filename = f"{name}_bm3d_test_slice_{test_index}_PSNR_plot.png"
    psnr_plot_path = os.path.join(output_folder, psnr_plot_filename)
    
    plt.savefig(psnr_plot_path)
    print(f"PSNR plot saved at {psnr_plot_path}")
    
    plt.show()
    
    return best_sigma_psd

if __name__ == "__main__":
    # Define input parameters and paths
    data_path = r"C:\Users\rausc\Documents\EMBL\data\droso-results\Good_Sample_02_t_1.TIFF"
    ground_truth_path = r"C:\Users\rausc\Documents\EMBL\data\droso-results\Drosophila20210316LogScale01L_Good_Sample_02_t_-average-100-offset--2.TIFF"
    test_index = 100  # Example index to test a specific slice
    output_folder = r"C:\Users\rausc\Documents\EMBL\data\droso-results"  # Folder to save the plots
    
    # Ranges for sigma_psd parameter
    sigma_psd_range = np.arange(0.05, 0.5, 0.01)
    
    # Step 1: Read the 3D image stack and the ground truth
    image_stack = tifffile.imread(data_path)
    ground_truth = tifffile.imread(ground_truth_path)
    
    # Step 2: Find the best sigma_psd for BM3D filtering and plot PSNR
    best_sigma_psd = bm3d_filter_3d_image_stack(image_stack, ground_truth, sigma_psd_range, test_index, output_folder)


