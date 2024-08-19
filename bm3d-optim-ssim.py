import os
import numpy as np
import bm3d
import tifffile
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

def bm3d_filter_3d_image_stack(image_stack, ground_truth, sigma_psd_range, test_index, output_folder=""):
    if test_index < 0 or test_index >= image_stack.shape[0]:
        raise IndexError(f"Test index {test_index} out of range.")
    
    # Convert to float, assuming the original data range is 0-65535 for 16-bit
    test_slice = image_stack[test_index].astype(np.float64) / 65535.0
    ground_truth_slice = ground_truth[test_index].astype(np.float64) / 65535.0
    
    ssim_values = []
    best_ssim = -np.inf
    best_sigma_psd = None
    
    for sigma_psd in sigma_psd_range:
        print(f"Testing sigma_psd={sigma_psd}")
        
        # Apply BM3D filter to the test slice
        filtered_test_slice = bm3d.bm3d(test_slice, sigma_psd=sigma_psd, stage_arg=bm3d.BM3DStages.ALL_STAGES)
        
        # Calculate SSIM
        current_ssim, _ = ssim(ground_truth_slice, filtered_test_slice, full=True, data_range=1.0)
        print(f"SSIM: {current_ssim}")

        ssim_values.append(current_ssim)

        if current_ssim > best_ssim:
            best_ssim = current_ssim
            best_sigma_psd = sigma_psd
    
    print(f"Best SSIM: {best_ssim} with sigma_psd={best_sigma_psd}")
    
    # Plot SSIM values
    plt.figure(figsize=(8, 6))
    plt.plot(sigma_psd_range, ssim_values, marker='o')
    plt.title('SSIM vs Sigma_psd')
    plt.xlabel('Sigma_psd')
    plt.ylabel('SSIM')
    plt.grid(True)
    
    # Save plot
    base_name = os.path.basename(data_path)
    name, ext = os.path.splitext(base_name)
    ssim_plot_filename = f"{name}_bm3d_test_slice_{test_index}_SSIM_plot.png"
    ssim_plot_path = os.path.join(output_folder, ssim_plot_filename)
    
    plt.savefig(ssim_plot_path)
    print(f"SSIM plot saved at {ssim_plot_path}")
    
    plt.show()
    
    return best_sigma_psd

if __name__ == "__main__":
    # Define input parameters and paths
    data_path = r"C:\Users\rausc\Documents\EMBL\data\droso-results\Good_Sample_02_t_1.TIFF"
    ground_truth_path = r"C:\Users\rausc\Documents\EMBL\data\droso-results\Drosophila20210316LogScale01L_Good_Sample_02_t_-average-100-offset--2.TIFF"
    test_index = 100  # Example index to test a specific slice
    output_folder = r"C:\Users\rausc\Documents\EMBL\data\droso-results"  # Folder to save the plots
    
    # Ranges for sigma_psd parameter
    sigma_psd_range = np.arange(0.05, 0.30, 0.01)
    
    # Step 1: Read the 3D image stack and the ground truth
    image_stack = tifffile.imread(data_path)
    ground_truth = tifffile.imread(ground_truth_path)
    
    # Step 2: Find the best sigma_psd for BM3D filtering and plot SSIM
    best_sigma_psd = bm3d_filter_3d_image_stack(image_stack, ground_truth, sigma_psd_range, test_index, output_folder)

