import os
import numpy as np
import bm3d
import tifffile
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim

def read_tiff_stack(filepath):
    with tifffile.TiffFile(filepath) as tiff:
        images = tiff.asarray()
    return images

def bm3d_filter_3d_image_stack(noisy_stack, ground_truth, sigma_psd_range, test_index):
    if test_index < 0 or test_index >= noisy_stack.shape[0]:
        raise IndexError(f"Test index {test_index} out of range.")
    
    # Convert to float, assuming the original data range is 0-65535 for 16-bit
    test_slice = noisy_stack[test_index].astype(np.float64) / 65535.0
    ground_truth_slice = ground_truth[test_index].astype(np.float64) / 65535.0
    
    psnr_values = []
    ssim_values = []
    best_psnr = -np.inf
    best_ssim = -np.inf
    best_sigma_psd_psnr = None
    best_sigma_psd_ssim = None
    
    for sigma_psd in sigma_psd_range:
        print(f"Testing sigma_psd={sigma_psd}")
        
        # Apply BM3D filter to the test slice
        filtered_test_slice = bm3d.bm3d(test_slice, sigma_psd=sigma_psd, stage_arg=bm3d.BM3DStages.ALL_STAGES)
        
        # Calculate PSNR and SSIM
        current_psnr = psnr(ground_truth_slice, filtered_test_slice)
        current_ssim = ssim(ground_truth_slice, filtered_test_slice, data_range=filtered_test_slice.max() - filtered_test_slice.min())
        print(f"PSNR: {current_psnr}, SSIM: {current_ssim}")

        psnr_values.append(current_psnr)
        ssim_values.append(current_ssim)

        if current_psnr > best_psnr:
            best_psnr = current_psnr
            best_sigma_psd_psnr = sigma_psd
        
        if current_ssim > best_ssim:
            best_ssim = current_ssim
            best_sigma_psd_ssim = sigma_psd
    
    print(f"Best PSNR: {best_psnr} with sigma_psd={best_sigma_psd_psnr}")
    print(f"Best SSIM: {best_ssim} with sigma_psd={best_sigma_psd_ssim}")
    
    return best_sigma_psd_psnr, best_sigma_psd_ssim, psnr_values, ssim_values

if __name__ == "__main__":
    output_dir = r"C:\Users\rausc\Documents\EMBL\data\general-results"
    sample_paths = [
        (r"C:\Users\rausc\Documents\EMBL\data\nema-results\Nematostella_B-average-100.TIFF",
         r"C:\Users\rausc\Documents\EMBL\data\nema-results\Nematostella_B_V0.TIFF", 60),
        (r"C:\Users\rausc\Documents\EMBL\data\droso-results\Drosophila20210316LogScale01L_Good_Sample_02_t_-average-100-offset--2.TIFF",
         r"C:\Users\rausc\Documents\EMBL\data\droso-results\Good_Sample_02_t_1.TIFF", 100),
        (r"C:\Users\rausc\Documents\EMBL\data\mouse-results\MouseEmbryo20230602LogScaleMouse_embyo_10hour-average-20.TIFF",
         r"C:\Users\rausc\Documents\EMBL\data\mouse-results\Mouse_embyo_10hour_V0.TIFF", 250)
        # Add more (ground_truth_path, noisy_file_path, test_index) tuples as needed
    ]
    
    sigma_psd_range = np.arange(0.01, 0.8, 0.01)
    custom_labels = [
        "nema",
        "droso",
        "mouse"
        # Add more custom labels as needed
    ]
    
    if len(custom_labels) != len(sample_paths):
        raise ValueError("The number of custom labels must match the number of samples.")
    
    all_psnr_scores = np.zeros((len(sample_paths), len(sigma_psd_range)))
    all_ssim_scores = np.zeros((len(sample_paths), len(sigma_psd_range)))
    best_sigma_psnr_values = []
    best_sigma_ssim_values = []

    for i, (gt_path, noisy_path, test_index) in enumerate(sample_paths):
        ground_truth = read_tiff_stack(gt_path)
        noisy_stack = read_tiff_stack(noisy_path)
        
        best_sigma_psnr, best_sigma_ssim, psnr_values, ssim_values = bm3d_filter_3d_image_stack(noisy_stack, ground_truth, sigma_psd_range, test_index)
        
        all_psnr_scores[i, :] = psnr_values
        all_ssim_scores[i, :] = ssim_values
        best_sigma_psnr_values.append(best_sigma_psnr)
        best_sigma_ssim_values.append(best_sigma_ssim)
    
    # Compute mean PSNR and SSIM scores
    mean_psnr_scores = np.mean(all_psnr_scores, axis=0)
    mean_ssim_scores = np.mean(all_ssim_scores, axis=0)
    
    # Rescale PSNR and SSIM scores to the 0-1 range using the respective minima and maxima
    normalized_mean_psnr_scores = (mean_psnr_scores - np.min(mean_psnr_scores)) / (np.max(mean_psnr_scores) - np.min(mean_psnr_scores))
    normalized_mean_ssim_scores = (mean_ssim_scores - np.min(mean_ssim_scores)) / (np.max(mean_ssim_scores) - np.min(mean_ssim_scores))
    
    # Compute the combined score as the average of the normalized PSNR and SSIM scores
    combined_scores = (normalized_mean_psnr_scores + normalized_mean_ssim_scores) / 2
    
    best_sigma_overall_combined = sigma_psd_range[np.argmax(combined_scores)]

    print(f"Best overall sigma_psd value considering both PSNR and SSIM: {best_sigma_overall_combined}")
    
    # Plot average PSNR and SSIM over all samples with separate y-axes
    fig, ax1 = plt.subplots(figsize=(12, 6))

    ax1.set_xlabel('Sigma_psd')
    ax1.set_ylabel('Average PSNR', color='tab:blue')
    ax1.plot(sigma_psd_range, mean_psnr_scores, label='Average PSNR', marker='o', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.set_ylabel('Average SSIM', color='tab:red')  # we already handled the x-label with ax1
    ax2.plot(sigma_psd_range, mean_ssim_scores, label='Average SSIM', marker='s', color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.title('Average PSNR and SSIM over all samples')
    plt.grid(True)

    plot_filename = 'psnr_ssim_comparison_bm3d.png'
    plt.savefig(os.path.join(output_dir, plot_filename), bbox_inches='tight')
    plt.show()
    
    print(f"Plot saved to {os.path.join(output_dir, plot_filename)}")
