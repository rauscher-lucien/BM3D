import os
import numpy as np
import bm3d
import tifffile
import matplotlib.pyplot as plt

def bm3d_filter_3d_image_stack(image_stack, sigma_psd, test_index=None):
    # Test a specific slice before processing the entire stack
    if test_index is not None:
        if test_index < 0 or test_index >= image_stack.shape[0]:
            raise IndexError(f"Test index {test_index} out of range.")
        
        # Convert to float, assuming the original data range is 0-65535 for 16-bit
        test_slice = image_stack[test_index].astype(np.float64) / 65535.0
        filtered_test_slice = bm3d.bm3d(test_slice, sigma_psd=sigma_psd, stage_arg=bm3d.BM3DStages.ALL_STAGES)
        
        # Display the test slice result
        plt.imshow(filtered_test_slice, cmap='gray')
        plt.title(f"Filtered Test Slice {test_index}")
        plt.colorbar()
        plt.show()

        # Ask user for confirmation to proceed
        proceed = input("Proceed with filtering the entire volume? (yes/no): ")
        if proceed.lower() != 'yes':
            print("Filtering stopped by the user.")
            return None

    # Initialize an empty array for the filtered stack, making sure to work with float64 for processing
    filtered_stack = np.zeros_like(image_stack, dtype=np.float64)

    # Apply BM3D filtering slice by slice
    for i, slice in enumerate(image_stack):
        print(f"Processing slice {i}")
        # Convert to float, assuming the original data range is 0-65535 for 16-bit
        slice_float = slice.astype(np.float64) / 65535.0
        filtered_slice = bm3d.bm3d(slice_float, sigma_psd=sigma_psd, stage_arg=bm3d.BM3DStages.ALL_STAGES)
        filtered_stack[i, :, :] = filtered_slice

    # Convert the result back to 16-bit if necessary, ensuring the data is properly scaled back to the 0-65535 range
    filtered_stack_16bit = np.clip(filtered_stack * 65535, 0, 65535).astype('uint16')
    return filtered_stack_16bit

def save_filtered_stack(filtered_stack, input_path, output_folder=""):
    if filtered_stack is None:
        return
    
    # Generate the output filename based on the input filename
    base_name = os.path.basename(input_path)
    name, ext = os.path.splitext(base_name)
    output_filename = f"{name}_filtered_bm3d{ext}"
    output_path = os.path.join(output_folder, output_filename)

    # Save the filtered 3D stack
    tifffile.imwrite(output_path, filtered_stack)
    print(f"Filtered image stack saved at {output_path}")

if __name__ == "__main__":
    # Define input parameters and paths
    data_path = r"C:\Users\rausc\Documents\EMBL\data\droso-results\Good_Sample_02_t_1.TIFF"
    sigma_psd = 0.1
    test_index = 95  # Example index to test a specific slice before processing the entire volume

    # Step 1: Read the 3D image stack
    image_stack = tifffile.imread(data_path)

    # Step 2: Apply BM3D filtering to the image stack with testing
    filtered_stack = bm3d_filter_3d_image_stack(image_stack, sigma_psd, test_index=test_index)

    # Step 3: Save the filtered 3D stack
    save_filtered_stack(filtered_stack, data_path)
