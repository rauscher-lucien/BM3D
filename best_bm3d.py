import os
import numpy as np
import tifffile
import matplotlib.pyplot as plt

def estimate_sigma_from_background(image_stack, x_range, y_range):
    """
    Estimate noise standard deviation (sigma) from the background of the image stack.

    Parameters:
    - image_stack: 3D numpy array
    - x_range: Tuple (x_min, x_max) defining the range of x-coordinates for the background region
    - y_range: Tuple (y_min, y_max) defining the range of y-coordinates for the background region

    Returns:
    - sigma: Estimated noise standard deviation
    """
    # Convert the image stack to the 0-1 range
    image_stack_normalized = image_stack.astype(np.float64) / 65535.0
    
    # Extract background pixels
    background_pixels = image_stack_normalized[:, y_range[0]:y_range[1], x_range[0]:x_range[1]].flatten()
    
    # Calculate standard deviation
    sigma = np.std(background_pixels)
    return sigma

if __name__ == "__main__":
    # Define input parameters and paths
    data_path = r"C:\Users\rausc\Documents\EMBL\data\nema-results\Nematostella_B_V0.TIFF"

    # Step 1: Read the 3D image stack
    image_stack = tifffile.imread(data_path)
    
    # Step 2: Define the background region (x_range, y_range)
    x_range = (95, 288)  # Example range, adjust based on your data
    y_range = (213, 321)  # Example range, adjust based on your data
    
    # Step 3: Estimate the noise standard deviation from the background
    sigma_psd = estimate_sigma_from_background(image_stack, x_range, y_range)
    print("Estimated noise standard deviation (sigma):", sigma_psd)

