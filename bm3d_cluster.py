import os
import sys
import argparse
import logging
import numpy as np
import bm3d
import tifffile

class StreamToLogger(object):
    """
    Fake file-like stream object that redirects writes to a logger instance.
    """
    def __init__(self, logger, log_level=logging.INFO):
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ''

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.log_level, line.rstrip())

    def flush(self):
        pass

logging.basicConfig(filename='logging.log',  # Log filename
                    filemode='a',  # Append mode, so logs are not overwritten
                    format='%(asctime)s - %(levelname)s - %(message)s',  # Include timestamp
                    level=logging.INFO,  # Logging level
                    datefmt='%Y-%m-%d %H:%M:%S')  # Timestamp format

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)  # Set logging level for console
logging.getLogger('').addHandler(console_handler)

# Redirect stdout and stderr to logging
sys.stdout = StreamToLogger(logging.getLogger('STDOUT'), logging.INFO)
sys.stderr = StreamToLogger(logging.getLogger('STDERR'), logging.ERROR)

def bm3d_filter_3d_image_stack(image_stack, sigma_psd):
    # Initialize an empty array for the filtered stack, making sure to work with float64 for processing
    filtered_stack = np.zeros_like(image_stack, dtype=np.float64)

    # Apply BM3D filtering slice by slice
    for i, slice in enumerate(image_stack):
        print(f"Processing slice {i}")
        # Convert to float, assuming the original data range is 0-65535 for 16-bit
        slice_float = slice.astype(np.float64) / 65535.0
        filtered_slice = bm3d.bm3d(slice_float, sigma_psd=sigma_psd)
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

def main():
    parser = argparse.ArgumentParser(description='Apply BM3D filtering to a 3D image stack.')

    parser.add_argument('--data_path', type=str, help='Path to the 3D image stack (TIFF file)')
    parser.add_argument('--sigma_psd', type=float, help='Sigma PSD value for BM3D filtering')
    parser.add_argument('--output_folder', type=str, default='', help='Folder to save the filtered image stack')

    if os.getenv('RUNNING_ON_SERVER') == 'true':
        args = parser.parse_args()

        data_path = args.data_path
        sigma_psd = args.sigma_psd
        output_folder = args.output_folder

        print(f"Using data path: {data_path}")
        print(f"Sigma PSD value: {sigma_psd}")
        print(f"Output folder: {output_folder}")

    else:
        # Default settings for local testing
        data_path = r"\\tier2.embl.de\prevedel\members\Rauscher\data\big_data_small-test\nema\Nematostella_B_V0.TIFF"
        sigma_psd = 0.09
        output_folder = r"C:\Users\rausc\Documents\EMBL\data\general-results"

        print("Running locally with default settings:")
        print(f"Using data path: {data_path}")
        print(f"Sigma PSD value: {sigma_psd}")
        print(f"Output folder: {output_folder}")

    # Step 1: Read the 3D image stack
    image_stack = tifffile.imread(data_path)

    # Step 2: Apply BM3D filtering to the image stack
    filtered_stack = bm3d_filter_3d_image_stack(image_stack, sigma_psd)

    # Step 3: Save the filtered 3D stack
    save_filtered_stack(filtered_stack, data_path, output_folder)

if __name__ == '__main__':
    main()

