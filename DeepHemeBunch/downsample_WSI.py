import openslide
from PIL import Image
import numpy as np
import os


def downsample_svs(input_svs_path, output_svs_path, downsample_factor=16):
    """
    Downsample an SVS file by a given factor and save it as a new SVS file.

    Args:
        input_svs_path (str): Path to the input SVS file.
        output_svs_path (str): Path to save the downsampled SVS file.
        downsample_factor (int, optional): The downsampling factor. Default is 16.

    Returns:
        None
    """
    try:
        # Open the input SVS file using OpenSlide
        slide = openslide.OpenSlide(input_svs_path)

        # Get the dimensions of level 0 (original size)
        level_0_dims = slide.level_dimensions[0]

        # Calculate the dimensions of the downsampled image
        downsampled_dims = (
            level_0_dims[0] // downsample_factor,
            level_0_dims[1] // downsample_factor,
        )

        # Read the entire image at level 0 as a NumPy array
        img = slide.read_region((0, 0), 0, level_0_dims).convert("RGB")

        # Convert the image to a NumPy array and downsample using PIL
        img = img.resize(downsampled_dims, Image.LANCZOS)

        # Save the downsampled image as a TIFF (intermediate step since SVS creation requires special tools)
        intermediate_tiff_path = output_svs_path.replace(".svs", "_downsampled.tiff")
        img.save(intermediate_tiff_path, format="TIFF")

        # Optionally, convert the TIFF to SVS format (using external tools or library support if required)
        # This part depends on your specific environment or SVS-compatible tools like vips
        os.rename(
            intermediate_tiff_path, output_svs_path
        )  # For example purpose, rename the TIFF as SVS

        print(f"Downsampled SVS saved at: {output_svs_path}")

    except Exception as e:
        print(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    test_slide_to_downsample = "/media/hdd3/neo/test_slide_tcga_lusc_small.svs"
    output_path = "/media/hdd3/neo/test_slide_tcga_lusc_small_downsampled_64.svs"

    downsample_svs(test_slide_to_downsample, output_path, downsample_factor=64)
