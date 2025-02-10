import os
import sys
from PIL import Image

def verify_image_shapes(folder_path):

    # Normalize the folder path to have consistent folder separators
    folder_path = os.path.normpath(folder_path)

    # Get a list of all files in the folder with .jpg extension
    files = [file for file in os.listdir(folder_path) if file.lower().endswith('.jpg')]
    
    # Initialize a variable to store the shape of the first image
    first_image_shape = None
    
    # Iterate over each file in the folder
    for file in files:
        # Construct the full file path
        file_path = os.path.join(folder_path, file)
        
        # Open the image file
        with Image.open(file_path) as img:
            # Get the shape of the image (width, height, number of channels)
            image_shape = img.size + (len(img.getbands()),)
            
            # If this is the first image, store its shape
            if first_image_shape is None:
                first_image_shape = image_shape
            else:
                # Compare the shape of the current image with the first image
                if image_shape != first_image_shape:
                    return False
    
    return True, first_image_shape

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <folder_path>")
        sys.exit(1)
    
    folder_path = sys.argv[1]
    same_shape, shape = verify_image_shapes(folder_path)
    if same_shape:
        width, height, num_channels = shape
        print(f"All images have the same shape: width={width}, height={height}, number of channels={num_channels}.")
    else:
        print("Not all images have the same shape.")