#!/opt/anaconda1anaconda2anaconda3/bin/python

import os
import sys
import urllib.request
from pathlib import Path

def download_progress(block_num, block_size, total_size):
    """Callback to display download progress."""
    downloaded = block_num * block_size
    if total_size > 0:
        percent = min(100, downloaded * 100 / total_size)
        # Using \r to overwrite the line for a clean progress output
        sys.stdout.write(f"\rDownloading... {percent:.1f}%")
        sys.stdout.flush()
    else:
        sys.stdout.write(f"\rDownloading... {downloaded} bytes")
        sys.stdout.flush()

def main():
    # 1. Locate the sam2 installation directory
    try:
        import sam2
        # Get the absolute path of the directory containing sam2
        sam2_init = Path(sam2.__file__).resolve()
        sam2_home = sam2_init.parent
    except ImportError:
        print("Error: 'sam2' package is not installed in the current environment.")
        sys.exit(1)

    # 2. Define the checkpoints directory
    checkpoint_home = Path(os.path.join(sam2_home,"checkpoints"))

    # 3. Create the directory if it doesn't exist
    try:
        checkpoint_home.mkdir(parents=True, exist_ok=True)
        print(f"Target directory: {checkpoint_home}")
    except Exception as e:
        print(f"Error: Could not create directory {checkpoint_home}. {e}")
        sys.exit(1)

    # 4. Define URLs
    base_url = "https://dl.fbaipublicfiles.com/segment_anything_2/092824"
    files = [
        "sam2.1_hiera_tiny.pt",
        "sam2.1_hiera_small.pt",
        "sam2.1_hiera_base_plus.pt",
        "sam2.1_hiera_large.pt"
    ]

    print("Starting download of SAM2.1 checkpoint files...")

    # 5. Download the files
    for filename in files:
        url = f"{base_url}/{filename}"
        destination = os.path.join(checkpoint_home,filename)

        print(f"\nProcessing: {filename}")

        try:
            # Replicates 'curl -O' behavior
            urllib.request.urlretrieve(url, destination, reporthook=download_progress)
            print(f"\nSuccessfully downloaded to {destination}")
        except Exception as e:
            print(f"\nFailed to download {filename}: {e}")
            sys.exit(1)

    print("\n" + "="*40)
    print("Success! SAM2 checkpoint files are downloaded.")
    print("You may now run GRIME-AI.")
    print("="*40)

if __name__ == "__main__":
    main()
