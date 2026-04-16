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
    # 1. Locate the ultralytics installation directory
    try:
        import ultralytics
        # Get the absolute path of the directory containing ultralytics
        ultralytics_init = Path(ultralytics.__file__).resolve()
        ultralytics_home = ultralytics_init.parent
    except ImportError:
        print("Error: 'ultralytics' package is not installed in the current environment.")
        sys.exit(1)

    # 2. Define the checkpoints directory
    assets_home = Path(os.path.join(ultralytics_home,"assets"))

    # 3. Create the directory if it doesn't exist
    try:
        assets_home.mkdir(parents=True, exist_ok=True)
        print(f"Target directory: {assets_home}")
    except Exception as e:
        print(f"Error: Could not create directory {assets_home}. {e}")
        sys.exit(1)

    # 4. Define URLs
    base_url = "https://huggingface.co/Ultralytics/YOLO11/resolve/main"
    files = [
        "yolo11n.pt",
        "yolo11s.pt",
        "yolo11m.pt",
        "yolo11l.pt"
    ]

    print("Starting download of YOLO11 model files...")

    # 5. Download the files
    for filename in files:
        url = f"{base_url}/{filename}"
        destination = os.path.join(assets_home,filename)

        print(f"\nProcessing: {filename}")

        try:
            # Replicates 'curl -O' behavior
            urllib.request.urlretrieve(url, destination, reporthook=download_progress)
            print(f"\nSuccessfully downloaded to {destination}")
        except Exception as e:
            print(f"\nFailed to download {filename}: {e}")
            sys.exit(1)

    print("\n" + "="*40)
    print("Success! YOLO11 model files are downloaded.")
    print("You may now run GRIME-AI.")
    print("="*40)

if __name__ == "__main__":
    main()
