#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: John Edward Stranzl, Jr.
# Affiliation(s): University of Nebraska-Lincoln, Blade Vision Systems, LLC
# Contact: jstranzl2@huskers.unl.edu, johnstranzl@gmail.com
# Created: Mar 6, 2022
# License: Apache License, Version 2.0, http://www.apache.org/licenses/LICENSE-2.0

import re
from PIL import Image
from PIL.ExifTags import TAGS
from datetime import datetime
from tkinter import Tk, filedialog, messagebox
from pytz import timezone


# ==================================================================================================================
# THIS CLASS SEARCHES FOR THE DATE-TIME USING THE FILENAME AND EXIF DATA.
# EACH FUNCTION RETURNS THE DATE-TIME IN THE ISO FORMAT.
# BEFORE ISO IS RETURNED, THE DATE-TIME IS SPLIT INTO IMAGE_DATE & IMAGE_TIME.
# THESE CAN BE RETURNED SEPARATELY IF HAVING THEM SEPARATE IS PREFERRED.
# ==================================================================================================================


class GRIME_AI_TimeStamp_Utils:
    def __init__(self):
        # A working list of all date-time patterns in regex format.
        self.patterns = [
            r'(\d{4})(\d{2})(\d{2})-(\d{2})(\d{2})(\d{2})',       # e.g., 20210615-123456 (Pi naming convention)
            r'(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})',       # e.g., 20210615_123456 (Reconyx)
            r'(\d{8})-(\d{6})',                                   # e.g., 20210615-123456 (Reconyx alternative)
            r'(\d{4})_(\d{2})_(\d{2})_(\d{2})(\d{2})(\d{2})',     # e.g., 2024_06_17_053004 (Phenocam)
            r'(\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}Z)',            # e.g., 2023-10-26T13-00-31Z (USGS)
            r'(\d{4})(\d{2})(\d{2})',                             # e.g., 20210615 (PBT KOLA)
            r'(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})_GC_KOLA',      # e.g., 20240510_1515_GC_KOLA (GaugeCam)
            r'KearneyHigh_(\d{8})_Harner_\d+',                    # e.g., KearneyHigh_20240624_Harner_302 (Kola Harner)
            r'(\d{4})(\d{2})(\d{2})-(\d{2})(\d{2})(\d{2})',       # e.g., 2021-06-15-12-34-56
        ]

    # ==================================================================================================================
    # EXTRACTING DATE-TIME FROM THE FILENAME
    # ==================================================================================================================
    def extract_datetime_from_filename(self, filename):
        for pattern in self.patterns:
            match = re.search(pattern, filename)
            if match:
                dt_parts = match.group(0)
                try:
                    # Handling USGS DT, where T and Z are used in filename.
                    if 'T' in dt_parts and 'Z' in dt_parts:
                        image_date, image_time = dt_parts.split('T')
                        image_time = image_time.rstrip('Z').replace('-', ':')
                        iso_format = f"{image_date}T{image_time}"
                        print(f"Extracted date: {image_date}")
                        print(f"Extracted time: {image_time}")
                        return iso_format
                    else:
                        # Otherwise, handle filenames using this.
                        dt_parts = match.groups()
                        if len(dt_parts) == 1:  # Specific pattern for KOLA Harner
                            image_date = dt_parts[0]
                            iso_format = datetime.strptime(image_date, "%Y%m%d").date().isoformat()
                            print(f"Extracted date: {image_date}")  # No time in filename, just date.
                            return iso_format
                        image_date = ''.join(dt_parts[:3])
                        image_time = ''.join(dt_parts[3:])
                        datetime_str = f"{image_date} {image_time}"
                        iso_format = datetime.strptime(datetime_str, "%Y%m%d %H%M%S").isoformat()
                        print(f"Extracted date: {datetime_str[:8]}")
                        print(f"Extracted time: {datetime_str[9:]}")
                        return iso_format
                except ValueError:
                    continue
        return None

    # ==================================================================================================================
    # EXTRACTING DATE-TIME FROM THE EXIF DATA
    # ==================================================================================================================
    def extract_datetime_from_exif(self, image_path):
        try:
            image = Image.open(image_path)
            exif_data = image._getexif()
            if not exif_data:
                print('No EXIF data to extract.')
                return None
            for tag, value in exif_data.items():
                decoded_tag = TAGS.get(tag, tag)
                print(decoded_tag, ":", value)  # List EXIF Data
                if decoded_tag == 'DateTimeOriginal':
                    image_date, image_time = value.split(' ')
                    iso_format = datetime.strptime(value, '%Y:%m:%d %H:%M:%S').isoformat()
                    print(f"Extracted date: {image_date}")
                    print(f"Extracted time: {image_time}")
                    return iso_format
        except Exception as e:
            print(f"Error extracting EXIF data: {e}")
            return None
        return None

    # ==================================================================================================================
    # GETTING THE DATE-TIME FROM AN IMAGE BY MERGING BOTH PREVIOUS FUNCTIONS
    # ==================================================================================================================
    def get_image_datetime(self, image_path):
        filename = image_path.split('/')[-1]

        datetime_from_filename = self.extract_datetime_from_filename(filename)
        if datetime_from_filename:
            return datetime_from_filename

        datetime_from_exif = self.extract_datetime_from_exif(image_path)
        if datetime_from_exif:
            return datetime_from_exif

        return None

    # ==================================================================================================================
    # A BASIC TKINTER WINDOW FOR SELECTING AN IMAGE
    # ==================================================================================================================
    def select_image(self):
        root = Tk()
        root.withdraw()
        img_path = filedialog.askopenfilename()
        root.destroy()
        if not img_path:
            messagebox.showerror("Error", "No image selected.")
            return None
        return img_path

    # Function to convert datetime to USGS UTC format
    def convert_to_usgs_utc(self, dt):
        """Converts date/time to UTC.

            Args:
                dt (str): The date/time from either exif or filename as a string.

            Returns:
                dt_utc: The UTC converted form of the dt argument.

            Raises:
                Exception: If there is an error converting the date/ time.
            """
        dt_utc = dt.astimezone(timezone('UTC'))
        try:
            return dt_utc.strftime('%Y-%m-%dT%H-%M-%SZ')
        except Exception as e:
            print(e)


# ==================================================================================================================
# MAIN - TESTS THE FUNCTIONALITY OF TIMESTAMP_UTILS
# ==================================================================================================================
if __name__ == "__main__":
    extractor = GRIME_AI_TimeStamp_Utils()
    image_path = extractor.select_image()

    if image_path:
        datetime_value_exif = extractor.extract_datetime_from_exif(image_path)
        datetime_value_filename = extractor.extract_datetime_from_filename(image_path.split('/')[-1])

        if datetime_value_exif == datetime_value_filename:
            print('Filename and EXIF match.')  # If they match, printing for fun.

        if datetime_value_exif:
            print(f"\nExtracted datetime EXIF: {datetime_value_exif}")
            usgs_utc_exif = extractor.convert_to_usgs_utc(datetime.fromisoformat(datetime_value_exif))
            print(f"USGS UTC datetime EXIF: {usgs_utc_exif}")
        if datetime_value_filename:
            print(f"\nExtracted datetime filename: {datetime_value_filename}")
            usgs_utc_filename = extractor.convert_to_usgs_utc(datetime.fromisoformat(datetime_value_filename))
            print(f"USGS UTC datetime filename: {usgs_utc_filename}")
        else:
            print("\nNo datetime found in filename or EXIF data.")