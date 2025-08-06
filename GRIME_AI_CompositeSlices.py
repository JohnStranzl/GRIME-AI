import os
import datetime
import math
from PIL import Image

from GRIME_AI_QProgressWheel import QProgressWheel

class GRIME_AI_CompositeSlices:
    def __init__(self, sliceCenter, sliceWidth, show_gui=True):
        """
        Initializes the object with the given slice center and slice width.

        Args:
            sliceCenter (int): The center of the slice to be cropped from an image.
            sliceWidth (int): The width of the slice to be cropped from an image.
        """

        self.sliceCenter = sliceCenter
        self.sliceWidth = sliceWidth

        self.show_gui = show_gui


    # ==================================================================================================================
    #
    # ==================================================================================================================
    def crop_side(self, image, height):
        """
        Crops a specified side of an image based on the slice center and slice width attributes of the class.

        Args:
            image (PIL.Image): The image to be cropped.
            height (int): The height of the cropped image.

        Returns:
            PIL.Image: The cropped image.

        Raises:
            ValueError: If the slice center or slice width attributes are not properly set.
        """

        left = self.sliceCenter - self.sliceWidth
        if left < 0:
            left = 0

        right = self.sliceCenter + self.sliceWidth
        if right >= image.width:
            right = image.width - 1

        top = 0

        bottom = height

        return image.crop((left, top, right, bottom))


    # ==================================================================================================================
    #
    # ==================================================================================================================
    def create_composite_image(self, imageList, output_path):
        """
        Creates a composite image from a list of images by extracting a region of interest (ROI) from each image and
        sequentially pasting it into the composite image.

        Args:
            imageList (list): A list of image files. Each element in the list is an object with a `fullPathAndFilename` attribute.
            output_path (str): The path where the composite image will be saved.

        Returns:
            None

        Raises:
            IOError: If an error occurs while opening, processing, or saving the images.
        """

        if os.path.exists(output_path) == False:
            os.makedirs(output_path, exist_ok=True)

        imageCount = len(imageList)

        # DISPLAY PROGRESS BAR
        # ----------------------------------------------------------------------------------------------------
        if self.show_gui:
            progressBar = QProgressWheel()
            progressBar.setRange(0, imageCount + 1)
            progressBar.show()

        # CREATE A BASE FILENAME TO WHICH AN INDEX NUMBER WILL BE APPENDED AT THE TIME WHEN THE IMAGE IS SAVED
        # ----------------------------------------------------------------------------------------------------
        # Get the current time in ISO format
        current_time = datetime.datetime.now().isoformat()
        # Replace colons with underscores to avoid issues with filename
        outputFilename = os.path.join(output_path, ("CompositeImage - " + current_time.replace(':', '_')))

        # OPEN THE FIRST IMAGE TO GET ITS DIMENSIONS
        # ----------------------------------------------------------------------------------------------------
        first_image = Image.open(imageList[0].fullPathAndFilename)
        output_height = first_image.height
        output_width = first_image.width

        # CALCULATE THE MAX IMAGE WIDTH REQUIRED FOR ALL THE SLICES IF WE WERE TO GENERATE A SINGLE IMAGE
        strip_width = self.sliceWidth * 2
        print(f"Slice Width: {strip_width}")

        maxWidthRequired = imageCount * strip_width
        print(f"MAX Width Required: {maxWidthRequired}")

        numImages, imageWidths, filesPerImage = self.check_image_width(maxWidthRequired, strip_width, option=1)

        composite_image = Image.new('RGB', (imageWidths[0], output_height))

        total_slices = len(imageList)
        slice_count = 0
        current_image_index = 0

        for i, image_file in enumerate(imageList):
            if self.show_gui:
                progressBar.setWindowTitle(image_file.fullPathAndFilename)
                progressBar.setValue(i)
                progressBar.repaint()

            # OPEN AN IMAGE AND EXTRACT OUT A SLICE (i.e., THE ROI SELECTED BY THE END-USER)
            # ----------------------------------------------------------------------------------------------------
            # open the image
            image = Image.open(image_file.fullPathAndFilename)
            # extract the ROI
            cropped_image = self.crop_side(image, output_height)
            # insert the extracted ROI into the composite image after the previously inserted slice
            composite_image.paste(cropped_image, ((i % filesPerImage[current_image_index]) * strip_width, 0))
            slice_count += 1

            if slice_count == filesPerImage[current_image_index]:

                # Save the composite image
                compFilename = f"{outputFilename}{'-'}{current_image_index}{'.jpg'}"
                composite_image.save(compFilename)

                print("\nComposite Filename: ", compFilename)
                print("Composite Image ", current_image_index+1, " of ", numImages)
                print("Files Used in this Composite Image: ", filesPerImage)
                print("Composite Image Width: ", imageWidths)

                total_slices -= filesPerImage[current_image_index]
                current_image_index += 1

                if current_image_index < len(imageWidths):
                    adjusted_max_width = imageWidths[current_image_index]
                    composite_image = Image.new('RGB', (adjusted_max_width, output_height))
                    slice_count = 0

        # clean-up before exiting function
        # 1. close and delete the progress bar
        # 2. close the EXIF log file, if opened
        if self.show_gui:
            progressBar.close()
            del progressBar


    # ==================================================================================================================
    #
    # ==================================================================================================================
    def check_image_width(self, width, strip_width, option=1):
        MAX_WIDTH = 65535

        if width > MAX_WIDTH:
            num_images = width // MAX_WIDTH
            remaining_width = width % MAX_WIDTH
            if remaining_width > 0:
                num_images += 1

            if option == 1:
                if remaining_width > 0:
                    image_widths = [MAX_WIDTH] * (num_images - 1) + [remaining_width]
                else:
                    image_widths = [MAX_WIDTH] * num_images

                # Adjust widths to be multiples of strip_width
                image_widths = [((w // strip_width) * strip_width) for w in image_widths]
                if remaining_width % strip_width != 0:
                    image_widths[-1] += strip_width - (image_widths[-1] % strip_width)
            elif option == 2:
                equal_width = width // num_images
                remainders = width % num_images
                image_widths = [equal_width] * num_images

                for i in range(remainders):
                    image_widths[i] += 1

                # Adjust widths to be multiples of strip_width
                image_widths = [((w // strip_width) * strip_width) for w in image_widths]
                for i in range(remainders):
                    if image_widths[i] % strip_width != 0:
                        image_widths[i] += strip_width - (image_widths[i] % strip_width)

            num_strips_per_image = [w // strip_width for w in image_widths]

            return num_images, image_widths, num_strips_per_image  # Return number of images, image widths, and number of strips per image
        else:
            return 1, [width], [width // strip_width]  # Return 1 image with the given width and number of strips if within limit

