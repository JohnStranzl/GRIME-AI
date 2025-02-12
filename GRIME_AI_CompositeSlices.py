import os
import datetime
import math
from PIL import Image

from GRIME_AI_QProgressWheel import QProgressWheel

class GRIME_AI_CompositeSlices():
    def __init__(self, sliceCenter, sliceWidth):
        """
        Initializes the object with the given slice center and slice width.

        Args:
            sliceCenter (int): The center of the slice to be cropped from an image.
            sliceWidth (int): The width of the slice to be cropped from an image.
        """

        self.sliceCenter = sliceCenter
        self.sliceWidth = sliceWidth


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


    def create_composite_image(self, imageList, output_path):
        """
        Creates a composite image from a list of images by extracting a region of interest (ROI) from each image and
        sequentially pasting it into the composite image.

        Args:
            imageList (list): A list of image files. Each element in the list is an object with a `fullPathAndFilename` attribute.
            output_path (str): The path where the composite image will be saved.
            side (str): The side of the image from which the ROI will be extracted.

        Returns:
            None

        Raises:
            IOError: If an error occurs while opening, processing, or saving the images.
        """

        imageCount = len(imageList)

        # DISPLAY PROGRESS BAR
        # ----------------------------------------------------------------------------------------------------
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

        # DETERMINE THE NUMBER OF COMPOSITE IMAGES NEEDED, OF THE DESIRED WIDTH, FOR THE SLICE WIDTH CHOSEN
        # ----------------------------------------------------------------------------------------------------
        maxWidth = 2048
        number_of_composite_images_required = math.ceil((imageCount * self.sliceWidth * 2) / maxWidth)
        number_of_slices_per_image = math.floor(maxWidth // (self.sliceWidth * 2))

        if imageCount < number_of_slices_per_image:
            number_of_slices_per_image = imageCount

        adjusted_max_width = number_of_slices_per_image * self.sliceWidth * 2

        # EXTRACT AN ROI OF A SPECIFIC WIDTH FOR THE HEIGHT OF THE IMAGE
        composite_image = Image.new('RGB', (adjusted_max_width, output_height))

        prevCompIndex = 0
        total_slices = len(imageList)
        slice_count = 0
        for i, image_file in enumerate(imageList):
            progressBar.setWindowTitle(image_file.fullPathAndFilename)
            progressBar.setValue(i)
            progressBar.repaint()

            compIndex = i // number_of_slices_per_image

            # OPEN AN IMAGE AND EXTRACT OUT A SLICE (i.e., THE ROI SELECTED BY THE END-USER)
            # ----------------------------------------------------------------------------------------------------
            # open the image
            image = Image.open(image_file.fullPathAndFilename)
            # extract the ROI
            cropped_image = self.crop_side(image, output_height)
            # insert the extracted ROI into the composite image after the previously inserted slice)
            composite_image.paste(cropped_image, ((i % number_of_slices_per_image) * self.sliceWidth * 2, 0))
            slice_count = slice_count + 1

            if slice_count == number_of_slices_per_image:
                # Save the composite image
                compFilename = f"{outputFilename}{'-'}{compIndex}{'.jpg'}"
                composite_image.save(compFilename)

                total_slices = total_slices - number_of_slices_per_image

                if (total_slices < number_of_slices_per_image):
                    number_of_slices_per_image = total_slices

                # create a new image buffer
                adjusted_max_width = number_of_slices_per_image * self.sliceWidth * 2
                composite_image = Image.new('RGB', (adjusted_max_width, output_height))

                slice_count = 0

        # clean-up before exiting function
        # 1. close and delete the progress bar
        # 2. close the EXIF log file, if opened
        progressBar.close()
        del progressBar
