from PIL import Image
import os


class GRIME_AI_CompositeSlices():
    def __init__(self):
        pass

    def crop_side(self, image, side, width, height):
        if side == "left":
            return image.crop((0, 0, width, height))
        elif side == "middle":
            left = (image.width - width) // 2
            return image.crop((left, 0, left + width, height))
        elif side == "right":
            return image.crop((image.width - width, 0, image.width, height))
        else:
            raise ValueError("Invalid side specified")

    def create_composite_image(self, imageList, output_path, side):
        # Open the first image to get its dimensions
        first_image = Image.open(imageList[0].fullPathAndFilename)
        output_width = first_image.width // len(imageList)
        output_height = first_image.height

        # Create a new image with the dimensions of the first image
        composite_image = Image.new('RGB', (first_image.width, output_height))

        # Paste each cropped image onto the composite image
        for i, image_file in enumerate(imageList):
            image = Image.open(image_file.fullPathAndFilename)

            # Crop the specified portion of the image.
            cropped_image = self.crop_side(image, side, output_width, output_height)

            # Paste the cropped image onto the composite image
            composite_image.paste(cropped_image, (i * output_width, 0))

        # Save the composite image
        composite_image.save(output_path)
        composite_image.show()
