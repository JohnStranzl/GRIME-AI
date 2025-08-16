import os
import cv2
import datetime

# VIDEO CREATION PACKAGES
# ----------------------------------------------------------------------------------------------------------------------
import imageio

#jes from moviepy.editor import ImageSequenceClip

#jes This line only if using PIL instead of imageio
#jes from PIL import Image


# GRIME-AI Classes
# ----------------------------------------------------------------------------------------------------------------------
from GRIME_AI_QProgressWheel import QProgressWheel
from GRIME_AI_Utils import GRIME_AI_Utils
from GRIME_AI_Color import GRIME_AI_Color
import GRIME_AI_Save_Utils


class GRIME_AI_Video:
    def __init__(self):
        self.className = "GRIME_AI_Video"

        from GRIME_AI_Save_Utils import GRIME_AI_Save_Utils
        self.myGRIMEAI_save_utils = GRIME_AI_Save_Utils()

    # ======================================================================================================================
    #
    # ======================================================================================================================
    def createVideo(self, rootFolder):

        out = None

        myGRIMe_Color = GRIME_AI_Color()

        filePath = self.myGRIMEAI_save_utils.create_video_folder(rootFolder)

        # ONLY LOOK FOR FILES WITH THE FOLLOWING EXTENSIONS
        extensions = ('.jpg', '.jpeg', '.png')

        myGRIMEAI_utils = GRIME_AI_Utils()
        imageCount = myGRIMEAI_utils.get_image_count(rootFolder, extensions)

        progressBar = QProgressWheel(0, imageCount)
        progressBar.show()

        for imageIndex, file in enumerate(os.listdir(rootFolder)):
            ext = os.path.splitext(file)[-1].lower()

            if ext in extensions:
                progressBar.setValue(imageIndex)

                img = myGRIMe_Color.loadColorImage(os.path.join(rootFolder, file))
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                height, width, layers = img.shape

                # WE CAN'T OPEN THE VIDEO STREAM UNTIL WE KNOW THE SIZE OF ONE OF THE IMAGES WHICH ALSO ASSUMES THAT
                # ALL IMAGES ARE OF THE SAME SIZE.
                if out == None:
                    videoFile = 'Original_' + datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + '.mp4'
                    out = cv2.VideoWriter(filePath + '/' + videoFile, cv2.VideoWriter_fourcc(*'mp4v'), 15, (width, height))

                out.write(img)

        out.release()

        progressBar.close()
        del progressBar


    # ======================================================================================================================
    #
    # ======================================================================================================================
    def createGIF(self, rootFolder):

        filePath = self.myGRIMEAI_save_utils.create_gif_folder(rootFolder)

        # ONLY LOOK FOR FILES WITH THE FOLLOWING EXTENSIONS
        extensions = ('.jpg', '.jpeg', '.png')

        myGRIMEAI_utils = GRIME_AI_Utils()
        imageCount = myGRIMEAI_utils.get_image_count(rootFolder, extensions)

        progressBar = QProgressWheel(0, 0)
        progressBar.show()

        filenames = []

        # RECURSE AND TRAVERSE FROM THE SPECIFIED FOLDER DOWN TO DETERMINE THE DATE RANGE FOR THE IMAGES FOUND
        file_count, files = myGRIMEAI_utils.getFileList(rootFolder, extensions)

        # APPEND THE SOURCE FOLDER PATH AND THE FILENAME
        for imageIndex, file in enumerate(files):
            if os.path.splitext(file)[-1].lower() in extensions:
                filenames.append(os.path.join(filePath, file))

        images = []
        gifFile = 'Original_' + datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + '.gif'

        progressBar.setRange(0, len(filenames)-1)

        for file_index, filename in enumerate(filenames):
            progressBar.setValue(file_index)

            images.append(imageio.v3.imread(filename))

            #jes This line only if using PIL instead of imageio
            #jes images.append(Image.open(filename))

            #jes This line only if using moviepy instead of imageio
            #jes images.append(os.path.join(rootFolder, filename))  # Changed line

        gifFile = os.path.join(filePath, gifFile)
        imageio.v3.imwrite(gifFile, images, duration=0.25)

        #jes This line only if using PIL instead of imageio
        #jes images[0].save(gifFile, save_all=True, append_images=images[1:], duration=250, loop=0)

        #jes This line only if using PIL instead of imageio
        #jes clip = ImageSequenceClip(images, fps=4)
        #jes clip.write_gif(gifFile)  # Changed line

        progressBar.close()
        del progressBar
