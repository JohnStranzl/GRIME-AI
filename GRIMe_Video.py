import os
import cv2
import imageio

from GRIMe_Utils import GRIMe_Utils
from GRIMe_Color import GRIMe_Color
from GRIMe_QProgressWheel import QProgressWheel
import datetime

class GRIMe_Video:
    def __init__(self):
        self.className = "GRIMe_Video"

    # ======================================================================================================================
    #
    # ======================================================================================================================
    def createVideo(self, rootFolder):

        out = None

        myGRIMe_Color = GRIMe_Color()

        filePath = os.path.join(rootFolder, "Videos")
        if not os.path.exists(filePath):
            os.mkdir(filePath)

        # ONLY LOOK FOR FILES WITH THE FOLLOWING EXTENSIONS
        extensions = ('.jpg', '.jpeg', '.png')

        imageCount = GRIMe_Utils.getImageCount(rootFolder, extensions)

        progressBar = QProgressWheel(0, imageCount + 1)
        progressBar.show()

        for imageIndex, file in enumerate(os.listdir(rootFolder)):
            ext = os.path.splitext(file)[-1].lower()

            if ext in extensions:
                progressBar.setValue(imageIndex)

                img = myGRIMe_Color.loadColorImage(os.path.join(rootFolder, file))
                height, width, layers = img.shape

                # WE CAN'T OPEN THE VIDEO STREAM UNTIL WE KNOW THE SIZE OF ONE OF THE IMAGES WHICH ALSO ASSUMES THAT
                # ALL IMAGES ARE OF THE SAME SIZE.
                if out == None:
                    videoFile = 'Original_' + datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + '.mp4'
                    out = cv2.VideoWriter(filePath + '/' + videoFile, cv2.VideoWriter_fourcc(*'mp4v'), 15, (width, height))

                out.write(img)

        out.release()

        del progressBar


    # ======================================================================================================================
    #
    # ======================================================================================================================
    def createGIF(self, rootFolder):

        filePath = os.path.join(rootFolder, "Videos")
        if not os.path.exists(filePath):
            os.mkdir(filePath)

        # ONLY LOOK FOR FILES WITH THE FOLLOWING EXTENSIONS
        extensions = ('.jpg', '.jpeg', '.png')

        imageCount = GRIMe_Utils.getImageCount(rootFolder, extensions)

        progressBar = QProgressWheel(0, imageCount + 1)
        progressBar.show()

        filenames = []

        # RECURSE AND TRAVERSE FROM THE SPECIFIED FOLDER DOWN TO DETERMINE THE DATE RANGE FOR THE IMAGES FOUND
        files = GRIMe_Utils.getFileList(filePath, extensions, bFetchRecursive)

        for imageIndex, file in enumerate(files):
            ext = os.path.splitext(file)[-1].lower()

            if ext in extensions:
                progressBar.setValue(imageIndex)

                filenames.append(os.path.join(filePath, file))

        images = []
        gifFile = 'Original_' + datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + '.gif'
        for filename in filenames:
            images.append(imageio.imread(filename))

        gifFile = os.path.join(filePath, gifFile)
        imageio.mimsave(gifFile, images, duration=0.25)

        del progressBar

