MSTFolder = 'C:/Users/Astrid Haugen/Documents/000 - MeanStride'

# ======================================================================================================================
#
# ======================================================================================================================
class MSTExtractFrames():
    def __init__(self, inputVideo, outFolder, start, end, *args, **kwargs):
        super(MSTExtractFrames, self).__init__(*args, **kwargs)
        self.inputVideo = inputVideo
        self.outFolder = outFolder
        self.vidcap = cv2.VideoCapture(inputVideo)
        self.frameCount = self.vidcap.get(cv2.CAP_PROP_FRAME_COUNT)
        self.fps = self.vidcap.get(cv2.CAP_PROP_FPS)
        self.frameFailures = 0

        self.startFrame = int(start)
        if end == 'ALL':
            self.endFrame = int(self.frameCount - 1)
        else:
            self.endFrame = int(end)

        # Now, go do it!
        self.makeOutFolder()

    def makeOutFolder(self):
        if not os.path.exists(self.outFolder):
            os.makedirs(self.outFolder)

    def process_frame(self, frameIndex):
        print("Get Frame at frameIndex", frameIndex)

        self.vidcap.set(cv2.CAP_PROP_POS_FRAMES, frameIndex)
        hasFrames, image1 = self.vidcap.read()
        num = str(frameIndex)
        znum = num.zfill(4)
        fn = "image" + znum + ".jpg"
        path = self.outFolder + "/" + fn
        cv2.imwrite(path, image1)

    def process_all_frames(self):
        for frameIndex in range(self.startFrame, self.endFrame):
            self.process_frame(frameIndex)


    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    def MSTFetchVideoFilenames(self):
        self.lineEditHardDriveFolder.setText(MSTFolder)
        videoFilePath = Path(MSTFolder)
        # JES videoFileList = [str(pp) for pp in videoFilePath.glob("**/*.mp4")]
        videoFileList = [str(pp) for pp in videoFilePath.glob("*.mp4")]

        self.MSTspinBoxSelectImage.setMaximum(len(videoFileList))
        self.MSTspinBoxSelectImage.setValue(1)

        for videoFilename in videoFileList:
            self.MSTlistWidgetVideoFiles.addItem(os.path.basename(videoFilename))


    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    def MSTExtractFrames(self):
        videoFrameFolder = MSTFolder + '/' + self.MSTlineEditFrameFolder.text()
        video = MSTFolder + '/' + self.MSTlistWidgetVideoFiles.selectedItems()[0].text()

        BVS = MSTExtractFrames(video, videoFrameFolder, 0, 'ALL')
        BVS.process_all_frames()


    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    def MSTFrameCountUpdate(self):
        frameFolder = self.lineEditHardDriveFolder.text() + '/' + self.MSTlineEditFrameFolder.text()
        nFrameCount = MSTGetFrameCount(self, frameFolder)

        strFrameCount = 'Frame Count: ' + nFrameCount.__str__()
        self.MSTFrameCount.setText(strFrameCount)

        self.MSTspinBoxSelectImage.setMaximum(nFrameCount)
        self.MSTspinBoxSelectImage.setValue(1)

        global gFrameCount
        gFrameCount = nFrameCount

        self.spinBoxDailyImage.setMinimum(1)
        self.spinBoxDailyImage.setMaximum(nFrameCount)
        self.spinBoxDailyImage.setValue(1)


    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    def MSTStop(self):
        global nStop
        nStop = 1


# ======================================================================================================================
#
# ======================================================================================================================
def MSTGetFrameCount(self, frameFolder):
    if os.path.exists(frameFolder):
        frameFolderPath = Path(frameFolder)
        # JES videoFileList = [str(pp) for pp in frameFolderPath.g("**/*.jpg")]
        videoFileList = [str(pp) for pp in frameFolderPath.glob("*.jpg")]
        nFrameCount = len(videoFileList)
    else:
        nFrameCount = 0

    return nFrameCount

