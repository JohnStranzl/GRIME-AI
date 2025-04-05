import os
import sys
import cv2
import supervision as sv

try:
    import torch

    print("GRIME AI Deep Learning: PyTorch imported successfully.")
except ImportError as e:
    print("GRIME AI Deep Learning: Error importing PyTorch:", e)
    # Remove the faulty package from sys.modules to prevent further issues
    if 'torch' in sys.modules:
        del sys.modules['torch']

# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from GRIME_AI_QProgressWheel import QProgressWheel
import shutil


class GRIME_AI_DeepLearning:

    # ======================================================================================================================
    #
    # ======================================================================================================================
    def __init__(self):
        pass

    # ======================================================================================================================
    #
    # ======================================================================================================================
    def SAM_001(self, modelSettings, dailyImagesList):
        # ====================================================================================================

        # ====================================================================================================
        DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        #CHECKPOINT_PATH = 'sam_vit_h_4b8939.pth'
        #CHECKPOINT_PATH = 'sam_model.pth'

        CHECKPOINT_PATH = modelSettings.model_file
        MODEL_TYPE = "vit_h"
        sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
        sam.to(device=DEVICE)

        # ====================================================================================================
        mask_generator = SamAutomaticMaskGenerator(sam)

        progressBar = QProgressWheel()
        progressBar.setRange(0, len(dailyImagesList.getVisibleList()) + 1)
        progressBar.show()

        for imageIndex, image in enumerate(dailyImagesList.getVisibleList()):
            # ====================================================================================================
            progressBar.setWindowTitle(image.fullPathAndFilename)
            progressBar.setValue(imageIndex)
            progressBar.repaint()

            # ====================================================================================================
            # READ IMAGE AND CONVERT FROM BGR TO RGB
            image_bgr = cv2.imread(image.fullPathAndFilename)
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            masks = mask_generator.generate(image_rgb)

            # ====================================================================================================
            # SINCE THE FILES CAN BE FROM DIFFERENT FOLDERS, WE HAVE TO CHECK TO SEE IF THE SUBFOLDER INTO WHICH
            # WE WANT TO PLACE THE RESULTS EXISTS EACH TIME WE PROCESS A FILE.
            filename, segmentationSubFolder = self.createSegmentationFolders(image.fullPathAndFilename)

            # ====================================================================================================
            mask_annotator = sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX)
            detections = sv.Detections.from_sam(masks)
            annotated_image = mask_annotator.annotate(image_rgb, detections)

            # sv.plot_image(annotated_image)
            saveFilename = os.path.join(segmentationSubFolder, f'{filename}_segmented.jpg')
            cv2.imwrite(saveFilename, annotated_image)

            # ====================================================================================================
            # 1. SAVE A COPY OF THE ORIGINAL FILE TO THE SEGMENTATION FOLDER
            # 2. SAVE THE MASKS INTO A FOLDER WHERE THE FOLDERNAME IS THE NAME OF THE FILE
            if modelSettings.getSaveOriginalModelMask():
                shutil.copy(image.fullPathAndFilename, segmentationSubFolder)

            if modelSettings.getSaveModelMasks():
                for i, mask in enumerate(masks):
                    saveFilename = os.path.join(segmentationSubFolder, f'{filename}_mask_{i}.png')
                    cv2.imwrite(saveFilename, mask['segmentation'] * 255)  # Convert mask to 8-bit image

        progressBar.close()
        del progressBar

        return


    # ======================================================================================================================
    #
    # ======================================================================================================================
    def SAM_002(self, modelSettings, dailyImagesList):
        from ultralytics import SAM
        import supervision as sv
        import matplotlib.pyplot as plt

        # ====================================================================================================
        #
        # ====================================================================================================
        DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        CHECKPOINT_PATH = 'sam_vit_h_4b8939.pth'
        MODEL_TYPE = "vit_h"
        sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
        sam.to(device=DEVICE)

        mask_predictor = SamPredictor(sam)

        progressBar = QProgressWheel()
        progressBar.setRange(0, len(dailyImagesList.getVisibleList()) + 1)
        progressBar.show()

        # ====================================================================================================
        ###mask_generator = SamAutomaticMaskGenerator(sam)

        for image in dailyImagesList.getVisibleList():
            # ====================================================================================================
            progressBar.setWindowTitle(IMAGE_PATH)
            progressBar.setValue(imageIndex)
            progressBar.repaint()

            # ====================================================================================================
            # READ IMAGE AND CONVERT FROM BGR TO RGB
            image_bgr = cv2.imread(image.fullPathAndFilename)
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            mask_predictor.set_image(image_rgb)

            # ====================================================================================================
            # SINCE THE FILES CAN BE FROM DIFFERENT FOLDERS, WE HAVE TO CHECK TO SEE IF THE SUBFOLDER INTO WHICH
            # WE WANT TO PLACE THE RESULTS EXISTS EACH TIME WE PROCESS A FILE.
            filename, segmentationSubFolder = self.createSegmentationFolders(image.fullPathAndFilename)

            # ====================================================================================================
            box = np.array([70, 247, 626, 926])
            masks, scores, logits = mask_predictor.predict(box=box, multimask_output=True )

            # ====================================================================================================
            # 1. SAVE A COPY OF THE ORIGINAL FILE TO THE SEGMENTATION FOLDER
            # 2. SAVE THE MASKS INTO A FOLDER WHERE THE FOLDERNAME IS THE NAME OF THE FILE
            shutil.copy(image.fullPathAndFilename, segmentationSubFolder)
            for i, mask in enumerate(masks):
                saveFilename = os.path.join(segmentationSubFolder, f'{filename}_mask_{i}.png')
                cv2.imwrite(saveFilename, mask * 255)  # Convert mask to 8-bit image

        return

    # ======================================================================1================================================
    #
    # ======================================================================================================================
    def createSegmentationFolders(self, fullPathAndFilename):
        fullPath, fullFilename = os.path.split(fullPathAndFilename)
        filename, fileext = os.path.splitext(fullFilename)

        segmentationMainFolder = os.path.join(fullPath, 'Segmented Images')
        if not os.path.exists(segmentationMainFolder):
            os.makedirs(segmentationMainFolder)

        segmentationSubFolder = os.path.join(segmentationMainFolder, filename)
        if not os.path.exists(segmentationSubFolder):
            os.makedirs(segmentationSubFolder)

        return filename, segmentationSubFolder


