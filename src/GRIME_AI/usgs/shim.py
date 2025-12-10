# usgs/shim.py
from PyQt5.QtWidgets import QMessageBox
from GRIME_AI.GRIME_AI_QMessageBox import GRIME_AI_QMessageBox
from .client import USGSClient

class USGS_NIMS_Shim:
    """
    Provides the same method names MainWindow expects,
    delegates to USGSClient, and shows old message boxes on errors
    until the GUI is migrated to handle errors itself.
    """

    def __init__(self):
        self._client = USGSClient()
        try:
            self._client.initialize()
        except Exception as e:
            msg = GRIME_AI_QMessageBox('USGS NIMS Error', 'Unable to access USGS NIMS Database!')
            msg.displayMsgBox()

    def get_camera_dictionary(self):
        # Optional: expose dictionary if needed
        return self._client._svc.camera_dictionary()

    def get_camera_list(self):
        return self._client.get_sites()

    def get_camera_info(self, cam_id):
        return self._client.get_camera_info_lines(cam_id)

    def get_camId(self):
        # For parity, return the cam_id last requested; if needed store it here
        return cam_id  # or track internally via client

    def get_latest_image(self, site_name):
        code, pix = self._client.get_latest_pixmap(site_name)
        if code == 404:
            return 404, []
        return 0, pix

    def get_image_count(self, siteName, nwisID, startDate, endDate, startTime, endTime):
        try:
            return self._client.image_count(siteName, startDate, endDate, startTime, endTime)
        except Exception:
            msg = GRIME_AI_QMessageBox('Images unavailable',
                                       'No images available for the site or for the time/date range specified.',
                                       QMessageBox.Close)
            msg.displayMsgBox()
            return 0

    def download_images(self, siteName, nwisID, startDate, endDate, startTime, endTime, saveFolder):
        try:
            downloaded, missing = self._client.download_images(siteName, startDate, endDate, startTime, endTime, saveFolder)
            # Old behavior used progress wheel + silent completion; keep silent for now
            return downloaded, missing
        except Exception:
            msg = GRIME_AI_QMessageBox('Images unavailable',
                                       'One or more images reported as available by NIMS are not available.',
                                       QMessageBox.Close)
            msg.displayMsgBox()
            return 0, 0

    def fetchStageAndDischarge(self, nwisID, siteName, startDate, endDate, startTime, endTime, saveFolder):
        try:
            return self._client.fetch_stage_and_discharge(nwisID, siteName, startDate, endDate, startTime, endTime, saveFolder)
        except Exception:
            msg = GRIME_AI_QMessageBox('USGS - Retrieval Error',
                                       'Unable to retrieve data from the USGS site.',
                                       QMessageBox.Close)
            msg.displayMsgBox()
            return None, None
