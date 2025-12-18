# usgs/shim.py
import datetime
from PyQt5.QtWidgets import QMessageBox
from GRIME_AI.GRIME_AI_QMessageBox import GRIME_AI_QMessageBox
from .client import USGSClient

class USGS_NIMS_Shim:

    def __init__(self):
        self._client = USGSClient()
        self._last_cam_id = None
        self._last_nwis_id = None
        self._last_cam_name = None
        try:
            self._client.initialize()
        except Exception:
            msg = GRIME_AI_QMessageBox('USGS NIMS Error',
                                       'Unable to access USGS NIMS Database!')
            msg.displayMsgBox()

    # --- Legacy interface parity ---

    def init_camera_dictionary(self):
        # Legacy method existed; provide access via client. Return dict.
        try:
            return self._client._svc.camera_dictionary()
        except Exception:
            return {}

    def get_camera_dictionary(self):
        return self._client._svc.camera_dictionary()

    def get_camera_list(self):
        return self._client.get_sites()

    def get_camera_info(self, camera_id):
        lines = self._client.get_camera_info_lines(camera_id)
        # Track legacy attributes
        self._last_cam_id = camera_id
        self._last_nwis_id = None
        self._last_cam_name = None
        for line in lines:
            if line.startswith("nwisId:"):
                self._last_nwis_id = line.split(":", 1)[1].strip()
            elif line.startswith("camName:"):
                self._last_cam_name = line.split(":", 1)[1].strip()
        return lines if lines else ["No information available for this site."]

    def get_nwisID(self):
        return self._last_nwis_id

    def get_camName(self):
        return self._last_cam_name

    def get_camId(self):
        return self._last_cam_id

    def get_latest_image(self, siteName):
        code, pixmap = self._client.get_latest_pixmap(siteName)
        if code == 404:
            return 404, []
        return 0, pixmap

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
            downloaded, missing = self._client.download_images(
                siteName, startDate, endDate, startTime, endTime, saveFolder
            )
            return downloaded, missing
        except Exception:
            msg = GRIME_AI_QMessageBox('Images unavailable',
                                       'One or more images reported as available by NIMS are not available.',
                                       QMessageBox.Close)
            msg.displayMsgBox()
            return 0, 0

    def fetchStageAndDischarge(self, nwisID, siteName, startDate, endDate, startTime, endTime, saveFolder):
        try:
            return self._client.fetch_stage_and_discharge(
                nwisID, siteName, startDate, endDate, startTime, endTime, saveFolder
            )
        except Exception:
            msg = GRIME_AI_QMessageBox('USGS - Retrieval Error',
                                       'Unable to retrieve data from the USGS site.',
                                       QMessageBox.Close)
            msg.displayMsgBox()
            return None, None

    # --- Legacy helpers preserved for compatibility ---

    def buildImageDateTimeFilter(self, index, startDate, endDate, startTime, endTime):
        startDay = startDate + datetime.timedelta(days=index)
        if startTime.hour == 0 and startTime.minute == 0 and endTime.hour == 0 and endTime.minute == 0:
            day_start = datetime.datetime.combine(startDay, datetime.time(0, 0, 0))
            day_end = datetime.datetime.combine(startDay, datetime.time(23, 59, 59))
        else:
            day_start = datetime.datetime.combine(startDay, startTime)
            day_end = datetime.datetime.combine(startDay, endTime)
        after_dt = day_start - datetime.timedelta(seconds=30)
        before_dt = day_end + datetime.timedelta(seconds=30)
        return "&after=" + after_dt.strftime("%Y-%m-%d:%H:%M:%S"), "&before=" + before_dt.strftime("%Y-%m-%d:%H:%M:%S")

    def fetchListOfImages(self, siteName, after, before):
        # Delegate to client; return legacy string payload (already formatted)
        return self._client.fetch_list_of_images(siteName, after, before)

    def reformat_file(self, input_path, output_path):
        return self._client.reformat_file(input_path, output_path)
