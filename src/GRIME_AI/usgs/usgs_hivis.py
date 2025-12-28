# usgs/usgs_hivis.py
import datetime
from PyQt5.QtWidgets import QMessageBox
from GRIME_AI.GRIME_AI_QMessageBox import GRIME_AI_QMessageBox
from .usgs_client import USGSClient

class USGS_HIVIS:

    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
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

    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    def init_camera_dictionary(self):
        # Legacy method existed; provide access via client. Return dict.
        try:
            return self._client._svc.camera_dictionary()
        except Exception:
            return {}

    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    def get_camera_dictionary(self):
        return self._client._svc.camera_dictionary()

    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    def get_camera_list(self):
        return self._client.get_sites()

    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
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


    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    def get_nwisID(self):
        return self._last_nwis_id

    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    def get_camName(self):
        return self._last_cam_name

    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    def get_camId(self):
        return self._last_cam_id

    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    def get_latest_image(self, siteName):
        code, pixmap = self._client.get_latest_pixmap(siteName)
        if code == 404:
            return 404, []
        return 0, pixmap

    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    def get_image_count(self, siteName, nwisID, startDate, endDate, startTime, endTime):
        """
        Build per-day HIVIS URLs using services.build_hivis_urls_and_count, call each URL,
        count files returned by each response, and return the running total.

        Returns:
            int: total number of files found across all per-day URLs
        """
        from urllib.request import Request, urlopen
        from urllib.error import HTTPError, URLError
        import json

        # Import the helper that builds per-day URLs and returns (count, urls)
        from GRIME_AI.usgs.usgs_service import build_hivis_urls_and_count

        try:
            # Build the per-day URLs
            _, urls = build_hivis_urls_and_count(
                endpoint="http://jj5utwupk5.execute-api.us-east-1.amazonaws.com/prod/listFiles",
                cam_id=siteName,  # assuming siteName maps to cam_id; adjust if needed
                start_date=startDate,
                end_date=endDate,
                start_time=startTime,
                end_time=endTime,
            )
        except Exception:
            msg = GRIME_AI_QMessageBox(
                'Images unavailable',
                'Failed to build request URLs for the site or for the time/date range specified.',
                QMessageBox.Close
            )
            msg.displayMsgBox()
            return 0

        total_count = 0
        per_url_counts = []

        # https://jj5utwupk5.execute-api.us-east-1.amazonaws.com/prod/listFiles?camId=CA_SALINAS_R_NR_BRADLEY_CA&after=2025-12-10&before=2025-12-10
        for url in urls:
            req = Request(url, headers={"User-Agent": "python-urllib/3"})
            print (f'URL: {url}\n')
            try:
                with urlopen(req, timeout=15) as resp:
                    # Parse JSON response
                    data = json.load(resp)

                    # Robust counting logic:
                    # - If the response is a list, that's the list of files
                    # - If it's a dict, try common keys ('files','items','results') that hold lists
                    # - If it has a numeric 'count' field, use that
                    # - Otherwise, fall back to 1 if non-empty dict, else 0
                    count = 0
                    if isinstance(data, list):
                        count = len(data)
                    elif isinstance(data, dict):
                        if 'files' in data and isinstance(data['files'], list):
                            count = len(data['files'])
                        elif 'items' in data and isinstance(data['items'], list):
                            count = len(data['items'])
                        elif 'results' in data and isinstance(data['results'], list):
                            count = len(data['results'])
                        elif 'count' in data and isinstance(data['count'], (int, float, str)):
                            try:
                                count = int(data['count'])
                            except Exception:
                                count = 0
                        else:
                            # If dict but none of the above, treat non-empty dict as 1 record
                            count = 1 if data else 0
                    else:
                        # Unexpected type (e.g., None), treat as zero
                        count = 0

                    total_count += count
                    per_url_counts.append((url, count))

            except HTTPError as e:
                # Server returned an error for this URL; log and continue
                print(f"HTTP error for {url}: {e.code} {e.reason}")
                per_url_counts.append((url, 0))
                continue
            except URLError as e:
                # Network error; log and continue
                print(f"Network error for {url}: {e.reason}")
                per_url_counts.append((url, 0))
                continue
            except json.JSONDecodeError:
                # Response was not JSON; log and continue
                print(f"Non-JSON response for {url}")
                per_url_counts.append((url, 0))
                continue
            except Exception as e:
                # Catch-all to avoid failing the whole loop
                print(f"Unexpected error for {url}: {e}")
                per_url_counts.append((url, 0))
                continue

        # Optional: debug print per-URL counts
        for u, c in per_url_counts:
            print(f"{u} -> {c} files")

        if total_count == 0:
            # Preserve original behavior: show message box when no images found
            msg = GRIME_AI_QMessageBox(
                'Images unavailable',
                'No images available for the site or for the time/date range specified.',
                QMessageBox.Close
            )
            msg.displayMsgBox()

        return total_count

    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    def download_images(self, siteName, startDate, endDate, startTime, endTime, saveFolder):
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

    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
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

    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
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

    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    def fetchListOfImages(self, siteName, after, before):
        # Delegate to client; return legacy string payload (already formatted)
        return self._client.fetch_list_of_images(siteName, after, before)

    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    def reformat_file(self, input_path, output_path):
        return self._client.reformat_file(input_path, output_path)
