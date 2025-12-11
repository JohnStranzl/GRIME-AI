import os
import json
import urllib.request
import requests
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import date, time, datetime, timedelta

from .types import CameraInfo, LatestImage

ENDPOINT = "https://jj5utwupk5.execute-api.us-east-1.amazonaws.com"
IMAGE_ENDPOINT = "https://usgs-nims-images.s3.amazonaws.com/overlay"

# ================================================================================
# ================================================================================
#                               class USGSService
# ================================================================================
# ================================================================================
class USGSService:
    """
    Pure service: fetches camera metadata, lists, images, and USGS discharge data.
    No Qt. No GUI side-effects. Raises exceptions on hard failures.
    """

    # --------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------
    def __init__(self):
        self._camera_dict: Dict[str, dict] = {}
        self._site_count: int = 0
        self._nwis_id: Optional[str] = None
        self._cam_id: Optional[str] = None
        self._cam_name: Optional[str] = None

    # --------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------
    def initialize(self) -> None:
        uri = f"{ENDPOINT}/prod/cameras?enabled=true"

        # WITH urllib, IT THROWS AN EXCEPTION INSTEAD OF RETURNING AN ERROR CODE WHEN IT DETECTS A NETWORK FAILURE
        try:
            data = urllib.request.urlopen(uri).read()

            camera_data = json.loads(data.decode("utf-8"))

            cam_dict: Dict[str, dict] = {}
            for element in camera_data:
                if element.get("locus") == "aws" and not element.get("hideCam", True):
                    cam_id = element.get("camId")
                    if isinstance(cam_id, str):
                        cam_dict[cam_id] = element
            self._camera_dict = cam_dict
        except Exception as e:
            self._camera_dict = {}

        self._site_count = len(self._camera_dict)

    # --------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------
    def camera_dictionary(self) -> Dict[str, dict]:
        return self._camera_dict

    # ------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------
    def camera_list(self) -> List[str]:
        return sorted(self._camera_dict.keys())

    # ------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------
    def camera_info(self, camera_id: str) -> CameraInfo:
        cam = self._camera_dict.get(camera_id)
        if cam is None:
            return CameraInfo(camera_id, None, None, None, None, None, None)
        self._nwis_id = cam.get("nwisId")
        self._cam_id = cam.get("camId")
        self._cam_name = cam.get("camName")
        return CameraInfo(
            cam_id=cam.get("camId", camera_id),
            nwis_id=cam.get("nwisId"),
            cam_name=cam.get("camName"),
            lat=self._to_float(cam.get("lat")),
            lng=self._to_float(cam.get("lng")),
            tz=cam.get("tz"),
            description=cam.get("camDesc")
        )

    # ------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------
    def latest_image(self, site_name: str) -> LatestImage:
        url = f"{IMAGE_ENDPOINT}/{site_name}/{site_name}_newest.jpg"
        r = requests.get(url, stream=True)
        if r.status_code == 404:
            return LatestImage(error_code=404, content=None)
        r.raise_for_status()
        content = urllib.request.urlopen(url).read()
        return LatestImage(error_code=0, content=content)

    # ------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------
    def image_count(self, site_name: str, start_date: date, end_date: date,
                    start_time: time, end_time: time,
                    progress: Optional[callable] = None) -> int:
        names = self._collect_image_names(site_name, start_date, end_date, start_time, end_time, progress)
        return len(names)

    # ------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------
    def download_images(self, site_name: str, start_date: date, end_date: date,
                        start_time: time, end_time: time, save_folder: str,
                        progress: Optional[callable] = None) -> Tuple[int, int]:
        os.makedirs(save_folder, exist_ok=True)
        names = self._collect_image_names(site_name, start_date, end_date, start_time, end_time, progress)
        downloaded, missing = 0, 0
        total = len(names)
        for idx, image in enumerate(names):
            if progress:
                progress(idx, total, image)
            if not image or image == "[]":
                continue
            try:
                file_url = f"{IMAGE_ENDPOINT}/{site_name}/{image}"
                dst = os.path.join(save_folder, image)
                if not os.path.isfile(dst):
                    urllib.request.urlretrieve(file_url, dst)
                downloaded += 1
            except Exception:
                missing += 1
        return downloaded, missing

    # ------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------
    def fetch_stage_and_discharge(self, nwis_id: str, site_name: str,
                                  start_date: date, end_date: date,
                                  start_time: time, end_time: time,
                                  save_folder: str) -> Tuple[str, str]:
        os.makedirs(save_folder, exist_ok=True)
        base = "https://waterservices.usgs.gov/nwis/iv/?format=rdb,1.0&sites="
        url = f"{base}{nwis_id}&startDT={start_date.strftime('%Y-%m-%d')}&endDT={end_date.strftime('%Y-%m-%d')}&siteStatus=all"
        timestamp = f"{start_date.strftime('%Y-%m-%d')}T{start_time.strftime('%H%M')} - {end_date.strftime('%Y-%m-%d')}T{end_time.strftime('%H%M')}"
        txt_path = os.path.join(save_folder, f"{site_name} - {nwis_id} - {timestamp}.txt")
        csv_path = os.path.join(save_folder, f"{site_name} - {nwis_id} - {timestamp}.csv")
        with urllib.request.urlopen(url) as resp:
            _ = resp.read()
        urllib.request.urlretrieve(url, txt_path)
        self._reformat_file(txt_path, csv_path)
        return txt_path, csv_path

    # ------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------
    def _collect_image_names(self, site_name: str, start_date: date, end_date: date,
                             start_time: time, end_time: time,
                             progress: Optional[callable]) -> List[str]:
        names: List[str] = []
        days = (end_date - start_date).days + 1
        for i in range(days):
            if progress:
                progress(i, days, None)
            after, before = self._build_image_datetime_filter(i, start_date, start_time, end_time)
            text = self._fetch_list_of_images(site_name, after, before)
            if text and text != "[]":
                cleaned = text.replace("[", "").replace("]", "").replace('"', "")
                if cleaned:
                    parts = [p for p in cleaned.split(",") if p]
                    names.extend(parts)
        return names

    # ------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------
    def _build_image_datetime_filter(self, index: int, start_date: date,
                                     start_time: time, end_time: time) -> Tuple[str, str]:
        start_day = start_date + timedelta(days=index)
        if start_time.hour == 0 and start_time.minute == 0 and end_time.hour == 0 and end_time.minute == 0:
            day_start = datetime.combine(start_day, time(0, 0, 0))
            day_end = datetime.combine(start_day, time(23, 59, 59))
        else:
            day_start = datetime.combine(start_day, start_time)
            day_end = datetime.combine(start_day, end_time)
        after_dt = day_start - timedelta(seconds=30)
        before_dt = day_end + timedelta(seconds=30)
        after = f"&after={after_dt.strftime('%Y-%m-%d:%H:%M:%S')}"
        before = f"&before={before_dt.strftime('%Y-%m-%d:%H:%M:%S')}"
        return after, before

    # ------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------
    def _fetch_list_of_images(self, site_name: str, after: str, before: str) -> str:
        url = f"{ENDPOINT}/prod/listFiles?camId={site_name}{after}{before}"
        resp = requests.get(url)
        resp.raise_for_status()
        return resp.text

    # ------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------
    def _reformat_file(self, input_txt: str, output_csv: str) -> None:
        df = pd.read_csv(input_txt, delimiter="\t", comment="#")
        df = df[~df["agency_cd"].astype(str).str.contains("5s")]
        df.to_csv(output_csv, index=False)

    # ------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------
    @staticmethod
    def _to_float(x) -> Optional[float]:
        try:
            return float(x) if x is not None else None
        except Exception:
            return None
