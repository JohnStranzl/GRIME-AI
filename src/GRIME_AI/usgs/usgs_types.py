from dataclasses import dataclass
from typing import List, Optional

@dataclass(frozen=True)
class CameraInfo:
    cam_id: str
    nwis_id: Optional[str]
    cam_name: Optional[str]
    lat: Optional[float]
    lng: Optional[float]
    tz: Optional[str]
    description: Optional[str]

@dataclass(frozen=True)
class ImageList:
    names: List[str]

@dataclass(frozen=True)
class LatestImage:
    error_code: int        # 0 = ok, 404 = not found
    content: Optional[bytes]  # raw bytes of the image
