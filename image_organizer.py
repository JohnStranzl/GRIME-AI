# image_organizer_core.py
import csv
import re
import piexif
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

try:
    from PIL import Image, ExifTags
except Exception:
    Image = None
    ExifTags = None

from GRIME_AI_logger import info as _info, debug as _debug, err as _err, warn as _warn

# ---------- constants ----------
IMG_EXTS = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'}
EXIF_TAGS = {v: k for k, v in getattr(ExifTags, 'TAGS', {}).items()} if ExifTags else {}

# Windows XP EXIF tags (UTF-16LE)
XPTitle    = 0x9C9B
XPComment  = 0x9C9C
XPKeywords = 0x9C9E

# ---------- XP helpers ----------
def _encode_xp(s: str) -> bytes:
    return (s or "").encode("utf-16le") + b"\x00\x00"

def _decode_xp(value) -> str:
    try:
        if isinstance(value, bytes):
            return value.decode('utf-16le', errors='ignore').rstrip('\x00')
        if isinstance(value, (list, tuple)):
            return bytes(value).decode('utf-16le', errors='ignore').rstrip('\x00')
        if isinstance(value, str):
            return value
    except Exception:
        return ""
    return ""

# ---------- XMP parsing ----------
def read_xmp_chunks(path: Path) -> Optional[str]:
    try:
        with open(path, 'rb') as f:
            data = f.read()
        start = data.find(b"<x:xmpmeta")
        if start == -1: return None
        end = data.find(b"</x:xmpmeta>", start)
        if end == -1: return None
        return data[start:end+12].decode('utf-8', errors='ignore')
    except Exception as e:
        _debug(f"read_xmp_chunks error for {path}: {e}")
        return None

def parse_xmp_for_fields(xmp_xml: str) -> Dict[str, object]:
    out = {"Title": "", "Subject": "", "Comments": "", "Tags": [], "Rights": ""}
    try:
        import xml.etree.ElementTree as ET
        ns = {
            'dc': 'http://purl.org/dc/elements/1.1/',
            'rdf': 'http://www.w3.org/1999/02/22-rdf-syntax-ns#',
            'xmp': 'http://ns.adobe.com/xap/1.0/'
        }
        root = ET.fromstring(xmp_xml)
        g = lambda p: root.find(p, ns)

        t = g('.//dc:title//rdf:Alt/rdf:li')
        if t is not None and t.text: out["Title"] = t.text.strip()
        d = g('.//dc:description//rdf:Alt/rdf:li')
        if d is not None and d.text: out["Comments"] = d.text.strip()
        bag = g('.//dc:subject/rdf:Bag')
        if bag is not None:
            tags = []
            for li in bag.findall('rdf:li', ns):
                if li.text: tags.append(li.text.strip())
            out["Tags"] = tags
        lab = g('.//xmp:Label')
        if lab is not None and lab.text: out["Subject"] = lab.text.strip()
        r = g('.//dc:rights//rdf:Alt/rdf:li')
        if r is not None and r.text: out["Rights"] = r.text.strip()
    except Exception as e:
        _debug(f"parse_xmp_for_fields error: {e}")
    return out

# ---------- metadata read ----------
def read_image_metadata(path: Path) -> Dict[str, object]:
    """
    Returns dict with (among others):
      - DateTimeFromMetadata (YYYY-MM-DD HH:MM:SS) or ""
      - HasSeconds (bool): whether EXIF originally had seconds precision
      - Title, Subject, Comments
      - Tags (list[str])
      - Exif (raw map)
      - CopyrightExisting
    """
    meta: Dict[str, object] = {
        "DateTimeFromMetadata": "",
        "HasSeconds": False,
        "Title": "",
        "Subject": "",
        "Comments": "",
        "Tags": [],
        "Exif": {},
        "CopyrightExisting": ""
    }
    exif_dt = None
    try:
        if Image is not None:
            with Image.open(str(path)) as img:
                exif_raw = img._getexif() or {}
                for tag_id, val in exif_raw.items():
                    name = ExifTags.TAGS.get(tag_id, str(tag_id))
                    meta["Exif"][name] = val

                # Prefer DateTimeOriginal, then Digitized, then DateTime
                for key in ("DateTimeOriginal", "DateTimeDigitized", "DateTime"):
                    if key in meta["Exif"]:
                        dt_str = meta["Exif"][key]
                        if isinstance(dt_str, bytes): dt_str = _decode_xp(dt_str)
                        if isinstance(dt_str, str):
                            if re.search(r"\d{4}:\d{2}:\d{2}\s+\d{2}:\d{2}:\d{2}$", dt_str.strip()):
                                meta["HasSeconds"] = True
                            try:
                                dt_norm = dt_str.replace(':', '-', 2)
                                exif_dt = datetime.strptime(dt_norm, "%Y-%m-%d %H:%M:%S")
                                break
                            except Exception:
                                pass

                meta["Title"]    = _decode_xp(meta["Exif"].get("XPTitle", "")) or meta["Exif"].get("ImageDescription", "")
                meta["Comments"] = _decode_xp(meta["Exif"].get("XPComment", ""))
                meta["Subject"]  = _decode_xp(meta["Exif"].get("XPSubject", ""))

                tags = _decode_xp(meta["Exif"].get("XPKeywords", ""))
                if tags:
                    parts = re.split(r'[;,]\s*', tags)
                    meta["Tags"] = [p for p in parts if p]

                cpy = meta["Exif"].get("Copyright")
                if isinstance(cpy, bytes): cpy = _decode_xp(cpy)
                if isinstance(cpy, str): meta["CopyrightExisting"] = cpy.strip()

                if hasattr(img, "info") and isinstance(img.info, dict):
                    if not meta["Comments"] and isinstance(img.info.get("Comment"), str):
                        meta["Comments"] = img.info["Comment"]
                    if not meta["Subject"] and isinstance(img.info.get("Subject"), str):
                        meta["Subject"] = img.info["Subject"]
                    if not meta["CopyrightExisting"] and isinstance(img.info.get("copyright"), str):
                        meta["CopyrightExisting"] = img.info["copyright"].strip()
    except Exception as e:
        _debug(f"read_image_metadata EXIF error for {path}: {e}")

    try:
        x = read_xmp_chunks(path)
        if x:
            xm = parse_xmp_for_fields(x)
            if not meta["Title"] and xm.get("Title"): meta["Title"] = xm["Title"]
            if not meta["Subject"] and xm.get("Subject"): meta["Subject"] = xm["Subject"]
            if not meta["Comments"] and xm.get("Comments"): meta["Comments"] = xm["Comments"]
            if xm.get("Tags"):
                existing = {t.lower() for t in meta["Tags"]}
                for t in xm["Tags"]:
                    if t.lower() not in existing:
                        meta["Tags"].append(t)
            if not meta["CopyrightExisting"] and xm.get("Rights"):
                meta["CopyrightExisting"] = xm["Rights"]
    except Exception as e:
        _debug(f"read_image_metadata XMP error for {path}: {e}")

    if exif_dt:
        meta["DateTimeFromMetadata"] = exif_dt.strftime("%Y-%m-%d %H:%M:%S")
    return meta

# ---------- discovery & file ops ----------
def iter_images(src: Path, recursive: bool) -> List[Path]:
    files: List[Path] = []
    if recursive:
        for p in src.rglob('*'):
            if p.is_file() and p.suffix.lower() in IMG_EXTS:
                files.append(p)
    else:
        for p in src.iterdir():
            if p.is_file() and p.suffix.lower() in IMG_EXTS:
                files.append(p)
    return files

def scan_for_datetime_presence(src_folder: Path, recursive: bool) -> Tuple[List[Path], List[Path]]:
    if not src_folder.exists():
        raise FileNotFoundError(f"Folder does not exist: {src_folder}")
    has_dt, missing_dt = [], []
    for p in iter_images(src_folder, recursive):
        try:
            m = read_image_metadata(p)
            if m.get("DateTimeFromMetadata"):
                has_dt.append(p)
            else:
                missing_dt.append(p)
        except Exception:
            missing_dt.append(p)
    return has_dt, missing_dt

def move_files_to_subfolder(paths: List[Path], base_dir: Path) -> Path:
    if not paths:
        sub = base_dir / "_NoDateTaken"
        sub.mkdir(parents=True, exist_ok=True)
        return sub
    sub = base_dir / f"_NoDateTaken_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    sub.mkdir(parents=True, exist_ok=True)
    for p in paths:
        try:
            target = sub / p.name
            i = 1
            while target.exists():
                target = sub / f"{p.stem}_{i:03d}{p.suffix}"
                i += 1
            p.rename(target)
            _info(f"Moved (no date): {p.name} -> {target.name}")
        except Exception as e:
            _warn(f"Move failed for {p}: {e}")
    return sub

# ---------- filename & writes ----------
def choose_dt_for_filename(path: Path, meta: Dict[str, object]) -> Tuple[datetime, bool]:
    if meta.get("DateTimeFromMetadata"):
        try:
            dt = datetime.strptime(meta["DateTimeFromMetadata"], "%Y-%m-%d %H:%M:%S")
            return dt, True
        except Exception:
            pass
    return datetime.fromtimestamp(path.stat().st_mtime), False

def make_filename(dt: datetime, site: str, include_seconds: bool, ext: str,
                  force_seconds: bool = False, counter: Optional[int] = None) -> str:
    fmt = "%Y%m%d_%H%M%S" if (include_seconds or force_seconds) else "%Y%m%d_%H%M"
    ts = dt.strftime(fmt)
    site_part = f"_{site.strip()}" if site and site.strip() else ""
    suffix = f"_{counter:02d}" if (counter is not None and counter > 1) else ""
    return f"{ts}{site_part}{suffix}{ext}"

def safe_rename(path: Path, new_name: str) -> Path:
    target = path.with_name(new_name)
    if not target.exists():
        path.rename(target); return target
    stem, ext = target.stem, target.suffix
    i = 1
    while True:
        cand = path.with_name(f"{stem}_{i:03d}{ext}")
        if not cand.exists():
            path.rename(cand); return cand
        i += 1

def write_copyright_exif(path: Path, holder: str, skip_if_exists: bool = True) -> None:
    if not holder or path.suffix.lower() not in {'.jpg', '.jpeg', '.tif', '.tiff'}:
        return
    try:
        exif = piexif.load(str(path))
        if skip_if_exists:
            existing = exif['0th'].get(piexif.ImageIFD.Copyright, b"")
            if isinstance(existing, (bytes, bytearray)) and existing.strip():
                _debug(f"Existing copyright present; skipping write for {path}")
                return
        exif['0th'][piexif.ImageIFD.Copyright] = holder.encode("utf-8", errors="ignore")
        piexif.insert(piexif.dump(exif), str(path))
        _debug(f"Wrote Copyright to {path}")
    except Exception as e:
        _warn(f"Failed to write Copyright EXIF for {path}: {e}")

def compose_final_comments(original_comments: str, title: str, site_info: str) -> str:
    parts = []
    if original_comments.strip():
        parts.append(original_comments.strip())
    if title.strip():
        parts.append(title.strip())
    if site_info.strip():
        parts.append(site_info.strip())
    if len(parts) <= 1:
        return parts[0] if parts else ""
    if site_info.strip():
        return " | ".join(parts[:-1]) + "\n" + parts[-1]
    return " | ".join(parts)

def write_comments_and_clear_title(path: Path, meta: Dict[str, object], site_info: str) -> str:
    """
    Build final Comments = [existing Comments][ | Title][\nSite Info],
    clear Title, and write back to EXIF for JPEG/TIFF.
    """
    final_comments = compose_final_comments(
        original_comments=str(meta.get("Comments","")),
        title=str(meta.get("Title","")),
        site_info=site_info or ""
    )
    if path.suffix.lower() in {'.jpg', '.jpeg', '.tif', '.tiff'}:
        try:
            ex = piexif.load(str(path))
            ex['0th'][XPTitle]   = _encode_xp("")                # clear title
            ex['0th'][XPComment] = _encode_xp(final_comments)    # set comments
            piexif.insert(piexif.dump(ex), str(path))
            _debug(f"Wrote Comments & cleared Title in {path.name}")
        except Exception as e:
            _warn(f"Failed to write comments/title for {path}: {e}")
    return final_comments

# ---------- filename tag extraction (position-agnostic) ----------
ANIMAL_ABBREV = {
    "GBH": "Great Blue Heron",
    "RTH": "Red-tailed Hawk",
    "WTDEER": "White-tailed Deer",
}
CANONICAL_TAGS: set = set()  # optional: fill with lowercase tags to whitelist
STOP_TOKENS = {
    "img", "image", "photo", "dsc", "dcim", "reconyx", "trailcam", "cam", "camera",
    "copy", "edited", "final"
}
DATE_PATTERNS = [
    re.compile(r"^\d{8}$"),
    re.compile(r"^\d{4}-\d{2}-\d{2}$"),
]
TIME_PATTERNS = [
    re.compile(r"^\d{4}$"),
    re.compile(r"^\d{6}$"),
    re.compile(r"^\d{2}-\d{2}-\d{2}$"),
]

def _split_tokens(name: str) -> List[str]:
    name = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", name)  # CamelCase -> spaced
    parts = re.split(r"[^A-Za-z]+", name)
    return [p for p in parts if p]

def _looks_like_date_or_time(tok: str) -> bool:
    for rx in DATE_PATTERNS:
        if rx.match(tok): return True
    for rx in TIME_PATTERNS:
        if rx.match(tok): return True
    return False

def _normalize_phrase(words: List[str]) -> str:
    return " ".join(w.strip() for w in words if w.strip())

def _contains_site_tokens(candidate_words: List[str], site_words: List[str]) -> bool:
    if not site_words or not candidate_words:
        return False
    cand = " ".join(candidate_words)
    site = " ".join(site_words)
    return cand == site or cand in site or site in cand

def infer_tags_from_filename(path: Path, site: str) -> List[str]:
    """
    Extract tags from ANYWHERE in the filename (position-agnostic).
    """
    stem = path.stem
    tokens = _split_tokens(stem)
    tokens_lower = [t.lower() for t in tokens]

    site_words = _split_tokens(site or "")
    site_words = [w.lower() for w in site_words]

    cleaned: List[str] = []
    for t, tl in zip(tokens, tokens_lower):
        if _looks_like_date_or_time(t) or tl.isdigit():
            continue
        if tl in STOP_TOKENS:
            continue
        if len(tl) < 3:
            continue
        cleaned.append(t)

    # Map abbreviations (single token)
    mapped_single: List[str] = []
    for t in cleaned:
        up = t.upper()
        if up in ANIMAL_ABBREV:
            mapped_single.append(ANIMAL_ABBREV[up])
        else:
            mapped_single.append(t)

    cleaned_lower = [t.lower() for t in mapped_single]

    candidates: List[str] = []

    # tri-grams
    for i in range(len(cleaned_lower) - 2):
        tri = cleaned_lower[i:i+3]
        if _contains_site_tokens(tri, site_words):
            continue
        phrase3 = _normalize_phrase([mapped_single[i], mapped_single[i+1], mapped_single[i+2]])
        if not CANONICAL_TAGS or phrase3.lower() in CANONICAL_TAGS:
            candidates.append(phrase3)

    # bi-grams
    for i in range(len(cleaned_lower) - 1):
        bi = cleaned_lower[i:i+2]
        if _contains_site_tokens(bi, site_words):
            continue
        phrase2 = _normalize_phrase([mapped_single[i], mapped_single[i+1]])
        if not CANONICAL_TAGS or phrase2.lower() in CANONICAL_TAGS:
            candidates.append(phrase2)

    # singles, avoid duplicating words used in phrases
    longer_words = set()
    for c in candidates:
        for w in c.lower().split():
            longer_words.add(w)

    for t in mapped_single:
        tl = t.lower()
        if tl in longer_words:
            continue
        if site_words and tl in site_words:
            continue
        if not CANONICAL_TAGS or tl in CANONICAL_TAGS:
            candidates.append(t)

    # dedupe preserving order
    seen = set()
    out: List[str] = []
    for c in candidates:
        key = c.lower()
        if key in seen: continue
        seen.add(key)
        out.append(c)
    return out

def write_keywords_exif(path: Path, tags: List[str]) -> None:
    if not tags or path.suffix.lower() not in {'.jpg', '.jpeg', '.tif', '.tiff'}:
        return
    try:
        ex = piexif.load(str(path))
        keywords = "; ".join(sorted({t.strip() for t in tags if t.strip()}))
        ex['0th'][XPKeywords] = _encode_xp(keywords)
        piexif.insert(piexif.dump(ex), str(path))
        _debug(f"Wrote XPKeywords to {path.name}: {keywords}")
    except Exception as e:
        _warn(f"Failed writing XPKeywords for {path}: {e}")

# ---------- main ----------
def collect_unique_tags(items: List[Dict[str, object]]) -> List[str]:
    uniq: Dict[str, str] = {}
    for it in items:
        for t in it.get("Tags", []):
            k = t.lower()
            if k not in uniq:
                uniq[k] = t
    return [uniq[k] for k in sorted(uniq.keys())]

def organize_images(src_folder: Path, recursive: bool, site: str, include_seconds: bool,
                    copyright_holder: str, site_info: str = ""):
    _info(f"Starting Image Organizer in folder: {src_folder} (recursive={recursive})")
    if not src_folder.exists():
        raise FileNotFoundError(f"Folder does not exist: {src_folder}")

    files = iter_images(src_folder, recursive)
    if not files:
        raise RuntimeError("No images found in the selected folder.")

    # read metadata pass
    item_infos = []
    for p in files:
        meta = read_image_metadata(p)
        dt, from_meta = choose_dt_for_filename(p, meta)
        if not from_meta:
            _warn(f"No timestamp in metadata for {p.name}; used file mtime.")
        # merge tags from filename (position-agnostic)
        fn_tags = infer_tags_from_filename(p, site)
        if fn_tags:
            existing_lower = {t.lower() for t in meta.get("Tags", [])}
            for t in fn_tags:
                if t.lower() not in existing_lower:
                    meta.setdefault("Tags", []).append(t)
                    existing_lower.add(t.lower())
        item_infos.append({"path": p, "meta": meta, "dt_for_name": dt})

    unique_tags = collect_unique_tags([i["meta"] for i in item_infos])

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_csv = src_folder / f"rename_log_{ts}.csv"
    base_fields = [
        "original_file_name", "new_file_name", "datetime_from_metadata",
        "converted_at", "Comments", "Title", "Subject", "Tags",
        "Copyright",
        "CopyrightExisting"
    ]
    header = base_fields + unique_tags

    used_names: Dict[str, int] = {}  # for collision management within this run
    rows_out: List[Dict[str, str]] = []
    errors: List[str] = []
    with open(log_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()

        for info in item_infos:
            p = info["path"]; meta = info["meta"]; dt = info["dt_for_name"]
            ext = p.suffix

            # If EXIF had seconds, force seconds in filename even if UI unchecked
            effective_seconds = include_seconds or bool(meta.get("HasSeconds", False))
            new_name = make_filename(dt, site, effective_seconds, ext)

            # If user unchecked seconds and collision arises, auto-upgrade to seconds
            if new_name in used_names and not effective_seconds:
                new_name = make_filename(dt, site, include_seconds, ext, force_seconds=True)

            # If still colliding (same second), add counter suffix
            ctr = used_names.get(new_name, 0) + 1
            if ctr > 1:
                new_name = make_filename(dt, site, include_seconds, ext, force_seconds=True, counter=ctr)
            used_names[new_name] = ctr

            try:
                new_path = safe_rename(p, new_name)

                # Comments <- existing + Title + Site Info; Title cleared
                final_comments = write_comments_and_clear_title(new_path, meta, site_info)

                # Copyright write (skip if exists)
                try:
                    write_copyright_exif(new_path, copyright_holder, skip_if_exists=True)
                except Exception as e:
                    _warn(f"EXIF Copyright write issue for {new_path}: {e}")

                # Write merged tags back to XPKeywords
                try:
                    write_keywords_exif(new_path, meta.get("Tags", []))
                except Exception as e:
                    _warn(f"Keyword write issue for {new_path}: {e}")

                converted_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                row = {
                    "original_file_name": p.name,
                    "new_file_name": new_path.name,
                    "datetime_from_metadata": meta.get("DateTimeFromMetadata", ""),
                    "converted_at": converted_at,
                    "Comments": final_comments,
                    "Title": "",  # cleared
                    "Subject": str(meta.get("Subject","")),
                    "Tags": "; ".join(meta.get("Tags", [])),
                    "Copyright": copyright_holder or "",
                    "CopyrightExisting": str(meta.get("CopyrightExisting",""))
                }
                tags_lower = {t.lower() for t in meta.get("Tags", [])}
                for t in unique_tags:
                    row[t] = 1 if t.lower() in tags_lower else 0

                writer.writerow(row)
                rows_out.append(row)
                _info(f"Renamed {p.name} -> {new_path.name}")
            except Exception as e:
                msg = f"Rename failed for {p}: {e}"
                _err(msg); errors.append(msg)

    _info(f"Wrote CSV log: {log_csv}")
    return log_csv, rows_out, unique_tags, errors

def example_filename(site: str, include_seconds: bool, example_dt: Optional[datetime] = None, ext: str = ".jpg") -> str:
    if example_dt is None: example_dt = datetime.now()
    # For example preview we just respect the checkbox (no seconds-forcing heuristic)
    return make_filename(example_dt, site, include_seconds, ext)
