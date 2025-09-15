import sys

# ---------- logging helpers ----------
def debug(msg: str): print(f"[DEBUG] {msg}")
def info(msg: str): print(f"[INFO]  {msg}")
def warn(msg: str): print(f"[WARN]  {msg}")
def err(msg: str): print(f"[ERROR] {msg}", file=sys.stderr)