#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# utils/ui_resource_utils.py
#
# Author: John Edward Stranzl, Jr.
# Affiliation(s): University of Nebraska-Lincoln, Blade Vision Systems, LLC
# Contact: jstranzl2@huskers.unl.edu, johnstranzl@gmail.com
# Created: Oct 21, 2025
# License: Apache License, Version 2.0, http://www.apache.org/licenses/LICENSE-2.0

import sys
from pathlib import Path

def _base_path() -> Path:
    """Return the base path depending on frozen vs. source run."""
    if hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS)
    return Path(__file__).resolve().parent.parent  # project root

def ui_path(relative_path: str) -> str:
    """
    Resolve a path to a .ui file inside dialogs/.
    Example: ui_path("file_utilities/QDialog_FileUtilities.ui")
    """
    return str(_base_path() / "dialogs" / relative_path)

def icon_path(relative_path: str, filename: str) -> str:
    """
    Resolve a path to an icon inside resources/app_icons or toolbar_icons.
    Example: icon_path("toolbar_icons/open.png")
    """
    return str(_base_path() / "resources" / relative_path / filename)

