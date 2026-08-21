"""
Path resolution utilities for Attendance System.

Handles resolving paths whether running from source (.py) or compiled
via PyInstaller (.exe, onedir mode).

Key concepts:
- APP_DIR: The directory where the application executable or main script lives.
  * When frozen (.exe): sys.executable's parent directory (e.g. dist/AttendanceSystem/).
    All runtime files (saved faces, databases, logs, exports, user config.yaml) are stored here.
  * When script (.py): project root directory.
- BUNDLE_DIR: The directory where PyInstaller extracts bundled read-only files (sys._MEIPASS).
  * When frozen (.exe): sys._MEIPASS (e.g. dist/AttendanceSystem/_internal/).
  * When script (.py): same as APP_DIR.
"""

import sys
import os
import shutil
from pathlib import Path


def get_app_dir() -> Path:
    """
    Returns the directory where the application executable or script resides.
    For frozen PyInstaller EXEs, this is Path(sys.executable).parent (the folder containing the .exe).
    For normal Python execution, this is the root project directory.
    """
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


def get_bundle_dir() -> Path:
    """
    Returns the directory containing bundled read-only resources.
    For frozen PyInstaller EXEs, this is Path(sys._MEIPASS).
    For normal Python execution, this is the root project directory.
    """
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS)
    return get_app_dir()


def resolve_path(path_input) -> Path:
    """
    Resolves a relative or absolute path against APP_DIR / BUNDLE_DIR.
    
    Rules:
    1. If path_input is None or empty, returns APP_DIR.
    2. If path_input is an absolute path:
       - If it exists on disk, returns Path(path_input).
       - If it doesn't exist (e.g. moved folder), tries resolving its name
         under APP_DIR or BUNDLE_DIR.
    3. If path_input is a relative path:
       - If it exists under APP_DIR, returns APP_DIR / path_input.
       - Else if it exists under BUNDLE_DIR, returns BUNDLE_DIR / path_input.
       - Otherwise defaults to APP_DIR / path_input (so newly created files/dirs go to APP_DIR beside the .exe).
    """
    if not path_input:
        return get_app_dir()

    p = Path(path_input)
    app_dir = get_app_dir()
    bundle_dir = get_bundle_dir()

    if p.is_absolute():
        if p.exists():
            return p
        app_sub = app_dir / p.name
        if app_sub.exists():
            return app_sub
        bundle_sub = bundle_dir / p.name
        if bundle_sub.exists():
            return bundle_sub
        return app_dir / p.name

    app_target = app_dir / p
    if app_target.exists():
        return app_target

    bundle_target = bundle_dir / p
    if bundle_target.exists():
        return bundle_target

    return app_target


def ensure_config_exists(config_name: str = "config.yaml") -> Path:
    """
    Ensures config.yaml exists in APP_DIR (beside .exe). If not present in APP_DIR
    but exists in BUNDLE_DIR, copies it over to APP_DIR.
    """
    app_config = get_app_dir() / config_name
    if not app_config.exists():
        bundle_config = get_bundle_dir() / config_name
        if bundle_config.exists():
            try:
                shutil.copy2(bundle_config, app_config)
            except Exception:
                pass
    return app_config if app_config.exists() else resolve_path(config_name)
