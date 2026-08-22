# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec for the Attendance System.
Build with:  .venv\Scripts\pyinstaller attendance.spec --noconfirm
"""
import sys
from pathlib import Path
from PyInstaller.utils.hooks import (
    collect_all,
    collect_data_files,
    collect_dynamic_libs,
    collect_submodules,
)

ROOT = Path(SPECPATH)

# مسار نماذج InsightFace على جهازك (مجلد المستخدم)
INSIGHTFACE_MODELS = Path.home() / ".insightface" / "models"

block_cipher = None

hidden_imports = [
    "PyQt6", "PyQt6.QtWidgets", "PyQt6.QtCore", "PyQt6.QtGui",
    "cv2", "numpy", "ultralytics", "ultralytics.nn.tasks",
    "ultralytics.utils", "ultralytics.utils.torch_utils",
    "insightface", "insightface.app", "insightface.model_zoo",
    "onnxruntime", "onnxruntime.capi", "faiss",
    "scipy", "scipy._cyutility", "albumentations",
    "pytesseract", "easyocr", "torch", "torchvision", "PIL",
    "sqlite3", "openpyxl", "pandas", "yaml",
    "matplotlib", "matplotlib.backends.backend_agg",
]
hidden_imports += collect_submodules("insightface")
hidden_imports += collect_submodules("onnxruntime")
hidden_imports += collect_submodules("easyocr")
hidden_imports += collect_submodules("scipy")
hidden_imports += collect_submodules("albumentations")
hidden_imports += collect_submodules("matplotlib")

scipy_datas, scipy_binaries, scipy_hidden = collect_all("scipy")
alb_datas, alb_binaries, alb_hidden = collect_all("albumentations")
mpl_datas, mpl_binaries, mpl_hidden = collect_all("matplotlib")
hidden_imports += scipy_hidden + alb_hidden + mpl_hidden

# تجهيز مسارات البيانات الإضافية
additional_datas = [
    (str(ROOT / "config.yaml"),         "."),
    (str(ROOT / "models"),              "models"),
    (str(ROOT / "gui"),                 "gui"),
    (str(ROOT / "src"),                 "src"),
    (str(ROOT / "assets"),              "assets"),
    
    # 1. إضافة قاعدة بيانات الوجوه الخاصة بالمشروع
    (str(ROOT / "enrollment_faces"),             "enrollment_faces"),
]

# 2. إضافة نماذج InsightFace من مجلد النظام المحلي إلى داخل الـ EXE
if INSIGHTFACE_MODELS.exists():
    additional_datas.append((str(INSIGHTFACE_MODELS), ".insightface/models"))

a = Analysis(
    [str(ROOT / "app.py")],
    pathex=[str(ROOT)],
    datas=additional_datas + collect_data_files("easyocr") + collect_data_files("insightface")  + collect_data_files("ultralytics")
          + scipy_datas + alb_datas + mpl_datas,
    binaries=collect_dynamic_libs("onnxruntime") + collect_dynamic_libs("torch")
            + scipy_binaries + alb_binaries + mpl_binaries,
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["tkinter", "IPython"],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="AttendanceSystem",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,            # retain a console so client startup errors are visible
    icon=str(ROOT / "assets" / "icon.ico") if (ROOT / "assets" / "icon.ico").exists() else None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="AttendanceSystem",
)