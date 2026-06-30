# -*- mode: python ; coding: utf-8 -*-
import os

import mediapipe as mp
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

hiddenimports = []
hiddenimports += collect_submodules('eyetrax')
hiddenimports += collect_submodules('cv2')
datas = []
datas += collect_data_files('mediapipe')
mediapipe_dir = os.path.dirname(mp.__file__)
datas += [(os.path.join(mediapipe_dir, "modules"), "mediapipe\\modules")]


a = Analysis(
    ['src\\eyetrax\\app\\gaze_typing_suite.py'],
    pathex=['.\\src'],
    binaries=[],
    datas=[('.\\src\\eyetrax\\models', 'eyetrax\\models')] + datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='eyetrax_gaze',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='eyetrax_gaze',
)
