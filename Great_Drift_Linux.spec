# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['GUI.py'],  # Assurez-vous que le fichier principal est correct
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=[
        'matplotlib.backends.backend_pdf',
        'matplotlib.backends.backend_agg',
    ],
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=True
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='The_Great_Drift',  # Supprimez l'extension .exe pour Linux
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    icon=None  # Supprimez l'icône Windows
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='The_Great_Drift'
)
