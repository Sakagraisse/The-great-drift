# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(['GUI.py'],
             pathex=[],
             binaries=[],
             datas=[],
             hiddenimports=[],
             hookspath=[],
             runtime_hooks=[],
             excludes=['tkinter'],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=True)

# Modify the spec file to create a one-file executable
# Set the exe name to The_Great_Drift.exe and add the --onefile option
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher,
             )

exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,
          [],
          name='The Great Drift',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          upx_exclude=[],
          runtime_tmpdir=None,
          console=False,
          icon='icon.ico'
          )


