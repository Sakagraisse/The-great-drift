# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(['GUI.py'],
             pathex=[],
             binaries=[],
             datas=[('frame_a.npy', '.'), ('frame_x.npy', '.'), ('frame_d.npy', '.'),('frame_surplus.npy', '.'),('frame_t.npy', '.'),('frame_u.npy', '.'),('frame_v.npy', '.'),('frame_fitnessToT.npy', '.'),(last_simulation_parameters.npy', '.')],
             hiddenimports=[],
             hookspath=[],
             runtime_hooks=[],
             excludes=['tkinter'],
             cipher=block_cipher,
             noarchive=True)

pyz = PYZ(a.pure, a.zipped_data,
          cipher=block_cipher)

exe = EXE(pyz,
          a.scripts,
          exclude_binaries=True,
          name='The Great Drift',
          debug=False,
          strip=False,
          upx=False,
          runtime_tmpdir=None,
          console=False,
          icon='icon.ico')

app = BUNDLE(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
                name='The Great Drift.app',
                icon='icon.ico',
                bundle_identifier=None)
