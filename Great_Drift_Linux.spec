# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(['GUI.py'],
             pathex=[],
             binaries=[],
             datas=[('icon.ico', '.')],
             hiddenimports=[
            'matplotlib.backends.backend_pdf',
            'matplotlib.backends.backend_agg',
             ],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             cipher=block_cipher,
             noarchive=True)

pyz = PYZ(a.pure, a.zipped_data,
          cipher=block_cipher)

exe = EXE(pyz,
          a.scripts,
          exclude_binaries=True,
          name='The_Great_Drift',
          debug=False,
          strip=False,
          upx=True,
          runtime_tmpdir=None,
          console=False,
          icon='icon.ico')

coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=True,
               upx_exclude=[],
               name='The_Great_Drift')

