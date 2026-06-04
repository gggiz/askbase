# -*- mode: python ; coding: utf-8 -*-


from PyInstaller.utils.hooks import collect_all

# 把容易缺文件的包全量收集
_pkgs = ['safehttpx', 'gradio', 'chromadb', 'groovy', 'huggingface_hub',
          'sentence_transformers', 'transformers', 'tokenizers', 'pydantic',
          'starlette', 'fastapi', 'certifi', 'httpx']
_datas = []; _binaries = []; _hidden = []
for p in _pkgs:
    d, b, h = collect_all(p)
    _datas += d; _binaries += b; _hidden += h

a = Analysis(
    ['app.py'],
    pathex=[],
    binaries=_binaries,
    datas=_datas,
    hiddenimports=_hidden + ['pydantic_settings', 'websockets', 'gradio_client'],
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
    name='AskBase',
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
    name='AskBase',
)
