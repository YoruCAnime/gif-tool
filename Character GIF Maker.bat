@echo off
rem Uses the Python of TagGUI Plus (GPU onnxruntime + CLIP). Falls back to the normal Python (slower, colours only).
set "TAG=%USERPROFILE%\Desktop\TagGUI Plus\venv\Scripts\pythonw.exe"
set "SYS=%LOCALAPPDATA%\Programs\Python\Python311\pythonw.exe"
if exist "%TAG%" (
    start "" "%TAG%" -X utf8 "%~dp0app.py"
) else (
    start "" "%SYS%" -X utf8 "%~dp0app.py"
)
