@echo off
call venv\Scripts\activate
pyinstaller --noconfirm --clean --windowed --name AIWorksheetCleaner ^
  --add-data "models;models" --add-data "config;config" ^
  --collect-all openvino --exclude-module PySide6.QtWebEngineCore --exclude-module PySide6.QtQml ^
  main.py
echo Output: dist\AIWorksheetCleaner\AIWorksheetCleaner.exe
