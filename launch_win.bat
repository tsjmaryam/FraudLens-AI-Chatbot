@echo off
setlocal
cd /d "%~dp0"

set VENV_DIR=test_env

if not exist "%VENV_DIR%\Scripts\python.exe" (
  echo [ERROR] test_env 不存在或不完整，请先在此目录手工执行:
  echo     python -m pip install --upgrade pip setuptools wheel virtualenv
  echo     python -m virtualenv test_env
  pause & exit /b 1
)

echo Activating venv: %VENV_DIR%
call "%VENV_DIR%\Scripts\activate.bat" || (echo [ERROR] 激活失败 & pause & exit /b 1)

python -m pip install --upgrade pip
pip install -r requirements.txt || (echo [ERROR] 依赖安装失败 & pause & exit /b 1)

set STREAMLIT_BROWSER_GATHERUSAGESTATS=false
python -m streamlit run app_chat.py --server.port=8501 --server.headless=false

pause
endlocal
