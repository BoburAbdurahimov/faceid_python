@echo off
echo Creating Python Virtual Environment...
python -m venv .venv
call .venv\Scripts\activate.bat

echo Installing dependencies...
pip install -r requirements.txt

echo Starting server...
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
