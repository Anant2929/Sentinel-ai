# Sentinel AI

## Run on Windows

Run the app with the included launcher. It creates the virtual environment if
needed, checks/installs dependencies, and starts Streamlit with that environment:

```powershell
.\run_app.bat
```

Or run the setup and start commands manually:

```powershell
py -3.11 -m venv venv
.\venv\Scripts\python.exe -m pip install --upgrade pip
.\venv\Scripts\python.exe -m pip install -r requirements.txt
.\venv\Scripts\python.exe -m streamlit run app.py
```

Using `python -m streamlit` ensures Streamlit and the app use the same Python
environment. If the `venv` directory already exists, skip the first command.

Set `GEMINI_API_KEY` in a `.env` file in the project directory before using the
Gemini 2.5 Flash analysis. Install dependencies again after pulling updates.
The app also needs camera access.
