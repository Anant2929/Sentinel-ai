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

Set `GEMINI_API_KEY` in a `.env` file in the project directory before using
Gemini analysis. The app uses `gemini-3.5-flash-lite` for image input and shows
the sampled, annotated frames even if the Gemini request fails.

Use **Check Gemini model access** in the sidebar to see whether the exact model
is listed for the project belonging to the configured key. The key is not tied
to a Streamlit user. Google applies quotas per Google project, so making another
key in the same project does not reset its quota. Check the project and current
usage in Google AI Studio; never paste or share the key in logs or chat.
If the sidebar reports that the process environment supplies the key, that value
takes precedence over `.env`. Remove an outdated `GEMINI_API_KEY` from the
terminal/VS Code environment if you want to use the `.env` value instead.

Uploaded videos are sampled at five points across the full clip. The custom
YOLO model recognizes people only; the Gemini report is a visual aid and must
not be treated as a validated safety alarm.

Run the automated checks with:

```powershell
.\venv\Scripts\python.exe -m unittest test_frame_display.py test_video_frames.py test_gemini_diagnostics.py
```
