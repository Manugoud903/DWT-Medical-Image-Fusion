# DWT Medical Image Fusion

DWT-based CT and MRI medical image fusion using Python, Flask, and PyWavelets.

## Run locally

From the project root (the folder containing `app.py`), install the dependencies
and start the Flask app:

```powershell
python -m pip install -r requirements.txt
python app.py
```

Open http://127.0.0.1:5000.

## Deploy to Render

This repository includes a Render Blueprint in `render.yaml`. Push the project
to GitHub, then in Render choose **New > Blueprint**, connect the repository,
and apply the Blueprint. Render will install `requirements.txt` and start the
app with Gunicorn. The Blueprint uses Render's free web service plan; free
services may take a short time to wake after inactivity.
