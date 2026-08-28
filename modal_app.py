"""
Deploys the Streamlit dashboard (app.py) on Modal.

Setup (one time):
    modal secret create supabase-credentials DATABASE_URL="postgresql://..."

Deploy:
    modal deploy modal_app.py
"""

import shlex
import subprocess
from pathlib import Path

import modal

app_script_local_path = Path(__file__).parent / "app.py"
app_script_remote_path = "/root/app.py"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "streamlit",
        "pandas",
        "numpy",
        "scikit-learn",
        "plotly",
        "sqlalchemy",
        "psycopg2-binary",
    )
    .add_local_file(app_script_local_path, app_script_remote_path)
)

app = modal.App(name="diabetes-risk-dashboard", image=image)


@app.function(secrets=[modal.Secret.from_name("supabase-credentials")])
@modal.concurrent(max_inputs=100)
@modal.web_server(8000, startup_timeout=60)
def run():
    target = shlex.quote(app_script_remote_path)
    cmd = (
        f"streamlit run {target} "
        "--server.port 8000 --server.address 0.0.0.0 "
        "--server.enableCORS=false --server.enableXsrfProtection=false"
    )
    subprocess.Popen(cmd, shell=True)
