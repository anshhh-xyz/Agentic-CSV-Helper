"""
run_api.py
-------------
Starts the Flask API that the frontend (see ../frontend) talks to.

Usage:
    python run_api.py

Then open frontend/index.html in a browser -- it's already configured to
call this API at http://localhost:5000.
"""

from dotenv import load_dotenv

load_dotenv()

from api.app import create_app  # noqa: E402  (must load .env first)

app = create_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
