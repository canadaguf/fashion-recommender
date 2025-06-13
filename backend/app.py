# backend/app.py
from flask import Flask
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

from recommender import app as recommender_app

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # Render использует PORT
    recommender_app.run(host="0.0.0.0", port=port)