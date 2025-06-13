# backend/app.py
from flask import Flask
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

from recommender import app as recommender_app

if __name__ == "__main__":
    recommender_app.run()