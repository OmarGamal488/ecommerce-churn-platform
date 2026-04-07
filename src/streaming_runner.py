"""Streaming simulator runner — generates synthetic events every 5 seconds."""

import time
import requests

session = requests.Session()

print("Streaming runner started. Generating events every 5 seconds...")

while True:
    try:
        session.post("http://fastapi:8000/streaming/generate")
    except Exception as e:
        print(f"Warning: {e}")
    time.sleep(5)
