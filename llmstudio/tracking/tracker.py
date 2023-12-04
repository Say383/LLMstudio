import requests
import json
from dotenv import load_dotenv
import os

load_dotenv()
TRACKING_HOST = os.getenv("TRACKING_HOST", "localhost")
TRACKING_PORT = int(os.getenv("TRACKING_PORT", 8080))

# URL for the POST request
URL = f"http://{TRACKING_HOST}:{TRACKING_PORT}/api/tracking/logs/"

# Headers to be sent with the request
HEADERS = {
    "accept": "application/json",
    "Content-Type": "application/json"
}

class Tracker:
    def __init__(self, url):
        self._session = requests.Session()
        self._url = url
    
    def log(self, data: dict):
        req = self._session.post(self._url, 
                                 headers=HEADERS, 
                                 data=json.dumps(data),
                                 timeout=100)
        return req

tracker = Tracker(URL)
