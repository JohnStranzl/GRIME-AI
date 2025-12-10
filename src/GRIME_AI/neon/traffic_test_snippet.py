import requests

# Replace with your assigned credentials
SYSTEM_ID = "your_system_id"
API_KEY = "your_api_key"

# Base URL for TrafficLand API
BASE_URL = "https://api.trafficland.com/v2"

# Example: Get all video feed metadata
def get_video_feeds():
    url = f"{BASE_URL}/video_feeds"
    params = {
        "system": SYSTEM_ID,
        "key": API_KEY
    }
    headers = {
        "Accept-Encoding": "gzip"
    }
    response = requests.get(url, params=params, headers=headers)
    response.raise_for_status()
    return response.json()

if __name__ == "__main__":
    try:
        feeds = get_video_feeds()
        # Print out some metadata for each camera
        for feed in feeds.get("video_feeds", []):
            print("Camera ID:", feed.get("publicId"))
            print("Name:", feed.get("name"))
            print("Location:", feed.get("location"))
            print("City:", feed.get("city"))
            print("State:", feed.get("state"))
            print("Country:", feed.get("country"))
            print("Image URL:", feed.get("imageUrl"))
            print("-" * 40)
    except Exception as e:
        print("Error fetching camera metadata:", e)
