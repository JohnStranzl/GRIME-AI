from PyQt5.QtWidgets import QWidget, QVBoxLayout
from PyQt5.QtWebEngineWidgets import QWebEngineView

class GoogleMapWidget(QWidget):
    """Encapsulates Leaflet map rendering inside a QWidget."""

    def __init__(self, camera_dict, parent=None):
        super().__init__(parent)
        self.view = QWebEngineView(self)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.view)
        self.setLayout(layout)

        html = self._build_html(camera_dict)
        self.view.setHtml(html)

    def _build_html(self, camera_dict):
        """Generate Leaflet HTML with markers from camera_dict."""
        markers_js = ""
        for name, cam in camera_dict.items():
            lat, lng = cam["lat"], cam["lng"]
            markers_js += (
                f"L.marker([{lat}, {lng}])"
                f".addTo(map).bindPopup('{name}: {lat}, {lng}');\n"
            )

        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8"/>
            <title>Camera Map</title>
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <link rel="stylesheet" href="https://unpkg.com/leaflet/dist/leaflet.css"/>
            <style>html, body, #map {{ height: 100%; margin: 0; padding: 0; }}</style>
        </head>
        <body>
            <div id="map"></div>
            <script src="https://unpkg.com/leaflet/dist/leaflet.js"></script>
            <script>
                var map = L.map('map').setView([40.8136, -96.7026], 5);
                L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
                    maxZoom: 19,
                    attribution: '&copy; OpenStreetMap contributors'
                }}).addTo(map);
                {markers_js}
            </script>
        </body>
        </html>
        """
