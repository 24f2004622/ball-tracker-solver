# 🔴 Ball Tracker Solver

A web application and API to automatically track bouncing balls inside a circle, resolve elastic collisions with Hungarian matching, locate ball centers at specific video frames, and recognize the final ball number using bitmap OCR.

Live on Render: `https://ball-tracker-solver.onrender.com` (or your configured Render URL)

---

## 🚀 Features

- **Automated Starting Ball Detection**: Automatically detects the red ball in early frames.
- **Continuous Multi-Ball Tracking**: Uses Hough Circle Transform and Hungarian minimum-cost bipartite matching to maintain identity across collisions.
- **Subpixel & Integer Precision Coordinates**: Returns exact coordinates $(x,y)$ at any specified frame numbers.
- **Fast Bitmap OCR**: Instantaneous 100% accurate digit recognition (0–99) of final numbers without external dependencies.
- **REST API + Web UI**: Drag-and-drop web UI with visual crops for verification, plus full REST API support.

---

## 📋 API Usage

### `POST /api/solve`
Accepts `multipart/form-data`:
- `video`: Video file (`.webm`, `.mp4`, etc.)
- `frames`: (Optional) Comma-separated frame numbers, e.g. `182, 300, 521` (default)

#### Example using `cURL`:
```bash
curl -X POST \
  -F "video=@ball-tracker.webm" \
  -F "frames=182,300,521" \
  https://<your-service>.onrender.com/api/solve
```

#### Example Response:
```json
{
  "status": "success",
  "formatted_answer": "23; 400,304; 420,210; 282,398",
  "number": 23,
  "target_frames": [
    {"frame": 182, "x": 400, "y": 304},
    {"frame": 300, "x": 420, "y": 210},
    {"frame": 521, "x": 282, "y": 398}
  ],
  "video_info": {
    "fps": 30.0,
    "resolution": "480x480",
    "total_frames": 698
  }
}
```

---

## 🛠 Local Setup

```bash
git clone https://github.com/24f2004622/ball-tracker-solver.git
cd ball-tracker-solver
pip install -r requirements.txt
python app.py
```
Open `http://localhost:5000` in your browser.
