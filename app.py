import os
import tempfile
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from tracker import track_and_solve

app = Flask(__name__)
CORS(app)

# Max upload size: 100MB
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/health')
def health():
    return jsonify({"status": "healthy", "service": "ball-tracker-solver"})

@app.route('/api/solve', methods=['POST'])
def solve():
    if 'video' not in request.files:
        return jsonify({"status": "error", "message": "No video file provided"}), 400
        
    file = request.files['video']
    if file.filename == '':
        return jsonify({"status": "error", "message": "Empty file name"}), 400
        
    # Target frames
    frames_str = request.form.get('frames', '182, 300, 521')
    try:
        target_frames = [int(f.strip()) for f in frames_str.split(',') if f.strip().isdigit()]
        if not target_frames:
            target_frames = [182, 300, 521]
    except Exception:
        target_frames = [182, 300, 521]
        
    # Save to temp file
    suffix = os.path.splitext(file.filename)[1]
    if not suffix:
        suffix = '.webm'
        
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp_path = tmp.name
        file.save(tmp_path)
        
    try:
        result = track_and_solve(tmp_path, target_frames)
        return jsonify(result)
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
