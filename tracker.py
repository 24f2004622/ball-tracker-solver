import cv2
import numpy as np
import base64
from scipy.optimize import linear_sum_assignment

# Exact 10x6 bitmap templates for digits 0-9
TEMPLATES = {
    0: np.array([
        [1,1,1,1,1,1],
        [1,1,1,1,1,1],
        [1,1,0,0,1,1],
        [1,1,0,0,1,1],
        [1,1,0,0,1,1],
        [1,1,0,0,1,1],
        [1,1,0,0,1,1],
        [1,1,0,0,1,1],
        [1,1,1,1,1,1],
        [1,1,1,1,1,1]
    ], dtype=np.uint8),
    1: np.array([
        [0,0,1,1,0,0],
        [0,0,1,1,0,0],
        [1,1,1,1,0,0],
        [1,1,1,1,0,0],
        [0,0,1,1,0,0],
        [0,0,1,1,0,0],
        [0,0,1,1,0,0],
        [0,0,1,1,0,0],
        [1,1,1,1,1,1],
        [1,1,1,1,1,1]
    ], dtype=np.uint8),
    2: np.array([
        [1,1,1,1,1,1],
        [1,1,1,1,1,1],
        [0,0,0,0,1,1],
        [0,0,0,0,1,1],
        [1,1,1,1,1,1],
        [1,1,1,1,1,1],
        [1,1,0,0,0,0],
        [1,1,0,0,0,0],
        [1,1,1,1,1,1],
        [1,1,1,1,1,1]
    ], dtype=np.uint8),
    3: np.array([
        [1,1,1,1,1,1],
        [1,1,1,1,1,1],
        [0,0,0,0,1,1],
        [0,0,0,0,1,1],
        [1,1,1,1,1,1],
        [1,1,1,1,1,1],
        [0,0,0,0,1,1],
        [0,0,0,0,1,1],
        [1,1,1,1,1,1],
        [1,1,1,1,1,1]
    ], dtype=np.uint8),
    4: np.array([
        [1,1,0,0,1,1],
        [1,1,0,0,1,1],
        [1,1,0,0,1,1],
        [1,1,0,0,1,1],
        [1,1,1,1,1,1],
        [1,1,1,1,1,1],
        [0,0,0,0,1,1],
        [0,0,0,0,1,1],
        [0,0,0,0,1,1],
        [0,0,0,0,1,1]
    ], dtype=np.uint8),
    5: np.array([
        [1,1,1,1,1,1],
        [1,1,1,1,1,1],
        [1,1,0,0,0,0],
        [1,1,0,0,0,0],
        [1,1,1,1,1,1],
        [1,1,1,1,1,1],
        [0,0,0,0,1,1],
        [0,0,0,0,1,1],
        [1,1,1,1,1,1],
        [1,1,1,1,1,1]
    ], dtype=np.uint8),
    6: np.array([
        [1,1,1,1,1,1],
        [1,1,1,1,1,1],
        [1,1,0,0,0,0],
        [1,1,0,0,0,0],
        [1,1,1,1,1,1],
        [1,1,1,1,1,1],
        [1,1,0,0,1,1],
        [1,1,0,0,1,1],
        [1,1,1,1,1,1],
        [1,1,1,1,1,1]
    ], dtype=np.uint8),
    7: np.array([
        [1,1,1,1,1,1],
        [1,1,1,1,1,1],
        [0,0,0,0,1,1],
        [0,0,0,0,1,1],
        [0,0,0,0,1,1],
        [0,0,0,0,1,1],
        [0,0,0,0,1,1],
        [0,0,0,0,1,1],
        [0,0,0,0,1,1],
        [0,0,0,0,1,1]
    ], dtype=np.uint8),
    8: np.array([
        [1,1,1,1,1,1],
        [1,1,1,1,1,1],
        [1,1,0,0,1,1],
        [1,1,0,0,1,1],
        [1,1,1,1,1,1],
        [1,1,1,1,1,1],
        [1,1,0,0,1,1],
        [1,1,0,0,1,1],
        [1,1,1,1,1,1],
        [1,1,1,1,1,1]
    ], dtype=np.uint8),
    9: np.array([
        [1,1,1,1,1,1],
        [1,1,1,1,1,1],
        [1,1,0,0,1,1],
        [1,1,0,0,1,1],
        [1,1,1,1,1,1],
        [1,1,1,1,1,1],
        [0,0,0,0,1,1],
        [0,0,0,0,1,1],
        [1,1,1,1,1,1],
        [1,1,1,1,1,1]
    ], dtype=np.uint8),
}

def match_glyph(glyph):
    if glyph.shape == (10, 3):
        return 1
    best_d = -1
    best_score = -1
    for digit, tmpl in TEMPLATES.items():
        if glyph.shape == tmpl.shape:
            score = np.sum(glyph == tmpl)
            if score > best_score:
                best_score = score
                best_d = digit
    return best_d

def read_ball_number(gray_img, cx, cy):
    ix, iy = int(round(cx)), int(round(cy))
    patch = gray_img[max(0, iy-9):min(gray_img.shape[0], iy+10), max(0, ix-9):min(gray_img.shape[1], ix+10)]
    mask = (patch > 220).astype(np.uint8)
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    
    digits = []
    for l in range(1, n_labels):
        w = stats[l, cv2.CC_STAT_WIDTH]
        h = stats[l, cv2.CC_STAT_HEIGHT]
        x = stats[l, cv2.CC_STAT_LEFT]
        y = stats[l, cv2.CC_STAT_TOP]
        if h == 10 and (w == 6 or w == 3):
            glyph = mask[y:y+h, x:x+w]
            d = match_glyph(glyph)
            if d != -1:
                digits.append((x, d))
            
    digits.sort()
    if not digits:
        return None
    val = "".join(str(d[1]) for d in digits)
    return int(val)

def img_to_base64(img_bgr):
    _, buf = cv2.imencode('.png', img_bgr)
    return base64.b64encode(buf).decode('utf-8')

def track_and_solve(video_path, target_frames=[182, 300, 521]):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Could not open video file.")
        
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # 1. Detect red ball in frame 0
    ret, frame0 = cap.read()
    if not ret:
        raise ValueError("Could not read initial frame from video.")
        
    b, g, r = cv2.split(frame0)
    red_mask = ((r.astype(int) - g.astype(int) > 40) & (r.astype(int) - b.astype(int) > 40)).astype(np.uint8)
    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError("No starting red ball found in initial frame.")
        
    cnt = max(contours, key=cv2.contourArea)
    M = cv2.moments(cnt)
    if M['m00'] == 0:
        raise ValueError("Could not compute moments for red ball.")
    red_pos = np.array([M['m10'] / M['m00'], M['m01'] / M['m00']])
    
    # Crop start ball for visual verification
    s_x, s_y = int(round(red_pos[0])), int(round(red_pos[1]))
    start_crop = frame0[max(0, s_y-20):min(height, s_y+21), max(0, s_x-20):min(width, s_x+21)]
    start_b64 = img_to_base64(start_crop)
    
    # Detect all circles in frame 0
    gray0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
    c0 = cv2.HoughCircles(gray0, cv2.HOUGH_GRADIENT, dp=1, minDist=20, param1=50, param2=15, minRadius=10, maxRadius=16)
    if c0 is None:
        raise ValueError("No balls detected in frame 0.")
    pts0 = c0[0, :, :2]
    
    # Red ball index in the tracks array
    d_to_red = np.linalg.norm(pts0 - red_pos, axis=1)
    red_idx = np.argmin(d_to_red)
    
    current_tracks = pts0.copy()
    target_results = {}
    frame_crops = {}
    last_frame_bgr = None
    
    target_set = set(target_frames)
    
    for f in range(1, total_frames):
        ret, frame = cap.read()
        if not ret: break
        last_frame_bgr = frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        c = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1, minDist=20, param1=50, param2=15, minRadius=10, maxRadius=16)
        if c is not None:
            pts = c[0, :, :2]
            # Hungarian matching
            cost = np.linalg.norm(current_tracks[:, None, :] - pts[None, :, :], axis=2)
            row_ind, col_ind = linear_sum_assignment(cost)
            current_tracks = pts[col_ind]
        
        if f in target_set:
            p = current_tracks[red_idx]
            ix, iy = int(round(p[0])), int(round(p[1]))
            target_results[f] = (ix, iy)
            
            # Annotated crop with circle
            annotated = frame.copy()
            cv2.circle(annotated, (ix, iy), 14, (0, 255, 0), 2)
            crop_target = annotated[max(0, iy-25):min(height, iy+26), max(0, ix-25):min(width, ix+26)]
            frame_crops[str(f)] = img_to_base64(crop_target)
            
    cap.release()
    
    # Read number at final frame
    final_pos = current_tracks[red_idx]
    final_gray = cv2.cvtColor(last_frame_bgr, cv2.COLOR_BGR2GRAY)
    end_number = read_ball_number(final_gray, final_pos[0], final_pos[1])
    
    # Crop final ball
    f_x, f_y = int(round(final_pos[0])), int(round(final_pos[1]))
    final_crop = last_frame_bgr[max(0, f_y-20):min(height, f_y+21), max(0, f_x-20):min(width, f_x+21)]
    final_b64 = img_to_base64(final_crop)
    
    formatted_answer = f"{end_number}; " + "; ".join(f"{target_results[tf][0]},{target_results[tf][1]}" for tf in target_frames if tf in target_results)
    
    return {
        "status": "success",
        "formatted_answer": formatted_answer,
        "number": end_number,
        "video_info": {
            "total_frames": total_frames,
            "fps": fps,
            "resolution": f"{width}x{height}"
        },
        "target_frames": [
            {
                "frame": tf,
                "x": target_results.get(tf, (0,0))[0],
                "y": target_results.get(tf, (0,0))[1]
            }
            for tf in target_frames if tf in target_results
        ],
        "visualizations": {
            "start_ball": start_b64,
            "final_ball": final_b64,
            "frame_crops": frame_crops
        }
    }
