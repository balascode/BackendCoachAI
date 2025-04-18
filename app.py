import os
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import cv2
import mediapipe as mp
import numpy as np
from werkzeug.utils import secure_filename
import uuid
import logging
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}, r"/static/*": {"origins": "*"}})

# Configuration
UPLOAD_FOLDER = 'static/uploads'
COMPARISON_FOLDER = 'static/comparisons'
PLOT_FOLDER = 'static/plots'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['COMPARISON_FOLDER'] = COMPARISON_FOLDER
app.config['PLOT_FOLDER'] = PLOT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB limit

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(COMPARISON_FOLDER, exist_ok=True)
os.makedirs(PLOT_FOLDER, exist_ok=True)

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
JOINT_NAMES = [lm.name for lm in mp_pose.PoseLandmark]

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_keypoints(video_path):
    logger.debug(f"Extracting keypoints from video: {video_path}")
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    cap = cv2.VideoCapture(video_path)
    keypoints = []
    
    if not cap.isOpened():
        logger.error(f"Failed to open video: {video_path}")
        return []
    
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)
        if results.pose_landmarks:
            frame_keypoints = [(lm.x, lm.y, lm.z) for lm in results.pose_landmarks.landmark]
            logger.debug(f"Frame {frame_count}: Detected {len(frame_keypoints)} keypoints")
        else:
            frame_keypoints = None
            logger.debug(f"Frame {frame_count}: No keypoints detected")
        keypoints.append(frame_keypoints)
        frame_count += 1
    
    cap.release()
    pose.close()
    logger.info(f"Extracted keypoints from {frame_count} frames in {video_path}")
    return keypoints

def calculate_difference(kp1, kp2):
    if not kp1 or not kp2:
        return None
    diffs = []
    for a, b in zip(kp1, kp2):
        if a and b:
            diffs.append(np.linalg.norm(np.array(a) - np.array(b)))
        else:
            diffs.append(None)
    return diffs

def compare_videos(teacher_kps, student_kps):
    logger.debug("Comparing videos")
    frame_differences = []
    min_length = min(len(teacher_kps), len(student_kps))
    
    for i in range(min_length):
        t_kp = teacher_kps[i]
        s_kp = student_kps[i]
        if t_kp and s_kp:
            frame_diff = calculate_difference(t_kp, s_kp)
            if frame_diff:
                frame_differences.append(frame_diff)
    
    logger.info(f"Compared {len(frame_differences)} frames")
    return frame_differences

def compute_accuracy_score(differences, threshold=0.1):
    scores = []
    for frame_diff in differences:
        valid_diffs = [d for d in frame_diff if d is not None]
        if not valid_diffs:
            scores.append(0)
            continue
        match_count = sum(1 for d in valid_diffs if d <= threshold)
        scores.append((match_count / len(valid_diffs)) * 100)
    return scores

def generate_heuristics(avg_diffs, differences, sport="general", joint_threshold=0.1, teacher_kps=None, student_kps=None):
    tips = []
    region_scores = {
        "upper_body": 0.0,
        "lower_body": 0.0,
        "core": 0.0
    }
    
    # Define body regions
    upper_body_joints = [
        mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER,
        mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.RIGHT_ELBOW,
        mp_pose.PoseLandmark.LEFT_WRIST, mp_pose.PoseLandmark.RIGHT_WRIST
    ]
    lower_body_joints = [
        mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP,
        mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.RIGHT_KNEE,
        mp_pose.PoseLandmark.LEFT_ANKLE, mp_pose.PoseLandmark.RIGHT_ANKLE
    ]
    core_joints = [
        mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP,
        mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER
    ]
    
    # Calculate region scores
    for region, joints in [("upper_body", upper_body_joints), ("lower_body", lower_body_joints), ("core", core_joints)]:
        valid_diffs = [avg_diffs[i] for i in joints if i < len(avg_diffs) and avg_diffs[i] is not None and not np.isnan(avg_diffs[i])]
        region_scores[region] = np.mean(valid_diffs) if valid_diffs else 0.0
    
    # Generate sport-specific tips
    if sport == "fencing":
        if region_scores["upper_body"] > joint_threshold:
            tips.append("Focus on arm alignment; keep your sword arm steady and aligned with your shoulder.")
        if region_scores["lower_body"] > joint_threshold:
            tips.append("Maintain a lower stance; bend your knees more to improve balance.")
    elif sport == "skating":
        if region_scores["lower_body"] > joint_threshold:
            tips.append("Work on knee bend and ankle flexibility for better glide control.")
        if region_scores["core"] > joint_threshold:
            tips.append("Engage your core to maintain balance during spins.")
    else:  # General tips
        if avg_diffs[mp_pose.PoseLandmark.RIGHT_WRIST.value] > joint_threshold:
            tips.append("Keep your right hand steadier and aligned with the coach's position.")
        if avg_diffs[mp_pose.PoseLandmark.LEFT_KNEE.value] > joint_threshold:
            tips.append("Adjust your left knee position to match the coach's form.")
        if avg_diffs[mp_pose.PoseLandmark.RIGHT_ANKLE.value] > joint_threshold:
            tips.append("Focus on right ankle alignment for better stability.")
    
    if not tips:
        tips.append("Your form is generally good! Focus on maintaining consistency with the coach's movements.")
    
    return tips, region_scores

def summarize_differences(differences, teacher_kps=None, student_kps=None, sport="general", threshold=0.1):
    if not differences:
        logger.error("No valid frame comparisons found.")
        return [], ["No movement data available for analysis. Check video quality or pose detection."]
    
    avg_diffs = np.nanmean(differences, axis=0)
    if not isinstance(avg_diffs, np.ndarray):
        logger.error("Average differences is not an array.")
        return [], ["Pose data may be incomplete."]
    
    tips, region_scores = generate_heuristics(avg_diffs, differences, sport=sport, joint_threshold=threshold,
                                             teacher_kps=teacher_kps, student_kps=student_kps)
    
    valid_diffs = avg_diffs[~np.isnan(avg_diffs)]
    mean_diff = np.mean(valid_diffs)
    accuracy_score = max(0, 100 - (mean_diff * 100))
    
    return avg_diffs, tips, region_scores, accuracy_score

import ffmpeg

def create_comparison_video(coach_path, player_path, output_normal_path, output_dynamic_path, threshold=0.1):
    logger.debug(f"Creating comparison videos: normal={output_normal_path}, dynamic={output_dynamic_path}")
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    cap1 = cv2.VideoCapture(coach_path)
    cap2 = cv2.VideoCapture(player_path)
    
    if not cap1.isOpened():
        logger.error(f"Failed to open coach video: {coach_path}")
        raise Exception(f"Failed to open coach video: {coach_path}")
    if not cap2.isOpened():
        logger.error(f"Failed to open player video: {player_path}")
        raise Exception(f"Failed to open player video: {player_path}")
    
    # Get raw video dimensions from OpenCV
    width1 = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height1 = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width2 = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
    height2 = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap1.get(cv2.CAP_PROP_FPS)
    
    # Detect rotation metadata using ffmpeg
    def get_rotation(video_path):
        try:
            probe = ffmpeg.probe(video_path)
            rotation = None
            for stream in probe['streams']:
                if 'tags' in stream and 'rotate' in stream['tags']:
                    rotation = int(stream['tags']['rotate'])
                    break
                if 'side_data_list' in stream:
                    for side_data in stream['side_data_list']:
                        if side_data['side_data_type'] == 'Rotation':
                            rotation = side_data['rotation']
                            break
            logger.debug(f"Detected rotation for {video_path}: {rotation} degrees")
            return rotation if rotation else 0
        except Exception as e:
            logger.warning(f"Failed to detect rotation for {video_path}: {str(e)}")
            return 0

    rotation1 = get_rotation(coach_path)
    rotation2 = get_rotation(player_path)
    
    # Determine intended orientation after accounting for rotation
    is_vertical1 = (height1 > width1 and rotation1 in [0, 180]) or (width1 > height1 and rotation1 in [90, 270])
    is_vertical2 = (height2 > width2 and rotation2 in [0, 180]) or (width2 > height2 and rotation2 in [90, 270])
    is_vertical = is_vertical1 and is_vertical2  # Both videos should be treated similarly
    
    logger.debug(f"Coach video: raw {width1}x{height1}, rotated vertical={is_vertical1}, rotation={rotation1}")
    logger.debug(f"Player video: raw {width2}x{height2}, rotated vertical={is_vertical2}, rotation={rotation2}")
    
    # Use the larger dimensions to maintain aspect ratio
    input_width = max(width1, width2)
    input_height = max(height1, height2)
    
    # Adjust for rotation if needed (rotate frames during processing)
    def rotate_frame(frame, rotation):
        if rotation == 90:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif rotation == 180:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        elif rotation == 270:
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return frame

    # Calculate output dimensions based on intended orientation
    if is_vertical:
        # Stack vertically: width remains the same, height is doubled
        output_width = input_width
        output_height = input_height * 2
    else:
        # Stack horizontally: height remains the same, width is doubled
        output_width = input_width * 2
        output_height = input_height
    
    logger.debug(f"Output dimensions: width={output_width}, height={output_height}, vertical={is_vertical}")
    
    # Initialize video writers with H264 codec
    fourcc = cv2.VideoWriter_fourcc(*'H264')
    out_normal = cv2.VideoWriter(output_normal_path, fourcc, fps, (output_width, output_height))
    out_dynamic = cv2.VideoWriter(output_dynamic_path, fourcc, fps, (output_width, output_height))
    
    if not out_normal.isOpened() or not out_dynamic.isOpened():
        logger.error(f"Failed to create video writer: normal={output_normal_path}, dynamic={output_dynamic_path}")
        raise Exception("Failed to create video writer")
    
    frame_count = 0
    window_size = 20
    window_diffs = []
    window_frames = []
    
    while cap1.isOpened() and cap2.isOpened():
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        
        if not ret1 or not ret2 or frame1 is None or frame2 is None:
            logger.debug(f"End of video reached at frame {frame_count}")
            break
        
        # Rotate frames based on metadata
        frame1 = rotate_frame(frame1, rotation1)
        frame2 = rotate_frame(frame2, rotation2)
        
        # Get dimensions after rotation
        h1, w1 = frame1.shape[:2]
        h2, w2 = frame2.shape[:2]
        
        # Resize frames to match the larger dimensions while preserving aspect ratio
        frame1 = cv2.resize(frame1, (input_width, input_height), interpolation=cv2.INTER_AREA)
        frame2 = cv2.resize(frame2, (input_width, input_height), interpolation=cv2.INTER_AREA)
        
        img1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
        res1 = pose.process(img1)
        res2 = pose.process(img2)
        
        diffs = None
        if res1.pose_landmarks and res2.pose_landmarks:
            kp1 = [(lm.x, lm.y, lm.z) for lm in res1.pose_landmarks.landmark]
            kp2 = [(lm.x, lm.y, lm.z) for lm in res2.pose_landmarks.landmark]
            diffs = calculate_difference(kp1, kp2)
        else:
            logger.warning(f"No pose landmarks detected in frame {frame_count}")
        
        window_diffs.append(diffs)
        window_frames.append((frame1, frame2, res1, res2))
        frame_count += 1
        
        # Normal video processing
        frame_normal = frame2.copy()
        if res1.pose_landmarks and res2.pose_landmarks:
            teacher_style = mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=1, circle_radius=2)
            mp_drawing.draw_landmarks(frame_normal, res1.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      landmark_drawing_spec=teacher_style)
            
            valid_diffs = [(i, d) for i, d in enumerate(diffs) if d is not None]
            valid_diffs.sort(key=lambda x: x[1], reverse=True)
            top_errors = valid_diffs[:1]
            
            for i, d in enumerate(diffs):
                h, w = frame_normal.shape[:2]
                p2 = res2.pose_landmarks.landmark[i]
                x, y = int(p2.x * w), int(p2.y * h)
                if (i, d) in top_errors and d is not None:
                    if d > threshold * 1.5:
                        color = (0, 0, 255)
                        label = f"{JOINT_NAMES[i]}: High"
                    elif d > threshold:
                        color = (0, 255, 255)
                        label = f"{JOINT_NAMES[i]}: Adjust"
                    else:
                        color = (0, 255, 0)
                        label = None
                    cv2.circle(frame_normal, (x, y), 5, color, -1)
                    if label:
                        cv2.putText(frame_normal, label, (x + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                else:
                    cv2.circle(frame_normal, (x, y), 5, (0, 255, 0), -1)
            
            overlay = frame_normal.copy()
            for i, d in top_errors:
                if d > threshold:
                    x, y = int(res2.pose_landmarks.landmark[i].x * w), int(res2.pose_landmarks.landmark[i].y * h)
                    intensity = min(255, int(d * 1000))
                    cv2.circle(overlay, (x, y), 20, (0, 0, intensity), -1)
            alpha = 0.3
            frame_normal = cv2.addWeighted(overlay, alpha, frame_normal, 1 - alpha, 0)
            
            mp_drawing.draw_landmarks(frame_normal, res2.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        # Combine frames based on intended orientation
        if is_vertical:
            combined_normal = cv2.vconcat([frame1, frame_normal])
        else:
            combined_normal = cv2.hconcat([frame1, frame_normal])
        
        if res1.pose_landmarks and res2.pose_landmarks:
            valid_diffs = [d for d in diffs if d is not None]
            avg_diff = np.mean(valid_diffs) if valid_diffs else 0
            if avg_diff > 1.5 * threshold:
                cv2.putText(combined_normal, "High Error Detected", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                for _ in range(3):
                    out_normal.write(combined_normal)
            else:
                out_normal.write(combined_normal)
        else:
            out_normal.write(combined_normal)
        
        # Dynamic video processing
        if len(window_frames) == window_size or (not ret1 or not ret2):
            max_diffs = [None] * len(JOINT_NAMES)
            if any(d is not None for d in window_diffs):
                for i in range(len(JOINT_NAMES)):
                    valid_diffs = [d[i] for d in window_diffs if d is not None and d[i] is not None]
                    max_diffs[i] = max(valid_diffs) if valid_diffs else None
            
            valid_diffs = [(i, d) for i, d in enumerate(max_diffs) if d is not None]
            valid_diffs.sort(key=lambda x: x[1], reverse=True)
            top_errors = valid_diffs[:1]
            avg_diff = np.mean([d for _, d in top_errors]) if top_errors else 0
            
            mid_idx = min(len(window_frames) // 2, len(window_frames) - 1)
            frame1, frame2, res1, res2 = window_frames[mid_idx]
            frame1 = rotate_frame(frame1, rotation1)
            frame2 = rotate_frame(frame2, rotation2)
            frame1 = cv2.resize(frame1, (input_width, input_height), interpolation=cv2.INTER_AREA)
            frame2 = cv2.resize(frame2, (input_width, input_height), interpolation=cv2.INTER_AREA)
            frame_dynamic = frame2.copy()
            
            if res1.pose_landmarks:
                teacher_style = mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=1, circle_radius=2)
                mp_drawing.draw_landmarks(frame_dynamic, res1.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                          landmark_drawing_spec=teacher_style)
            
            if res2.pose_landmarks:
                overlay = frame_dynamic.copy()
                for i, d in top_errors:
                    h, w = frame_dynamic.shape[:2]
                    p2 = res2.pose_landmarks.landmark[i]
                    x, y = int(p2.x * w), int(p2.y * h)
                    if d > threshold:
                        intensity = min(255, int(d * 1000))
                        cv2.circle(overlay, (x, y), 20, (0, 0, intensity), -1)
                alpha = 0.3
                frame_dynamic = cv2.addWeighted(overlay, alpha, frame_dynamic, 1 - alpha, 0)
                mp_drawing.draw_landmarks(frame_dynamic, res2.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
            # Combine frames based on intended orientation
            if is_vertical:
                combined_base = cv2.vconcat([frame1, frame_dynamic])
            else:
                combined_base = cv2.hconcat([frame1, frame_dynamic])
            
            num_repeats = 20 if avg_diff <= 1.5 * threshold else 30
            for t in range(num_repeats):
                combined_dynamic = combined_base.copy()
                
                pulse = 0.5 * (1 + np.sin(2 * np.pi * t / num_repeats))
                circle_radius = int(5 + 3 * pulse)
                text_scale = 0.7 + 0.2 * pulse
                
                if res2.pose_landmarks:
                    for i, d in top_errors:
                        h, w = frame_dynamic.shape[:2]
                        p2 = res2.pose_landmarks.landmark[i]
                        x, y = int(p2.x * w), int(p2.y * h)
                        x_offset = 0 if is_vertical else input_width
                        if d > threshold * 1.5:
                            color = (0, 0, 255)
                            label = f"{JOINT_NAMES[i]}: High"
                        elif d > threshold:
                            color = (0, 255, 255)
                            label = f"{JOINT_NAMES[i]}: Adjust"
                        else:
                            color = (0, 255, 0)
                            label = None
                        cv2.circle(combined_dynamic, (x + x_offset, y), circle_radius, color, -1)
                        if label:
                            cv2.putText(combined_dynamic, label, (x + x_offset + 10, y), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
                
                if avg_diff > 1.5 * threshold:
                    cv2.putText(combined_dynamic, "High Error Detected", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, text_scale, (0, 0, 255), 2)
                
                if top_errors:
                    summary = "Focus on: " + ", ".join(JOINT_NAMES[i] for i, _ in top_errors[:3])
                    opacity = 1.0
                    if t < num_repeats // 2:
                        opacity = t / (num_repeats // 2)
                    else:
                        opacity = 1 - (t - num_repeats // 2) / (num_repeats // 2)
                    overlay = combined_dynamic.copy()
                    cv2.putText(overlay, summary, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                                0.7, (255, 255, 255), 2)
                    cv2.addWeighted(overlay, opacity, combined_dynamic, 1 - opacity, 0, combined_dynamic)
                
                out_dynamic.write(combined_dynamic)
            
            window_diffs = []
            window_frames = []
    
    cap1.release()
    cap2.release()
    out_normal.release()
    out_dynamic.release()
    pose.close()
    
    if not os.path.exists(output_normal_path) or not os.path.exists(output_dynamic_path):
        logger.error(f"Comparison videos not created: normal={output_normal_path}, dynamic={output_dynamic_path}")
        raise Exception("Comparison videos not created")
    
    logger.info(f"Normal video size: {os.path.getsize(output_normal_path)} bytes")
    logger.info(f"Dynamic video size: {os.path.getsize(output_dynamic_path)} bytes")
    logger.info(f"Created comparison videos: normal={output_normal_path}, dynamic={output_dynamic_path}")
    return output_normal_path, output_dynamic_path
def export_differences_to_csv(differences, output_csv):
    try:
        with open(output_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["frame"] + JOINT_NAMES)
            for i, frame_diff in enumerate(differences):
                row = [i] + [f"{d:.4f}" if d is not None else "" for d in frame_diff]
                writer.writerow(row)
        logger.info(f"Exported differences to {output_csv}")
        return output_csv
    except Exception as e:
        logger.error(f"Error exporting CSV: {str(e)}")
        raise

def plot_average_errors(avg_diffs, output_path):
    try:
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(avg_diffs)), avg_diffs, color="skyblue")
        plt.xticks(range(len(avg_diffs)), JOINT_NAMES, rotation=90)
        plt.xlabel("Joint")
        plt.ylabel("Average Error")
        plt.title("Average Error per Joint")
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        logger.info(f"Saved average error chart to {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"Error plotting average errors: {str(e)}")
        raise

def plot_joint_errors_over_time(differences, joint_indices, output_prefix):
    num_frames = len(differences)
    plot_paths = []
    for idx in joint_indices:
        errors = [frame[idx] if frame[idx] is not None else np.nan for frame in differences]
        try:
            plt.figure(figsize=(10, 6))
            plt.plot(range(num_frames), errors, marker="o", linestyle="-")
            plt.xlabel("Frame Index")
            plt.ylabel("Error")
            plt.title(f"Error over Time: {JOINT_NAMES[idx]}")
            plt.tight_layout()
            out_file = f"{output_prefix}_{JOINT_NAMES[idx]}.png"
            plt.savefig(out_file)
            plt.close()
            logger.info(f"Saved time-series for {JOINT_NAMES[idx]} to {out_file}")
            plot_paths.append(out_file)
        except Exception as e:
            logger.error(f"Error plotting errors for joint {JOINT_NAMES[idx]}: {str(e)}")
            continue
    return plot_paths

def generate_analysis_results(differences, teacher_kps, student_kps, sport="general", threshold=0.1):
    if not differences:
        logger.error("No valid comparisons made")
        return {
            'success': False,
            'message': 'No valid comparisons could be made. Check video quality or content.'
        }
    
    avg_diffs, tips, region_scores, movement_accuracy = summarize_differences(
        differences, teacher_kps, student_kps, sport, threshold
    )
    
    accuracy_scores = compute_accuracy_score(differences, threshold)
    overall_accuracy = np.mean(accuracy_scores) if accuracy_scores else 0
    
    return {
        'success': True,
        'teacher_frames': sum(kp is not None for kp in teacher_kps),
        'student_frames': sum(kp is not None for kp in student_kps),
        'movement_accuracy': round(movement_accuracy, 2),
        'average_differences': avg_diffs.tolist(),
        'suggestions': tips,
        'overall_accuracy': round(overall_accuracy, 2),
        'region_scores': region_scores
    }

@app.route('/api/upload', methods=['POST'])
def upload_video():
    logger.debug("Received video upload request")
    if 'video' not in request.files:
        logger.error("No video file part in request")
        return jsonify({'success': False, 'message': 'No video file part'}), 400
        
    file = request.files['video']
    if file.filename == '':
        logger.error("No selected file")
        return jsonify({'success': False, 'message': 'No selected file'}), 400
        
    if not allowed_file(file.filename):
        logger.error(f"Invalid file type: {file.filename}")
        return jsonify({'success': False, 'message': 'Invalid file type'}), 400
        
    try:
        filename = f"{uuid.uuid4()}.{file.filename.rsplit('.', 1)[1].lower()}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        logger.info(f"Uploaded video: {filepath}")
        
        return jsonify({
            'success': True,
            'videoUrl': f"http://localhost:5000/static/uploads/{filename}",
            'videoId': filename.split('.')[0]
        })
    except Exception as e:
        logger.error(f"Upload failed: {str(e)}")
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/analyze', methods=['POST'])
def analyze_videos():
    logger.debug("Received analyze request")
    data = request.get_json()
    coach_video_id = data.get('coachVideoId')
    player_video_id = data.get('playerVideoId')
    sport = data.get('sport', 'general')  # Allow sport specification
    threshold = data.get('threshold', 0.1)  # Allow custom threshold
    
    if not coach_video_id or not player_video_id:
        logger.error("Missing video IDs")
        return jsonify({'success': False, 'message': 'Missing video IDs'}), 400
    
    coach_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{coach_video_id}.mp4")
    player_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{player_video_id}.mp4")
    
    if not os.path.exists(coach_path):
        logger.error(f"Coach video not found: {coach_path}")
        return jsonify({'success': False, 'message': f"Coach video not found: {coach_path}"}), 404
    if not os.path.exists(player_path):
        logger.error(f"Player video not found: {player_path}")
        return jsonify({'success': False, 'message': f"Player video not found: {player_path}"}), 404
    
    try:
        # Generate file names
        session_id = uuid.uuid4()
        normal_output = os.path.join(app.config['COMPARISON_FOLDER'], f"comparison_normal_{session_id}.mp4")
        dynamic_output = os.path.join(app.config['COMPARISON_FOLDER'], f"comparison_dynamic_{session_id}.mp4")
        csv_output = os.path.join(app.config['COMPARISON_FOLDER'], f"differences_{session_id}.csv")
        avg_error_plot = os.path.join(app.config['PLOT_FOLDER'], f"avg_errors_{session_id}.png")
        joint_error_prefix = os.path.join(app.config['PLOT_FOLDER'], f"joint_errors_{session_id}")
        
        # Extract keypoints and compare
        teacher_kps = extract_keypoints(coach_path)
        student_kps = extract_keypoints(player_path)
        
        if not teacher_kps or not student_kps:
            logger.error("Could not detect poses in one or both videos")
            return jsonify({'success': False, 'message': 'Could not detect poses in one or both videos'}), 400
        
        differences = compare_videos(teacher_kps, student_kps)
        results = generate_analysis_results(differences, teacher_kps, student_kps, sport, threshold)
        
        if not results['success']:
            logger.error("Analysis failed: No valid comparisons")
            return jsonify(results), 400
            
        # Generate comparison videos
        normal_path, dynamic_path = create_comparison_video(coach_path, player_path, normal_output, dynamic_output, threshold)
        
        # Export differences to CSV
        csv_path = export_differences_to_csv(differences, csv_output)
        
        # Generate plots
        avg_plot_path = plot_average_errors(np.array(results['average_differences']), avg_error_plot)
        
        key_joints = [
            mp_pose.PoseLandmark.RIGHT_WRIST.value,
            mp_pose.PoseLandmark.LEFT_KNEE.value,
            mp_pose.PoseLandmark.RIGHT_ANKLE.value
        ]
        joint_plot_paths = plot_joint_errors_over_time(differences, key_joints, joint_error_prefix)
        
        full_normal_url = f"http://localhost:5000/static/comparisons/{os.path.basename(normal_path)}"
        full_dynamic_url = f"http://localhost:5000/static/comparisons/{os.path.basename(dynamic_path)}"
        full_csv_url = f"http://localhost:5000/static/comparisons/{os.path.basename(csv_path)}"
        full_avg_plot_url = f"http://localhost:5000/static/plots/{os.path.basename(avg_plot_path)}"
        full_joint_plot_urls = [
            f"http://localhost:5000/static/plots/{os.path.basename(path)}" for path in joint_plot_paths
        ]
        
        results.update({
            'normalVideoUrl': full_normal_url,
            'dynamicVideoUrl': full_dynamic_url,
            'csvUrl': full_csv_url,
            'avgErrorPlotUrl': full_avg_plot_url,
            'jointErrorPlotUrls': full_joint_plot_urls
        })
        
        logger.info(f"Returning analysis results with URLs: normal={full_normal_url}, dynamic={full_dynamic_url}")
        return jsonify(results)
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        return jsonify({'success': False, 'message': f"Analysis failed: {str(e)}"}), 500

# @app.route('/static/comparisons/<filename>')
# def serve_comparison(filename):
#     logger.debug(f"Serving comparison file: {filename}")
#     try:
#         filepath = os.path.join(app.config['COMPARISON_FOLDER'], filename)
#         if not os.path.exists(filepath):
#             logger.error(f"Comparison file not found: {filepath}")
#             return jsonify({'success': False, 'message': f"Comparison file not found: {filepath}"}), 404
#         mime_type = 'video/mp4' if filename.endswith('.mp4') else 'text/csv'
#         logger.info(f"Serving comparison file from: {filepath}")
#         return send_from_directory(app.config['COMPARISON_FOLDER'], filename, mimetype=mime_type)
#     except Exception as e:
#         logger.error(f"Failed to serve comparison file: {str(e)}")
#         return jsonify({'success': False, 'message': str(e)}), 500
@app.route('/static/comparisons/<filename>')
def serve_comparison(filename):
    filepath = os.path.join(app.config['COMPARISON_FOLDER'], filename)
    logger.info(f"Attempting to serve file: {filepath}")
    if not os.path.exists(filepath):
        logger.error(f"Comparison file not found: {filepath}")
        return jsonify({'success': False, 'message': f"Comparison file not found: {filepath}"}), 404
    mime_type = 'video/mp4' if filename.endswith('.mp4') else 'text/csv'
    logger.info(f"Serving comparison file: {filepath} with mime_type: {mime_type}")
    return send_from_directory(app.config['COMPARISON_FOLDER'], filename, mimetype=mime_type)

@app.route('/static/plots/<filename>')
def serve_plot(filename):
    logger.debug(f"Serving plot file: {filename}")
    try:
        filepath = os.path.join(app.config['PLOT_FOLDER'], filename)
        if not os.path.exists(filepath):
            logger.error(f"Plot file not found: {filepath}")
            return jsonify({'success': False, 'message': f"Plot file not found: {filepath}"}), 404
        logger.info(f"Serving plot file from: {filepath}")
        return send_from_directory(app.config['PLOT_FOLDER'], filename, mimetype='image/png')
    except Exception as e:
        logger.error(f"Failed to serve plot file: {str(e)}")
        return jsonify({'success': False, 'message': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)