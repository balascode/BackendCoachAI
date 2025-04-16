import os
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import cv2
import mediapipe as mp
import numpy as np
from werkzeug.utils import secure_filename
import tempfile
import shutil
import uuid

app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests from frontend

# Configuration
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB limit

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_keypoints(video_path):
    """Extract pose keypoints from video using MediaPipe"""
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    cap = cv2.VideoCapture(video_path)
    keypoints = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Process frame
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)
        
        if results.pose_landmarks:
            frame_keypoints = [(lm.x, lm.y, lm.z) for lm in results.pose_landmarks.landmark]
        else:
            frame_keypoints = None
            
        keypoints.append(frame_keypoints)
    
    cap.release()
    pose.close()
    return keypoints

def calculate_difference(kp1, kp2):
    """Calculate Euclidean distance between corresponding keypoints"""
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
    """Compare keypoints frame by frame"""
    frame_differences = []
    min_length = min(len(teacher_kps), len(student_kps))
    
    for i in range(min_length):
        t_kp = teacher_kps[i]
        s_kp = student_kps[i]
        
        if t_kp and s_kp:
            frame_diff = calculate_difference(t_kp, s_kp)
            if frame_diff:
                frame_differences.append(frame_diff)
    
    return frame_differences

def compute_accuracy_score(differences, threshold=0.1):
    """Calculate accuracy scores for each frame"""
    scores = []
    for frame_diff in differences:
        valid_diffs = [d for d in frame_diff if d is not None]
        if not valid_diffs:
            scores.append(0)
            continue
        match_count = sum(1 for d in valid_diffs if d <= threshold)
        scores.append((match_count / len(valid_diffs)) * 100)
    return scores

def generate_analysis_results(differences, teacher_kps, student_kps):
    """Generate comprehensive analysis results matching the expected output format"""
    if not differences:
        return {
            'success': False,
            'message': 'No valid comparisons could be made. Check video quality or content.'
        }
    
    # Calculate average differences per landmark
    valid_diffs = np.array([d for d in differences if d is not None])
    avg_diffs = np.nanmean(valid_diffs, axis=0)
    
    # Generate performance metrics
    accuracy_scores = compute_accuracy_score(differences)
    overall_accuracy = np.mean(accuracy_scores) if accuracy_scores else 0
    
    # Calculate movement accuracy (similar to your original output)
    mean_diff = np.mean(avg_diffs[~np.isnan(avg_diffs)])
    movement_accuracy = max(0, 100 - (mean_diff * 100))
    
    # Generate improvement tips
    tips = []
    if len(avg_diffs) > mp_pose.PoseLandmark.RIGHT_WRIST.value and avg_diffs[mp_pose.PoseLandmark.RIGHT_WRIST.value] > 0.1:
        tips.append("Try keeping your right hand steadier.")
    if len(avg_diffs) > mp_pose.PoseLandmark.LEFT_KNEE.value and avg_diffs[mp_pose.PoseLandmark.LEFT_KNEE.value] > 0.1:
        tips.append("Work on your left knee position during lunges.")
    if len(avg_diffs) > mp_pose.PoseLandmark.RIGHT_KNEE.value and avg_diffs[mp_pose.PoseLandmark.RIGHT_KNEE.value] > 0.1:
        tips.append("Focus on maintaining right knee alignment.")
    
    if not tips:
        tips.append("Your form is generally good! Focus on maintaining consistency with the coach's movements.")
    
    return {
        'success': True,
        'teacher_frames': sum(kp is not None for kp in teacher_kps),
        'student_frames': sum(kp is not None for kp in student_kps),
        'movement_accuracy': round(movement_accuracy, 2),
        'average_differences': avg_diffs.tolist(),
        'suggestions': tips,
        'overall_accuracy': round(overall_accuracy, 2)
    }

def create_comparison_video(coach_path, player_path, output_path):
    """Create a side-by-side comparison video with pose landmarks"""
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    
    cap1 = cv2.VideoCapture(coach_path)
    cap2 = cv2.VideoCapture(player_path)
    
    # Get video properties
    width = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap1.get(cv2.CAP_PROP_FPS)
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width * 2, height))
    
    while cap1.isOpened() and cap2.isOpened():
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        
        if not ret1 or not ret2:
            break
            
        # Process both frames
        for frame in [frame1, frame2]:
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)
            
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
                )
        
        # Combine frames side by side
        combined = cv2.hconcat([frame1, frame2])
        out.write(combined)
    
    cap1.release()
    cap2.release()
    out.release()
    pose.close()
    return output_path

@app.route('/api/upload', methods=['POST'])
def upload_video():
    """Handle video uploads"""
    if 'video' not in request.files:
        return jsonify({'success': False, 'message': 'No video file part'}), 400
        
    file = request.files['video']
    if file.filename == '':
        return jsonify({'success': False, 'message': 'No selected file'}), 400
        
    if not allowed_file(file.filename):
        return jsonify({'success': False, 'message': 'Invalid file type'}), 400
        
    try:
        # Create unique filename
        filename = f"{uuid.uuid4()}.{file.filename.rsplit('.', 1)[1].lower()}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        return jsonify({
            'success': True,
            'videoUrl': f"/static/uploads/{filename}",
            'videoId': filename.split('.')[0]
        })
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/analyze', methods=['POST'])
def analyze_videos():
    """Analyze and compare two videos"""
    data = request.get_json()
    coach_video_id = data.get('coachVideoId')
    player_video_id = data.get('playerVideoId')
    
    if not coach_video_id or not player_video_id:
        return jsonify({'success': False, 'message': 'Missing video IDs'}), 400
    
    coach_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{coach_video_id}.mp4")
    player_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{player_video_id}.mp4")
    
    if not os.path.exists(coach_path) or not os.path.exists(player_path):
        return jsonify({'success': False, 'message': 'Video files not found'}), 404
    
    try:
        # Create temp directory for output
        temp_dir = tempfile.mkdtemp()
        output_video_path = os.path.join(temp_dir, 'comparison.mp4')
        
        # Process videos
        teacher_kps = extract_keypoints(coach_path)
        student_kps = extract_keypoints(player_path)
        
        if not teacher_kps or not student_kps:
            return jsonify({'success': False, 'message': 'Could not detect poses in one or both videos'}), 400
        
        differences = compare_videos(teacher_kps, student_kps)
        results = generate_analysis_results(differences, teacher_kps, student_kps)
        
        if not results['success']:
            return jsonify(results), 400
            
        # Create comparison video
        comparison_path = create_comparison_video(coach_path, player_path, output_video_path)
        results['comparisonVideoUrl'] = f"/api/comparison/{os.path.basename(comparison_path)}"
        
        return jsonify(results)
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500
    finally:
        # Clean up temp files
        if 'temp_dir' in locals() and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

@app.route('/api/comparison/<filename>', methods=['GET'])
def get_comparison_video(filename):
    """Serve the comparison video"""
    try:
        temp_dir = tempfile.gettempdir()
        filepath = os.path.join(temp_dir, filename)
        
        if not os.path.exists(filepath):
            return jsonify({'success': False, 'message': 'Comparison video not found'}), 404
            
        return send_file(filepath, mimetype='video/mp4')
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)