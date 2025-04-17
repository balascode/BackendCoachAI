import os
import cv2
import numpy as np
import mediapipe as mp
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from flask_socketio import SocketIO, emit
from werkzeug.utils import secure_filename
import uuid
import logging
import base64
from io import BytesIO
from PIL import Image
from live_realtime import load_teacher_reference, calculate_difference, generate_heuristics, mp_pose, mp_drawing, JOINT_NAMES
from mediapipe.framework.formats import landmark_pb2

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}, r"/static/*": {"origins": "*"}})
socketio = SocketIO(app, cors_allowed_origins="*")

# Configuration
UPLOAD_FOLDER = 'static/live_uploads'
COMPARISON_FOLDER = 'static/live_comparisons'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['COMPARISON_FOLDER'] = COMPARISON_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB limit

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(COMPARISON_FOLDER, exist_ok=True)

# Global storage for teacher keypoints
teacher_kps = None
session_differences = []
session_frame_count = 0
last_tips = []
last_update = 0
threshold = 0.1
sport = "general"

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/api/live/upload', methods=['POST'])
def upload_coach_video():
    logger.debug("Received coach video upload request")
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
        logger.info(f"Uploaded coach video: {filepath}")

        # Load teacher keypoints
        global teacher_kps
        teacher_kps = load_teacher_reference(filepath)
        if not teacher_kps:
            logger.error("Failed to extract keypoints from coach video")
            return jsonify({'success': False, 'message': 'Failed to extract keypoints from coach video'}), 400

        return jsonify({
            'success': True,
            'videoUrl': f"http://localhost:5001/static/live_uploads/{filename}",
            'videoId': filename.split('.')[0]
        })
    except Exception as e:
        logger.error(f"Upload failed: {str(e)}")
        return jsonify({'success': False, 'message': str(e)}), 500

@socketio.on('connect')
def handle_connect():
    logger.debug("Client connected to WebSocket")
    emit('connection_status', {'status': 'connected'})

@socketio.on('disconnect')
def handle_disconnect():
    logger.debug("Client disconnected from WebSocket")

@socketio.on('live_frame')
def handle_live_frame(data):
    global session_differences, session_frame_count, last_tips, last_update
    if not teacher_kps:
        emit('error', {'message': 'No coach video uploaded'})
        return

    try:
        # Decode base64 frame
        img_data = base64.b64decode(data['frame'].split(',')[1])
        img = Image.open(BytesIO(img_data))
        frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        # Process frame with MediaPipe
        pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose.process(img_rgb)

        student_kps = None
        diffs = None
        accuracy = 100.0
        if res.pose_landmarks:
            student_kps = [(lm.x, lm.y, lm.z) for lm in res.pose_landmarks.landmark]
            diffs = calculate_difference(teacher_kps, student_kps)
            session_differences.append(diffs)

            # Limit differences history
            if len(session_differences) > 30:
                session_differences.pop(0)

            # Calculate accuracy
            valid_diffs = [d for d in diffs if d is not None]
            mean_diff = np.mean(valid_diffs) if valid_diffs else 0
            accuracy = max(0, 100 - (mean_diff * 100))

            # Generate tips every 30 frames
            if session_frame_count % 30 == 0:
                avg_diffs = np.nanmean(session_differences, axis=0)
                if isinstance(avg_diffs, np.ndarray):
                    new_tips, _ = generate_heuristics(
                        avg_diffs,
                        session_differences,
                        sport=sport,
                        joint_threshold=threshold,
                        teacher_kps=[teacher_kps],
                        student_kps=[student_kps]
                    )
                    last_tips = new_tips[:2]
                    last_update = session_frame_count

        # Draw teacher keypoints as ghost overlay
        teacher_pose = landmark_pb2.NormalizedLandmarkList()
        for kp in teacher_kps:
            landmark = teacher_pose.landmark.add()
            landmark.x, landmark.y, landmark.z = kp
        overlay = frame.copy()
        teacher_style = mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=1, circle_radius=2)
        mp_drawing.draw_landmarks(overlay, teacher_pose, mp_pose.POSE_CONNECTIONS,
                                landmark_drawing_spec=teacher_style)
        alpha = 0.4
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

        # Visual alerts for student keypoints
        if res.pose_landmarks and diffs:
            overlay = frame.copy()
            for i, d in enumerate(diffs):
                if d is not None and d > threshold:
                    h, w = frame.shape[:2]
                    lm = res.pose_landmarks.landmark[i]
                    x, y = int(lm.x * w), int(lm.y * h)
                    pulse = 0.5 * (1 + np.sin(2 * np.pi * session_frame_count / 20))
                    radius = int(5 + 3 * pulse)
                    color = (0, 0, 255) if d > threshold * 1.5 else (0, 255, 255)
                    cv2.circle(overlay, (x, y), radius, color, -1)
                    if d > threshold * 1.5:
                        cv2.putText(overlay, f"{JOINT_NAMES[i]}: High", (x + 10, y),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            alpha = 0.4
            frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

            # Draw student landmarks
            mp_drawing.draw_landmarks(frame, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Display accuracy
        text = f"Accuracy: {accuracy:.2f}%"
        cv2.rectangle(frame, (5, 5, 220, 45), (0, 0, 0), -1)
        cv2.putText(frame, text, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4)
        cv2.putText(frame, text, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        # Display tips
        cv2.rectangle(frame, (900, 50, 1270, 300), (0, 0, 0), -1)
        cv2.putText(frame, "Feedback", (910, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4)
        cv2.putText(frame, "Feedback", (910, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        for i, tip in enumerate(last_tips):
            y_pos = 110 + i * 40
            cv2.putText(frame, tip, (910, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3)
            cv2.putText(frame, tip, (910, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)
        cv2.putText(frame, f"Last updated: frame {last_update}", (910, 290),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
        cv2.putText(frame, f"Last updated: frame {last_update}", (910, 290),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Encode frame to base64
        _, buffer = cv2.imencode('.jpg', frame)
        frame_b64 = base64.b64encode(buffer).decode('utf-8')
        pose.close()

        # Emit processed frame and feedback
        emit('processed_frame', {
            'frame': f"data:image/jpeg;base64,{frame_b64}",
            'accuracy': accuracy,
            'tips': last_tips
        })

        session_frame_count += 1

    except Exception as e:
        logger.error|f"Error processing frame: {str(e)}"
        emit('error', {'message': f"Error processing frame: {str(e)}"})

@app.route('/static/live_uploads/<filename>')
def serve_uploaded_video(filename):
    logger.debug(f"Serving uploaded video: {filename}")
    try:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if not os.path.exists(filepath):
            logger.error(f"Video not found: {filepath}")
            return jsonify({'success': False, 'message': f"Video not found: {filepath}"}), 404
        return send_from_directory(app.config['UPLOAD_FOLDER'], filename, mimetype='video/mp4')
    except Exception as e:
        logger.error(f"Failed to serve video: {str(e)}")
        return jsonify({'success': False, 'message': str(e)}), 500

if __name__ == '__main__':
    try:
        logger.info("Starting live server on port 5001")
        socketio.run(app, host='0.0.0.0', port=5001, debug=True, use_reloader=False)
    finally:
        pose.close()