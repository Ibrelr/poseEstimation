#!/usr/bin/env python3
"""
Web Server for Pose Estimation
==============================

A Flask web server that serves the pose estimation functionality
using the existing demo.py code.

Usage:
    python web_server.py
    Then open: http://localhost:5000
"""

import cv2
import numpy as np
import base64
import json
from flask import Flask, render_template, Response, jsonify, request
from pose_estimation_advanced import PoseConfig, PoseDataProcessor, ActivityClassifier
import mediapipe as mp

app = Flask(__name__)

# Global variables
config = PoseConfig()
processor = PoseDataProcessor(config)
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Initialize MediaPipe Pose
pose = mp_pose.Pose(
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

@app.route('/')
def index():
    """Serve the main page."""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route."""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def generate_frames():
    """Generate video frames with pose detection."""
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    while True:
        success, frame = cap.read()
        if not success:
            break
        
        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process pose
        results = pose.process(rgb_frame)
        
        # Draw pose landmarks
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
            )
            
            # Extract landmarks for processing
            landmarks = processor.extract_landmarks(frame)
            if landmarks is not None:
                # Extract features
                features = processor.extract_features(landmarks)
                
                # Add FPS and feature info to frame
                cv2.putText(frame, f"Features: {len(features)} detected", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Add title
        cv2.putText(frame, "Pose Estimation - FUNGERER!", 
                   (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Encode frame
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    
    cap.release()

@app.route('/pose_data')
def pose_data():
    """Get current pose data as JSON."""
    # This would return current pose landmarks and features
    # For now, return dummy data
    return jsonify({
        'landmarks_detected': True,
        'activity': 'Standing',
        'confidence': 0.85,
        'keypoints': 17,
        'status': 'Active'
    })

if __name__ == '__main__':
    print("🚀 Starting Pose Estimation Web Server...")
    print("📺 Open your browser to: http://localhost:5000")
    print("⏹️  Press Ctrl+C to stop")
    
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True) 