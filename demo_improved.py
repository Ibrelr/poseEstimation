#!/usr/bin/env python3
"""
Improved Pose Estimation Demo
============================

A demo script that properly draws pose landmarks with skeleton connections.
This version shows the full pose structure, not just dots.
"""

import cv2
import numpy as np
import time
import argparse
import mediapipe as mp
from pose_estimation_advanced import PoseConfig, PoseDataProcessor, ActivityClassifier

def main():
    """Main demo function."""
    parser = argparse.ArgumentParser(description='Improved Pose Estimation Demo')
    parser.add_argument('--model', type=str, help='Path to trained model')
    parser.add_argument('--camera', type=int, default=0, help='Camera device ID')
    parser.add_argument('--width', type=int, default=640, help='Camera width')
    parser.add_argument('--height', type=int, default=480, help='Camera height')
    
    args = parser.parse_args()
    
    # Initialize configuration
    config = PoseConfig()
    
    # Initialize pose processor
    processor = PoseDataProcessor(config)
    
    # Initialize MediaPipe drawing utilities
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_pose = mp.solutions.pose
    
    # Initialize activity classifier (if model provided)
    classifier = None
    if args.model:
        try:
            classifier = ActivityClassifier(num_classes=5)
            classifier.load_model(args.model)
            print(f"Loaded model from {args.model}")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Running without activity classification")
    
    # Initialize camera
    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    print("Improved Pose Estimation Demo")
    print("Press 'q' to quit, 's' to save screenshot")
    print("This version shows skeleton connections!")
    
    frame_count = 0
    start_time = time.time()
    fps = 0.0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break
        
        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)
        
        # Extract pose landmarks
        landmarks = processor.extract_landmarks(frame)
        
        if landmarks is not None:
            # Draw pose landmarks with skeleton
            frame = draw_pose_with_skeleton(frame, landmarks, mp_drawing, mp_pose)
            
            # Extract features
            features = processor.extract_features(landmarks)
            
            # Classify activity if model is available
            if classifier:
                class_id, confidence = classifier.predict(landmarks)
                activity = f"Activity: {class_id} ({confidence:.2f})"
                cv2.putText(frame, activity, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Display features
            display_features(frame, features)
        
        # Calculate and display FPS
        frame_count += 1
        if frame_count % 30 == 0:
            elapsed_time = time.time() - start_time
            fps = 30 / elapsed_time
            start_time = time.time()
        
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, args.height - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Display frame
        cv2.imshow('Improved Pose Estimation Demo', frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # Save screenshot
            timestamp = int(time.time())
            filename = f"pose_demo_improved_{timestamp}.jpg"
            cv2.imwrite(filename, frame)
            print(f"Screenshot saved: {filename}")
    
    cap.release()
    cv2.destroyAllWindows()

def draw_pose_with_skeleton(frame, landmarks, mp_drawing, mp_pose):
    """Draw pose landmarks with skeleton connections using MediaPipe."""
    landmarks_3d = landmarks.reshape(-1, 3)
    h, w, _ = frame.shape
    
    # Create MediaPipe pose landmarks object
    pose_landmarks = mp_pose.PoseLandmark
    
    # Convert our landmarks to MediaPipe format
    mp_landmarks = []
    for i, (x, y, z) in enumerate(landmarks_3d):
        if i < 33:  # MediaPipe pose has 33 landmarks
            landmark = pose_landmarks()
            landmark.x = x
            landmark.y = y
            landmark.z = z
            landmark.visibility = 1.0  # Assume visible
            mp_landmarks.append(landmark)
    
    # Draw pose with MediaPipe's built-in drawing
    mp_drawing.draw_landmarks(
        frame,
        mp_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
    )
    
    return frame

def draw_pose_landmarks_custom(frame, landmarks):
    """Draw pose landmarks with custom skeleton connections."""
    landmarks_3d = landmarks.reshape(-1, 3)
    h, w, _ = frame.shape

    # Define pose connections (skeleton structure)
    pose_connections = [
        # Face
        (0, 1), (1, 2), (2, 3), (3, 7),  # Nose to eyes to ears
        (0, 4), (4, 5), (5, 6), (6, 8),  # Nose to mouth
        
        # Upper body
        (11, 12),  # Shoulders
        (11, 13), (13, 15),  # Left arm
        (12, 14), (14, 16),  # Right arm
        
        # Torso
        (11, 23), (12, 24),  # Shoulders to hips
        (23, 24),  # Hips
        
        # Lower body
        (23, 25), (25, 27),  # Left leg
        (24, 26), (26, 28),  # Right leg
    ]

    # Draw connections (lines)
    for connection in pose_connections:
        start_idx, end_idx = connection
        if start_idx < len(landmarks_3d) and end_idx < len(landmarks_3d):
            start_point = landmarks_3d[start_idx]
            end_point = landmarks_3d[end_idx]
            
            # Convert to pixel coordinates
            start_x, start_y = int(start_point[0] * w), int(start_point[1] * h)
            end_x, end_y = int(end_point[0] * w), int(end_point[1] * h)
            
            # Draw line
            cv2.line(frame, (start_x, start_y), (end_x, end_y), (255, 0, 0), 2)

    # Draw landmarks (points)
    for i, (x, y, z) in enumerate(landmarks_3d):
        cx, cy = int(x * w), int(y * h)
        # Different colors for different body parts
        if i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:  # Face
            color = (0, 255, 255)  # Yellow
        elif i in [11, 12, 13, 14, 15, 16]:  # Arms
            color = (0, 255, 0)  # Green
        elif i in [23, 24, 25, 26, 27, 28]:  # Legs
            color = (255, 0, 0)  # Blue
        else:
            color = (255, 255, 255)  # White for others
        
        cv2.circle(frame, (cx, cy), 4, color, -1)

    return frame

def display_features(frame, features):
    """Display extracted features on frame."""
    if not features:
        return
    
    y_offset = 60
    for feature, value in features.items():
        if isinstance(value, bool):
            text = f"{feature}: {'Yes' if value else 'No'}"
        else:
            text = f"{feature}: {value:.3f}"
        
        cv2.putText(frame, text, (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        y_offset += 20
        
        # Limit number of features displayed
        if y_offset > 300:
            break

if __name__ == "__main__":
    main() 