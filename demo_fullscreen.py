#!/usr/bin/env python3
"""
Fullscreen Pose Estimation Demo
===============================

A demo script that shows pose estimation in fullscreen or large window.
This version maximizes the camera view for better visibility.
"""

import cv2
import numpy as np
import time
import argparse
from pose_estimation_advanced import PoseConfig, PoseDataProcessor, ActivityClassifier

def main():
    """Main demo function."""
    parser = argparse.ArgumentParser(description='Fullscreen Pose Estimation Demo')
    parser.add_argument('--model', type=str, help='Path to trained model')
    parser.add_argument('--camera', type=int, default=0, help='Camera device ID')
    parser.add_argument('--width', type=int, default=1280, help='Camera width')
    parser.add_argument('--height', type=int, default=720, help='Camera height')
    parser.add_argument('--fullscreen', action='store_true', help='Run in fullscreen mode')
    
    args = parser.parse_args()
    
    # Initialize configuration
    config = PoseConfig()
    
    # Initialize pose processor
    processor = PoseDataProcessor(config)
    
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
    
    # Set camera properties for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    # Get actual camera resolution
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera resolution: {actual_width}x{actual_height}")
    
    print("Fullscreen Pose Estimation Demo")
    print("Press 'q' to quit, 's' to save screenshot, 'f' to toggle fullscreen")
    print("This version maximizes the camera view!")
    
    # Create window
    window_name = 'Fullscreen Pose Estimation Demo'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    if args.fullscreen:
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    else:
        # Set window size to camera resolution
        cv2.resizeWindow(window_name, actual_width, actual_height)
    
    frame_count = 0
    start_time = time.time()
    fps = 0.0
    fullscreen_mode = args.fullscreen
    
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
            # Draw pose landmarks
            frame = draw_pose_landmarks(frame, landmarks)
            
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
        
        # Display FPS and resolution info
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, actual_height - 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Resolution: {actual_width}x{actual_height}", (10, actual_height - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Display frame
        cv2.imshow(window_name, frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # Save screenshot
            timestamp = int(time.time())
            filename = f"pose_demo_fullscreen_{timestamp}.jpg"
            cv2.imwrite(filename, frame)
            print(f"Screenshot saved: {filename}")
        elif key == ord('f'):
            # Toggle fullscreen
            fullscreen_mode = not fullscreen_mode
            if fullscreen_mode:
                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                print("Fullscreen mode ON")
            else:
                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(window_name, actual_width, actual_height)
                print("Fullscreen mode OFF")
    
    cap.release()
    cv2.destroyAllWindows()

def draw_pose_landmarks(frame, landmarks):
    """Draw pose landmarks and connections on frame using OpenCV."""
    # landmarks: np.ndarray shape (99,) -> (33, 3)
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
        
        # Additional connections for better visualization
        (11, 23), (12, 24),  # Torso sides
        (23, 25), (24, 26),  # Hip to knees
        (25, 27), (26, 28),  # Knees to ankles
    ]

    # Draw connections (lines) - thicker for better visibility
    for connection in pose_connections:
        start_idx, end_idx = connection
        if start_idx < len(landmarks_3d) and end_idx < len(landmarks_3d):
            start_point = landmarks_3d[start_idx]
            end_point = landmarks_3d[end_idx]
            
            # Convert to pixel coordinates
            start_x, start_y = int(start_point[0] * w), int(start_point[1] * h)
            end_x, end_y = int(end_point[0] * w), int(end_point[1] * h)
            
            # Draw line - thicker for better visibility
            cv2.line(frame, (start_x, start_y), (end_x, end_y), (255, 0, 0), 3)

    # Draw landmarks (points) - larger for better visibility
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
        
        # Larger circles for better visibility
        cv2.circle(frame, (cx, cy), 6, color, -1)
        
        # Add landmark number for key points
        if i in [0, 11, 12, 23, 24]:  # nose, shoulders, hips
            cv2.putText(frame, str(i), (cx+8, cy-8), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

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
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_offset += 25
        
        # Limit number of features displayed
        if y_offset > 400:
            break

if __name__ == "__main__":
    main() 