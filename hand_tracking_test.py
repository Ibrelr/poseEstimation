#!/usr/bin/env python3
"""
Hand Tracking Test
=================

A simple test to demonstrate the difference between pose estimation and hand tracking.
This shows why you need MediaPipe Hands for precise finger detection.
"""

import cv2
import numpy as np
import mediapipe as mp
from pose_estimation_advanced import PoseConfig, PoseDataProcessor

def main():
    """Main test function."""
    print("Hand Tracking Test")
    print("=" * 50)
    print("This test shows the difference between:")
    print("1. Pose estimation (body joints only)")
    print("2. Hand tracking (precise finger detection)")
    print("=" * 50)
    
    # Initialize pose processor
    config = PoseConfig()
    processor = PoseDataProcessor(config)
    
    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        max_num_hands=2
    )
    
    # Initialize MediaPipe drawing
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    print("Instructions:")
    print("- Show your hands to the camera")
    print("- Move your fingers to see the difference")
    print("- Press 'q' to quit")
    print("- Press '1' to show only pose estimation")
    print("- Press '2' to show only hand tracking")
    print("- Press '3' to show both")
    
    show_pose = True
    show_hands = True
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Flip frame
        frame = cv2.flip(frame, 1)
        
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process pose estimation
        pose_landmarks = None
        if show_pose:
            pose_landmarks = processor.extract_landmarks(frame)
        
        # Process hand tracking
        hand_results = None
        if show_hands:
            hand_results = hands.process(rgb_frame)
        
        # Draw pose landmarks (if enabled)
        if show_pose and pose_landmarks is not None:
            frame = draw_pose_simple(frame, pose_landmarks)
            cv2.putText(frame, "POSE ESTIMATION: Body joints only", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw hand landmarks (if enabled)
        if show_hands and hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                # Draw hand landmarks
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style(),
                    connection_drawing_spec=mp_drawing_styles.get_default_hand_connections_style()
                )
                
                # Count extended fingers
                extended_fingers = count_extended_fingers(hand_landmarks, frame.shape[1], frame.shape[0])
                cv2.putText(frame, f"Extended fingers: {extended_fingers}", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            cv2.putText(frame, "HAND TRACKING: Precise finger detection", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # Display mode info
        mode_text = f"Mode: {'Pose' if show_pose else ''}{' + ' if show_pose and show_hands else ''}{'Hands' if show_hands else ''}"
        cv2.putText(frame, mode_text, (10, frame.shape[0] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Display frame
        cv2.imshow('Hand Tracking Test', frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('1'):
            show_pose = True
            show_hands = False
            print("Showing only pose estimation")
        elif key == ord('2'):
            show_pose = False
            show_hands = True
            print("Showing only hand tracking")
        elif key == ord('3'):
            show_pose = True
            show_hands = True
            print("Showing both pose and hand tracking")
    
    cap.release()
    cv2.destroyAllWindows()

def draw_pose_simple(frame, landmarks):
    """Draw pose landmarks in a simple way."""
    landmarks_3d = landmarks.reshape(-1, 3)
    h, w, _ = frame.shape
    
    # Draw only arm landmarks (wrists and elbows)
    arm_landmarks = [11, 12, 13, 14, 15, 16]  # shoulders, elbows, wrists
    
    for i in arm_landmarks:
        if i < len(landmarks_3d):
            x, y, z = landmarks_3d[i]
            cx, cy = int(x * w), int(y * h)
            cv2.circle(frame, (cx, cy), 8, (0, 255, 0), -1)
            cv2.putText(frame, str(i), (cx+10, cy-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    return frame

def count_extended_fingers(hand_landmarks, width, height):
    """Count how many fingers are extended."""
    landmarks = []
    for landmark in hand_landmarks.landmark:
        x = landmark.x * width
        y = landmark.y * height
        landmarks.append((x, y))
    
    # Check if fingers are extended
    # A finger is extended if the tip is higher than the middle joint
    thumb_extended = landmarks[4][1] < landmarks[3][1]
    index_extended = landmarks[8][1] < landmarks[6][1]
    middle_extended = landmarks[12][1] < landmarks[10][1]
    ring_extended = landmarks[16][1] < landmarks[14][1]
    pinky_extended = landmarks[20][1] < landmarks[18][1]
    
    return sum([thumb_extended, index_extended, middle_extended, ring_extended, pinky_extended])

if __name__ == "__main__":
    main() 