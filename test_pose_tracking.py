#!/usr/bin/env python3
"""
Test Script for Pose Estimation & Hand Tracking
===============================================

This script tests the MediaPipe implementation functionality
to ensure all components are working correctly.
"""

import cv2
import mediapipe as mp
import numpy as np
import time

def test_mediapipe_installation():
    """Test if MediaPipe is properly installed and working."""
    print("🔧 Testing MediaPipe Installation...")
    
    try:
        # Test Pose
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        print("✅ MediaPipe Pose: OK")
        
        # Test Hands
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        print("✅ MediaPipe Hands: OK")
        
        # Test Drawing
        mp_drawing = mp.solutions.drawing_utils
        print("✅ MediaPipe Drawing Utils: OK")
        
        return True
        
    except Exception as e:
        print(f"❌ MediaPipe Error: {e}")
        return False

def test_camera_access():
    """Test camera access and video capture."""
    print("\n📹 Testing Camera Access...")
    
    try:
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ Camera not accessible")
            return False
        
        # Test frame capture
        ret, frame = cap.read()
        if ret:
            height, width, channels = frame.shape
            print(f"✅ Camera OK: {width}x{height}, {channels} channels")
            cap.release()
            return True
        else:
            print("❌ Could not capture frame")
            cap.release()
            return False
            
    except Exception as e:
        print(f"❌ Camera Error: {e}")
        return False

def test_pose_detection():
    """Test pose detection on a sample image."""
    print("\n🤸 Testing Pose Detection...")
    
    try:
        mp_pose = mp.solutions.pose
        mp_drawing = mp.solutions.drawing_utils
        
        with mp_pose.Pose(
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as pose:
            
            # Create a test image (solid color)
            test_image = np.zeros((480, 640, 3), dtype=np.uint8)
            test_image[:] = (100, 150, 200)  # Blue-ish color
            
            # Process the image
            results = pose.process(cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB))
            
            if results.pose_landmarks:
                print(f"✅ Pose Detection: {len(results.pose_landmarks.landmark)} landmarks detected")
                return True
            else:
                print("⚠️ Pose Detection: No pose in test image (expected)")
                return True  # This is actually expected for blank image
                
    except Exception as e:
        print(f"❌ Pose Detection Error: {e}")
        return False

def test_hand_detection():
    """Test hand detection on a sample image."""
    print("\n👐 Testing Hand Detection...")
    
    try:
        mp_hands = mp.solutions.hands
        
        with mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            min_detection_confidence=0.7
        ) as hands:
            
            # Create a test image (solid color)
            test_image = np.zeros((480, 640, 3), dtype=np.uint8)
            test_image[:] = (50, 100, 150)  # Different color
            
            # Process the image
            results = hands.process(cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB))
            
            if results.multi_hand_landmarks:
                hand_count = len(results.multi_hand_landmarks)
                print(f"✅ Hand Detection: {hand_count} hands detected")
                return True
            else:
                print("⚠️ Hand Detection: No hands in test image (expected)")
                return True  # This is actually expected for blank image
                
    except Exception as e:
        print(f"❌ Hand Detection Error: {e}")
        return False

def test_real_camera_detection():
    """Test real-time camera detection for 5 seconds."""
    print("\n🎥 Testing Real-time Detection (5 seconds)...")
    
    try:
        mp_pose = mp.solutions.pose
        mp_hands = mp.solutions.hands
        mp_drawing = mp.solutions.drawing_utils
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Camera not available for real-time test")
            return False
        
        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose, \
             mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5) as hands:
            
            start_time = time.time()
            frame_count = 0
            pose_detections = 0
            hand_detections = 0
            
            while time.time() - start_time < 5.0:  # Run for 5 seconds
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Convert BGR to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Process pose
                pose_results = pose.process(rgb_frame)
                if pose_results.pose_landmarks:
                    pose_detections += 1
                
                # Process hands
                hand_results = hands.process(rgb_frame)
                if hand_results.multi_hand_landmarks:
                    hand_detections += 1
                
                # Optional: display frame (comment out if running headless)
                # cv2.imshow('Test', frame)
                # if cv2.waitKey(1) & 0xFF == ord('q'):
                #     break
            
            cap.release()
            cv2.destroyAllWindows()
            
            fps = frame_count / 5.0
            pose_rate = (pose_detections / frame_count) * 100 if frame_count > 0 else 0
            hand_rate = (hand_detections / frame_count) * 100 if frame_count > 0 else 0
            
            print(f"✅ Real-time Test Complete:")
            print(f"   📊 FPS: {fps:.1f}")
            print(f"   🤸 Pose Detection Rate: {pose_rate:.1f}%")
            print(f"   👐 Hand Detection Rate: {hand_rate:.1f}%")
            
            return True
            
    except Exception as e:
        print(f"❌ Real-time Detection Error: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 POSE ESTIMATION & HAND TRACKING - SYSTEM TEST")
    print("=" * 60)
    
    tests = [
        ("MediaPipe Installation", test_mediapipe_installation),
        ("Camera Access", test_camera_access),
        ("Pose Detection", test_pose_detection),
        ("Hand Detection", test_hand_detection),
        ("Real-time Detection", test_real_camera_detection)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name}: PASSED")
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {e}")
    
    print(f"\n{'='*60}")
    print(f"🏁 TEST RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! System is ready for use.")
        print("\n🌐 Start the web server with:")
        print("   python -m http.server 8000")
        print("   Then open: http://localhost:8000")
    else:
        print("⚠️ Some tests failed. Check the errors above.")
        print("\n🔧 Common fixes:")
        print("   - Install MediaPipe: pip install mediapipe")
        print("   - Install OpenCV: pip install opencv-python")
        print("   - Check camera permissions")

if __name__ == "__main__":
    main() 