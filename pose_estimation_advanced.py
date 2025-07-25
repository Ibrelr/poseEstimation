#!/usr/bin/env python3
"""
Advanced Pose Estimation System
===============================

This script provides advanced pose estimation capabilities including:
- Data processing and augmentation
- Model training and fine-tuning
- Pose analysis and visualization
- Activity classification
- Performance evaluation

Author: Pose Estimation Project
Date: 2024
"""

import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import json
import os
import argparse
from typing import List, Dict, Tuple, Optional
import logging
from dataclasses import dataclass
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class PoseConfig:
    """Configuration for pose estimation system."""
    model_complexity: int = 1
    smooth_landmarks: bool = True
    enable_segmentation: bool = False
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    max_num_poses: int = 1

class PoseDataProcessor:
    """Handles pose data processing and augmentation."""
    
    def __init__(self, config: PoseConfig):
        self.config = config
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            model_complexity=config.model_complexity,
            smooth_landmarks=config.smooth_landmarks,
            enable_segmentation=config.enable_segmentation,
            min_detection_confidence=config.min_detection_confidence,
            min_tracking_confidence=config.min_tracking_confidence
        )
        
    def extract_landmarks(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Extract pose landmarks from image."""
        try:
            results = self.pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            if results.pose_landmarks:
                landmarks = []
                for landmark in results.pose_landmarks.landmark:
                    landmarks.extend([landmark.x, landmark.y, landmark.z])
                return np.array(landmarks)
            return None
        except Exception as e:
            logger.error(f"Error extracting landmarks: {e}")
            return None
    
    def normalize_landmarks(self, landmarks: np.ndarray) -> np.ndarray:
        """Normalize landmarks to be scale and position invariant."""
        if landmarks is None or len(landmarks) == 0:
            return np.array([])
        
        # Reshape to (num_landmarks, 3)
        landmarks_3d = landmarks.reshape(-1, 3)
        
        # Center around hip center (landmarks 23 and 24)
        hip_center = (landmarks_3d[23] + landmarks_3d[24]) / 2
        landmarks_3d = landmarks_3d - hip_center
        
        # Scale by shoulder width
        shoulder_width = np.linalg.norm(landmarks_3d[11] - landmarks_3d[12])
        if shoulder_width > 0:
            landmarks_3d = landmarks_3d / shoulder_width
        
        return landmarks_3d.flatten()
    
    def augment_data(self, landmarks: np.ndarray, noise_factor: float = 0.01) -> np.ndarray:
        """Add noise to landmarks for data augmentation."""
        if landmarks is None:
            return landmarks
        
        noise = np.random.normal(0, noise_factor, landmarks.shape)
        return landmarks + noise
    
    def extract_features(self, landmarks: np.ndarray) -> Dict[str, float]:
        """Extract meaningful features from pose landmarks."""
        if landmarks is None or len(landmarks) == 0:
            return {}
        
        landmarks_3d = landmarks.reshape(-1, 3)
        
        features = {
            # Body proportions
            'shoulder_width': np.linalg.norm(landmarks_3d[11] - landmarks_3d[12]),
            'hip_width': np.linalg.norm(landmarks_3d[23] - landmarks_3d[24]),
            'torso_height': np.linalg.norm(landmarks_3d[11] - landmarks_3d[23]),
            
            # Arm positions
            'left_arm_angle': self._calculate_arm_angle(landmarks_3d, 'left'),
            'right_arm_angle': self._calculate_arm_angle(landmarks_3d, 'right'),
            'left_arm_raised': landmarks_3d[15][1] < landmarks_3d[11][1],
            'right_arm_raised': landmarks_3d[16][1] < landmarks_3d[12][1],
            
            # Leg positions
            'left_leg_angle': self._calculate_leg_angle(landmarks_3d, 'left'),
            'right_leg_angle': self._calculate_leg_angle(landmarks_3d, 'right'),
            'left_leg_bent': self._is_leg_bent(landmarks_3d, 'left'),
            'right_leg_bent': self._is_leg_bent(landmarks_3d, 'right'),
            
            # Overall pose
            'pose_height': np.linalg.norm(landmarks_3d[0] - landmarks_3d[25]),
            'pose_balance': self._calculate_balance(landmarks_3d)
        }
        
        return features
    
    def _calculate_arm_angle(self, landmarks: np.ndarray, side: str) -> float:
        """Calculate arm angle for given side."""
        if side == 'left':
            shoulder = landmarks[11]
            elbow = landmarks[13]
            wrist = landmarks[15]
        else:
            shoulder = landmarks[12]
            elbow = landmarks[14]
            wrist = landmarks[16]
        
        v1 = elbow - shoulder
        v2 = wrist - elbow
        
        # Check for zero vectors to avoid division by zero
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        
        if norm_v1 == 0 or norm_v2 == 0:
            return 0.0
        
        cos_angle = np.dot(v1, v2) / (norm_v1 * norm_v2)
        return np.arccos(np.clip(cos_angle, -1, 1))
    
    def _calculate_leg_angle(self, landmarks: np.ndarray, side: str) -> float:
        """Calculate leg angle for given side."""
        if side == 'left':
            hip = landmarks[23]
            knee = landmarks[25]
            ankle = landmarks[27]
        else:
            hip = landmarks[24]
            knee = landmarks[26]
            ankle = landmarks[28]
        
        v1 = knee - hip
        v2 = ankle - knee
        
        # Check for zero vectors to avoid division by zero
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        
        if norm_v1 == 0 or norm_v2 == 0:
            return 0.0
        
        cos_angle = np.dot(v1, v2) / (norm_v1 * norm_v2)
        return np.arccos(np.clip(cos_angle, -1, 1))
    
    def _is_leg_bent(self, landmarks: np.ndarray, side: str) -> bool:
        """Check if leg is bent."""
        angle = self._calculate_leg_angle(landmarks, side)
        return angle < np.pi * 0.8  # Less than 144 degrees
    
    def _calculate_balance(self, landmarks: np.ndarray) -> float:
        """Calculate pose balance (0 = perfectly balanced, 1 = unbalanced)."""
        hip_center = (landmarks[23] + landmarks[24]) / 2
        shoulder_center = (landmarks[11] + landmarks[12]) / 2
        
        # Calculate center of mass approximation
        com = (hip_center + shoulder_center) / 2
        
        # Distance from center line
        center_line = (landmarks[23] + landmarks[24]) / 2
        balance = np.linalg.norm(com[:2] - center_line[:2])
        
        return min(balance, 1.0)

class ActivityClassifier:
    """Neural network for activity classification."""
    
    def __init__(self, num_classes: int, input_shape: int = 99):
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.model = self._build_model()
        
    def _build_model(self) -> keras.Model:
        """Build the neural network model."""
        model = keras.Sequential([
            keras.layers.Dense(128, activation='relu', input_shape=(self.input_shape,)),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(self.num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: np.ndarray = None, y_val: np.ndarray = None,
              epochs: int = 100, batch_size: int = 32) -> keras.callbacks.History:
        """Train the activity classifier."""
        callbacks = [
            keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
        ]
        
        validation_data = (X_val, y_val) if X_val is not None else None
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def predict(self, landmarks: np.ndarray) -> Tuple[int, float]:
        """Predict activity class and confidence."""
        if landmarks is None or len(landmarks) == 0:
            return 0, 0.0
        
        # Pad or truncate to input shape
        if len(landmarks) < self.input_shape:
            landmarks = np.pad(landmarks, (0, self.input_shape - len(landmarks)))
        elif len(landmarks) > self.input_shape:
            landmarks = landmarks[:self.input_shape]
        
        prediction = self.model.predict(landmarks.reshape(1, -1), verbose=0)
        class_id = np.argmax(prediction[0])
        confidence = np.max(prediction[0])
        
        return class_id, confidence
    
    def save_model(self, filepath: str):
        """Save the trained model."""
        self.model.save(filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model."""
        self.model = keras.models.load_model(filepath)
        logger.info(f"Model loaded from {filepath}")

class PoseAnalyzer:
    """Advanced pose analysis and visualization."""
    
    def __init__(self, config: PoseConfig):
        self.config = config
        self.processor = PoseDataProcessor(config)
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
    def analyze_video(self, video_path: str, output_path: str = None) -> Dict:
        """Analyze pose in video file."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        analysis_results = {
            'frames_analyzed': 0,
            'poses_detected': 0,
            'activities': [],
            'landmarks_history': [],
            'features_history': []
        }
        
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, 30.0, 
                                (int(cap.get(3)), int(cap.get(4))))
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            landmarks = self.processor.extract_landmarks(frame)
            if landmarks is not None:
                analysis_results['poses_detected'] += 1
                analysis_results['landmarks_history'].append(landmarks)
                
                features = self.processor.extract_features(landmarks)
                analysis_results['features_history'].append(features)
                
                # Draw pose
                if output_path:
                    frame = self._draw_pose_on_frame(frame, landmarks)
            
            analysis_results['frames_analyzed'] += 1
            
            if output_path:
                out.write(frame)
        
        cap.release()
        if output_path:
            out.release()
        
        return analysis_results
    
    def _draw_pose_on_frame(self, frame: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        """Draw pose landmarks on frame."""
        # Convert landmarks back to MediaPipe format
        mp_landmarks = self.mp_pose.PoseLandmark
        pose_landmarks = self.mp_pose.PoseLandmark
        
        # Create MediaPipe landmark list
        landmarks_list = []
        landmarks_3d = landmarks.reshape(-1, 3)
        
        for i, (x, y, z) in enumerate(landmarks_3d):
            landmark = pose_landmarks()
            landmark.x = x
            landmark.y = y
            landmark.z = z
            landmarks_list.append(landmark)
        
        # Draw pose
        self.mp_drawing.draw_landmarks(
            frame,
            landmarks_list,
            self.mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
        )
        
        return frame
    
    def generate_report(self, analysis_results: Dict, output_path: str):
        """Generate analysis report."""
        report = {
            'summary': {
                'total_frames': analysis_results['frames_analyzed'],
                'poses_detected': analysis_results['poses_detected'],
                'detection_rate': analysis_results['poses_detected'] / analysis_results['frames_analyzed']
            },
            'features_analysis': self._analyze_features(analysis_results['features_history']),
            'pose_statistics': self._calculate_pose_statistics(analysis_results['landmarks_history'])
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Analysis report saved to {output_path}")
    
    def _analyze_features(self, features_history: List[Dict]) -> Dict:
        """Analyze feature statistics over time."""
        if not features_history:
            return {}
        
        feature_names = features_history[0].keys()
        analysis = {}
        
        for feature in feature_names:
            values = [f[feature] for f in features_history if feature in f]
            if values:
                analysis[feature] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values)
                }
        
        return analysis
    
    def _calculate_pose_statistics(self, landmarks_history: List[np.ndarray]) -> Dict:
        """Calculate pose statistics."""
        if not landmarks_history:
            return {}
        
        landmarks_array = np.array(landmarks_history)
        
        return {
            'total_landmarks': len(landmarks_history),
            'landmark_variance': np.var(landmarks_array, axis=0).tolist(),
            'movement_magnitude': np.mean(np.linalg.norm(np.diff(landmarks_array, axis=0), axis=1))
        }

def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(description='Advanced Pose Estimation System')
    parser.add_argument('--mode', choices=['analyze', 'train', 'predict'], required=True,
                       help='Operation mode')
    parser.add_argument('--input', required=True, help='Input video or data path')
    parser.add_argument('--output', help='Output path')
    parser.add_argument('--model', help='Model path for training/loading')
    parser.add_argument('--config', help='Configuration file path')
    
    args = parser.parse_args()
    
    # Load configuration
    config = PoseConfig()
    if args.config:
        with open(args.config, 'r') as f:
            config_data = json.load(f)
            for key, value in config_data.items():
                setattr(config, key, value)
    
    if args.mode == 'analyze':
        analyzer = PoseAnalyzer(config)
        results = analyzer.analyze_video(args.input, args.output)
        
        if args.output:
            report_path = args.output.replace('.mp4', '_report.json')
            analyzer.generate_report(results, report_path)
        
        print(f"Analysis complete: {results['poses_detected']} poses detected")
    
    elif args.mode == 'train':
        # Training mode - would need training data
        print("Training mode requires training data. Please implement data loading.")
    
    elif args.mode == 'predict':
        if not args.model:
            print("Model path required for prediction mode")
            return
        
        classifier = ActivityClassifier(num_classes=10)
        classifier.load_model(args.model)
        
        # Load and process input
        processor = PoseDataProcessor(config)
        # Implementation depends on input format
        
        print("Prediction mode - implement based on input format")

if __name__ == "__main__":
    main() 