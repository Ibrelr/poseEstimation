#!/usr/bin/env python3
"""
Pose Estimation Model Training Script
=====================================

This script demonstrates how to train a custom pose estimation model
for specific activities like yoga poses, exercise movements, etc.

Usage:
    python train_model.py --data_path /path/to/data --output_model model.h5
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
import logging

# Import our custom classes
from pose_estimation_advanced import PoseDataProcessor, ActivityClassifier, PoseConfig

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PoseModelTrainer:
    """Handles training of pose estimation models."""
    
    def __init__(self, config: PoseConfig):
        self.config = config
        self.processor = PoseDataProcessor(config)
        self.label_encoder = LabelEncoder()
        
    def load_training_data(self, data_path: str) -> tuple:
        """Load and preprocess training data."""
        logger.info(f"Loading training data from {data_path}")
        
        # This is a simplified example - in practice you'd load from:
        # - COCO dataset
        # - Custom annotated videos
        # - Synthetic data
        
        # For demonstration, we'll create synthetic data
        X, y = self._generate_synthetic_data()
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        logger.info(f"Loaded {len(X)} samples with {len(np.unique(y))} classes")
        logger.info(f"Classes: {list(self.label_encoder.classes_)}")
        
        return X, y_encoded
    
    def _generate_synthetic_data(self, num_samples: int = 1000) -> tuple:
        """Generate synthetic pose data for demonstration."""
        activities = ['standing', 'sitting', 'walking', 'yoga_pose', 'exercise']
        X = []
        y = []
        
        for activity in activities:
            samples_per_class = num_samples // len(activities)
            
            for _ in range(samples_per_class):
                # Generate synthetic landmarks based on activity
                landmarks = self._generate_activity_landmarks(activity)
                X.append(landmarks)
                y.append(activity)
        
        return np.array(X), np.array(y)
    
    def _generate_activity_landmarks(self, activity: str) -> np.ndarray:
        """Generate synthetic landmarks for a specific activity."""
        # Base landmarks (33 keypoints * 3 coordinates = 99 values)
        landmarks = np.random.normal(0, 0.1, 99)
        
        # Modify based on activity
        if activity == 'standing':
            # Standing pose: straight posture
            landmarks[11*3+1] = 0.3  # Left shoulder
            landmarks[12*3+1] = 0.3  # Right shoulder
            landmarks[23*3+1] = 0.6  # Left hip
            landmarks[24*3+1] = 0.6  # Right hip
            
        elif activity == 'sitting':
            # Sitting pose: lower hip position
            landmarks[23*3+1] = 0.8  # Left hip
            landmarks[24*3+1] = 0.8  # Right hip
            landmarks[25*3+1] = 0.9  # Left knee
            landmarks[26*3+1] = 0.9  # Right knee
            
        elif activity == 'walking':
            # Walking pose: asymmetric leg positions
            landmarks[25*3+1] = 0.7  # Left knee
            landmarks[26*3+1] = 0.6  # Right knee
            landmarks[27*3+1] = 0.8  # Left ankle
            landmarks[28*3+1] = 0.7  # Right ankle
            
        elif activity == 'yoga_pose':
            # Yoga pose: raised arms
            landmarks[15*3+1] = 0.1  # Left wrist
            landmarks[16*3+1] = 0.1  # Right wrist
            landmarks[13*3+1] = 0.2  # Left elbow
            landmarks[14*3+1] = 0.2  # Right elbow
            
        elif activity == 'exercise':
            # Exercise pose: bent knees and arms
            landmarks[25*3+1] = 0.8  # Left knee
            landmarks[26*3+1] = 0.8  # Right knee
            landmarks[13*3+1] = 0.4  # Left elbow
            landmarks[14*3+1] = 0.4  # Right elbow
        
        return landmarks
    
    def preprocess_data(self, X: np.ndarray) -> np.ndarray:
        """Preprocess the training data."""
        logger.info("Preprocessing training data...")
        
        # Normalize landmarks
        X_normalized = []
        for landmarks in X:
            normalized = self.processor.normalize_landmarks(landmarks)
            X_normalized.append(normalized)
        
        X_normalized = np.array(X_normalized)
        
        # Handle NaN values
        X_normalized = np.nan_to_num(X_normalized, nan=0.0)
        
        logger.info(f"Preprocessed data shape: {X_normalized.shape}")
        return X_normalized
    
    def train_model(self, X: np.ndarray, y: np.ndarray, 
                   validation_split: float = 0.2,
                   epochs: int = 100,
                   batch_size: int = 32) -> tuple:
        """Train the activity classification model."""
        logger.info("Starting model training...")
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42, stratify=y
        )
        
        # Create and train model
        num_classes = len(np.unique(y))
        classifier = ActivityClassifier(num_classes=num_classes, input_shape=X.shape[1])
        
        history = classifier.train(
            X_train, y_train,
            X_val, y_val,
            epochs=epochs,
            batch_size=batch_size
        )
        
        # Evaluate model
        val_loss, val_accuracy = classifier.model.evaluate(X_val, y_val, verbose=0)
        logger.info(f"Validation accuracy: {val_accuracy:.4f}")
        
        return classifier, history, (X_val, y_val)
    
    def save_model(self, classifier: ActivityClassifier, model_path: str):
        """Save the trained model and metadata."""
        # Save model
        classifier.save_model(model_path)
        
        # Save metadata
        metadata = {
            'label_encoder_classes': self.label_encoder.classes_.tolist(),
            'input_shape': classifier.input_shape,
            'num_classes': classifier.num_classes,
            'config': {
                'model_complexity': self.config.model_complexity,
                'smooth_landmarks': self.config.smooth_landmarks,
                'min_detection_confidence': self.config.min_detection_confidence,
                'min_tracking_confidence': self.config.min_tracking_confidence
            }
        }
        
        metadata_path = model_path.replace('.h5', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Model and metadata saved to {model_path} and {metadata_path}")
    
    def plot_training_history(self, history, output_path: str = None):
        """Plot training history."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot accuracy
        ax1.plot(history.history['accuracy'], label='Training Accuracy')
        ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # Plot loss
        ax2.plot(history.history['loss'], label='Training Loss')
        ax2.plot(history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Training history plot saved to {output_path}")
        
        plt.show()
    
    def evaluate_model(self, classifier: ActivityClassifier, X_val: np.ndarray, y_val: np.ndarray):
        """Evaluate model performance."""
        from sklearn.metrics import classification_report, confusion_matrix
        
        # Make predictions
        y_pred = []
        for landmarks in X_val:
            class_id, confidence = classifier.predict(landmarks)
            y_pred.append(class_id)
        
        y_pred = np.array(y_pred)
        
        # Print classification report
        print("\nClassification Report:")
        print(classification_report(y_val, y_pred, 
                                  target_names=self.label_encoder.classes_))
        
        # Plot confusion matrix
        cm = confusion_matrix(y_val, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.label_encoder.classes_,
                   yticklabels=self.label_encoder.classes_)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.show()

def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train Pose Estimation Model')
    parser.add_argument('--data_path', type=str, help='Path to training data')
    parser.add_argument('--output_model', type=str, default='pose_model.h5',
                       help='Output model path')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--validation_split', type=float, default=0.2,
                       help='Validation split ratio')
    
    args = parser.parse_args()
    
    # Initialize configuration
    config = PoseConfig()
    
    # Initialize trainer
    trainer = PoseModelTrainer(config)
    
    # Load data
    X, y = trainer.load_training_data(args.data_path or 'synthetic')
    
    # Preprocess data
    X_processed = trainer.preprocess_data(X)
    
    # Train model
    classifier, history, (X_val, y_val) = trainer.train_model(
        X_processed, y,
        validation_split=args.validation_split,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
    
    # Save model
    trainer.save_model(classifier, args.output_model)
    
    # Plot training history
    trainer.plot_training_history(history, args.output_model.replace('.h5', '_history.png'))
    
    # Evaluate model
    trainer.evaluate_model(classifier, X_val, y_val)
    
    logger.info("Training completed successfully!")

if __name__ == "__main__":
    main() 