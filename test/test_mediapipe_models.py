#!/usr/bin/env python3
"""
Test MediaPipe Model Loading and Initialization
Validates that all required MediaPipe models are present and can be loaded successfully.
Updated for gesturebot package (renamed from gesturebot_vision).
"""

import os
import sys
import time
import unittest
from pathlib import Path
from typing import Dict, Any

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_py
from mediapipe.tasks.python import vision as mp_vis


class TestMediaPipeModels(unittest.TestCase):
    """Test MediaPipe model availability and loading."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment."""
        cls.package_dir = Path(__file__).parent.parent
        cls.models_dir = cls.package_dir / 'models'
        cls.test_image = cls.create_test_image()
        
        print(f"Testing MediaPipe models in: {cls.models_dir}")
    
    @classmethod
    def create_test_image(cls) -> np.ndarray:
        """Create a simple test image for model testing."""
        # Create 640x480 test image with some basic shapes
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Add some shapes that might be detected
        cv2.rectangle(img, (100, 100), (200, 200), (255, 0, 0), -1)  # Blue rectangle
        cv2.circle(img, (400, 150), 50, (0, 255, 0), -1)  # Green circle
        cv2.putText(img, 'TEST', (250, 300), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        
        return img
    
    def test_models_directory_exists(self):
        """Test that models directory exists."""
        self.assertTrue(self.models_dir.exists(), 
                       f"Models directory not found: {self.models_dir}")
        self.assertTrue(self.models_dir.is_dir(), 
                       f"Models path is not a directory: {self.models_dir}")
    
    def test_efficientdet_model_exists(self):
        """Test that EfficientDet model file exists."""
        model_path = self.models_dir / 'efficientdet.tflite'
        self.assertTrue(model_path.exists(), 
                       f"EfficientDet model not found: {model_path}")
        
        # Check file size (should be > 1MB for a valid model)
        file_size = model_path.stat().st_size
        self.assertGreater(file_size, 1024 * 1024, 
                          f"EfficientDet model file too small: {file_size} bytes")
    
    def test_efficientdet_model_loading(self):
        """Test EfficientDet model can be loaded and initialized."""
        model_path = self.models_dir / 'efficientdet.tflite'
        
        if not model_path.exists():
            self.skipTest(f"EfficientDet model not found: {model_path}")
        
        try:
            # Initialize object detector
            base_options = mp_py.BaseOptions(model_asset_path=str(model_path))
            options = mp_vis.ObjectDetectorOptions(
                base_options=base_options,
                max_results=5,
                score_threshold=0.3,
                running_mode=mp_vis.RunningMode.IMAGE
            )
            
            detector = mp_vis.ObjectDetector.create_from_options(options)
            self.assertIsNotNone(detector, "Failed to create object detector")
            
            # Test detection on test image
            rgb_image = cv2.cvtColor(self.test_image, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
            
            result = detector.detect(mp_image)
            self.assertIsNotNone(result, "Detection result is None")
            
            # Clean up
            detector.close()
            
        except Exception as e:
            self.fail(f"Failed to load EfficientDet model: {e}")
    
    def test_hand_landmarker_model_exists(self):
        """Test that Hand Landmarker model file exists."""
        model_path = self.models_dir / 'hand_landmarker.task'
        
        if not model_path.exists():
            self.skipTest(f"Hand Landmarker model not found: {model_path}")
        
        # Check file size
        file_size = model_path.stat().st_size
        self.assertGreater(file_size, 1024 * 1024, 
                          f"Hand Landmarker model file too small: {file_size} bytes")
    
    def test_hand_landmarker_model_loading(self):
        """Test Hand Landmarker model can be loaded."""
        model_path = self.models_dir / 'hand_landmarker.task'
        
        if not model_path.exists():
            self.skipTest(f"Hand Landmarker model not found: {model_path}")
        
        try:
            # Initialize hand landmarker
            base_options = mp_py.BaseOptions(model_asset_path=str(model_path))
            options = mp_vis.HandLandmarkerOptions(
                base_options=base_options,
                running_mode=mp_vis.RunningMode.IMAGE,
                num_hands=2
            )
            
            landmarker = mp_vis.HandLandmarker.create_from_options(options)
            self.assertIsNotNone(landmarker, "Failed to create hand landmarker")
            
            # Test on test image
            rgb_image = cv2.cvtColor(self.test_image, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
            
            result = landmarker.detect(mp_image)
            self.assertIsNotNone(result, "Hand landmark result is None")
            
            # Clean up
            landmarker.close()
            
        except Exception as e:
            self.fail(f"Failed to load Hand Landmarker model: {e}")
    
    def test_gesture_recognizer_model_exists(self):
        """Test that Gesture Recognizer model file exists."""
        model_path = self.models_dir / 'gesture_recognizer.task'
        
        if not model_path.exists():
            self.skipTest(f"Gesture Recognizer model not found: {model_path}")
        
        # Check file size
        file_size = model_path.stat().st_size
        self.assertGreater(file_size, 1024 * 1024, 
                          f"Gesture Recognizer model file too small: {file_size} bytes")
    
    def test_gesture_recognizer_model_loading(self):
        """Test Gesture Recognizer model can be loaded."""
        model_path = self.models_dir / 'gesture_recognizer.task'
        
        if not model_path.exists():
            self.skipTest(f"Gesture Recognizer model not found: {model_path}")
        
        try:
            # Initialize gesture recognizer
            base_options = mp_py.BaseOptions(model_asset_path=str(model_path))
            options = mp_vis.GestureRecognizerOptions(
                base_options=base_options,
                running_mode=mp_vis.RunningMode.IMAGE,
                num_hands=2
            )
            
            recognizer = mp_vis.GestureRecognizer.create_from_options(options)
            self.assertIsNotNone(recognizer, "Failed to create gesture recognizer")
            
            # Test on test image
            rgb_image = cv2.cvtColor(self.test_image, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
            
            result = recognizer.recognize(mp_image)
            self.assertIsNotNone(result, "Gesture recognition result is None")
            
            # Clean up
            recognizer.close()
            
        except Exception as e:
            self.fail(f"Failed to load Gesture Recognizer model: {e}")
    
    def test_mediapipe_import(self):
        """Test that MediaPipe can be imported successfully."""
        try:
            import mediapipe as mp
            from mediapipe.tasks import python as mp_py
            from mediapipe.tasks.python import vision as mp_vis
            
            # Check version
            self.assertIsNotNone(mp.__version__, "MediaPipe version not available")
            print(f"MediaPipe version: {mp.__version__}")
            
        except ImportError as e:
            self.fail(f"Failed to import MediaPipe: {e}")
    
    def test_model_performance_benchmark(self):
        """Benchmark model loading and inference performance."""
        model_path = self.models_dir / 'efficientdet.tflite'
        
        if not model_path.exists():
            self.skipTest(f"EfficientDet model not found: {model_path}")
        
        try:
            # Measure model loading time
            start_time = time.time()
            
            base_options = mp_py.BaseOptions(model_asset_path=str(model_path))
            options = mp_vis.ObjectDetectorOptions(
                base_options=base_options,
                max_results=5,
                score_threshold=0.3,
                running_mode=mp_vis.RunningMode.IMAGE
            )
            
            detector = mp_vis.ObjectDetector.create_from_options(options)
            loading_time = (time.time() - start_time) * 1000  # ms
            
            # Measure inference time
            rgb_image = cv2.cvtColor(self.test_image, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
            
            # Warm up
            detector.detect(mp_image)
            
            # Benchmark inference
            inference_times = []
            for _ in range(10):
                start_time = time.time()
                result = detector.detect(mp_image)
                inference_time = (time.time() - start_time) * 1000  # ms
                inference_times.append(inference_time)
            
            avg_inference_time = sum(inference_times) / len(inference_times)
            
            # Performance assertions
            self.assertLess(loading_time, 5000, f"Model loading too slow: {loading_time:.1f}ms")
            self.assertLess(avg_inference_time, 500, f"Inference too slow: {avg_inference_time:.1f}ms")
            
            print(f"Model loading time: {loading_time:.1f}ms")
            print(f"Average inference time: {avg_inference_time:.1f}ms")
            
            # Clean up
            detector.close()
            
        except Exception as e:
            self.fail(f"Performance benchmark failed: {e}")


def main():
    """Run MediaPipe model tests."""
    unittest.main(verbosity=2)


if __name__ == '__main__':
    main()
