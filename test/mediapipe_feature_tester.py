#!/usr/bin/env python3
"""
Comprehensive MediaPipe Feature Testing Framework for GestureBot
Tests all MediaPipe capabilities on Pi 5 hardware and generates performance reports.
"""

import os
import sys
import time
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict

import cv2
import numpy as np
import mediapipe as mp


@dataclass
class FeatureTestResult:
    """Test result for a single MediaPipe feature."""
    feature_name: str
    success: bool
    avg_processing_time: float  # milliseconds
    max_processing_time: float
    min_processing_time: float
    fps_capability: float
    memory_usage: float  # MB
    error_message: str = ""
    model_size: float = 0.0  # MB
    accuracy_score: float = 0.0  # If applicable


class MediaPipeFeatureTester:
    """Comprehensive testing framework for all MediaPipe features."""
    
    def __init__(self):
        self.script_dir = Path(__file__).parent
        self.package_dir = self.script_dir.parent
        self.models_dir = self.package_dir / 'models'
        self.results_dir = self.script_dir / 'results'
        
        # Create directories
        self.models_dir.mkdir(exist_ok=True)
        self.results_dir.mkdir(exist_ok=True)
        
        # Test configuration
        self.test_duration = 30  # seconds per test
        self.test_frames = 100   # number of frames to process
        self.image_width = 640
        self.image_height = 480
        
        # Results storage
        self.test_results: List[FeatureTestResult] = []
        
        print("=== MediaPipe Feature Testing Framework ===")
        print(f"Test duration: {self.test_duration} seconds per feature")
        print(f"Test frames: {self.test_frames} frames per feature")
        print(f"Resolution: {self.image_width}x{self.image_height}")
    
    def capture_test_image(self) -> Optional[np.ndarray]:
        """Capture a test image using rpicam-still."""
        try:
            temp_file = '/tmp/test_image.jpg'
            
            cmd = [
                'rpicam-still',
                '--output', temp_file,
                '--width', str(self.image_width),
                '--height', str(self.image_height),
                '--timeout', '1000',
                '--nopreview'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            if result.returncode != 0:
                print(f"❌ Failed to capture test image: {result.stderr}")
                return None
            
            image = cv2.imread(temp_file)
            if image is None:
                print("❌ Failed to load captured image")
                return None
            
            os.unlink(temp_file)
            return image
            
        except Exception as e:
            print(f"❌ Error capturing test image: {e}")
            return None
    
    def test_object_detection(self) -> FeatureTestResult:
        """Test MediaPipe object detection."""
        print("\n🔍 Testing Object Detection...")
        
        try:
            from mediapipe.tasks import python as mp_py
            from mediapipe.tasks.python import vision as mp_vis
            
            # Check if model exists
            model_path = self.models_dir / 'efficientdet.tflite'
            if not model_path.exists():
                # Try alternative locations
                alt_paths = [
                    Path.home() / 'GestureBot' / 'mediapipe-test' / 'efficientdet.tflite',
                    Path('/opt/ros/humble/share/gesturebot_vision/models/efficientdet.tflite')
                ]
                
                for alt_path in alt_paths:
                    if alt_path.exists():
                        model_path = alt_path
                        break
                else:
                    return FeatureTestResult(
                        feature_name="object_detection",
                        success=False,
                        avg_processing_time=0,
                        max_processing_time=0,
                        min_processing_time=0,
                        fps_capability=0,
                        memory_usage=0,
                        error_message="EfficientDet model not found"
                    )
            
            # Initialize detector
            base_options = mp_py.BaseOptions(model_asset_path=str(model_path))
            options = mp_vis.ObjectDetectorOptions(
                base_options=base_options,
                max_results=5,
                score_threshold=0.3,
                running_mode=mp_vis.RunningMode.IMAGE
            )
            
            detector = mp_vis.ObjectDetector.create_from_options(options)
            
            # Capture test image
            test_image = self.capture_test_image()
            if test_image is None:
                return FeatureTestResult(
                    feature_name="object_detection",
                    success=False,
                    avg_processing_time=0,
                    max_processing_time=0,
                    min_processing_time=0,
                    fps_capability=0,
                    memory_usage=0,
                    error_message="Failed to capture test image"
                )
            
            # Run performance test
            processing_times = []
            detections_count = 0
            
            print(f"Processing {self.test_frames} frames...")
            
            for i in range(self.test_frames):
                start_time = time.time()
                
                # Convert to RGB
                rgb_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
                
                # Run detection
                result = detector.detect(mp_image)
                
                processing_time = (time.time() - start_time) * 1000
                processing_times.append(processing_time)
                
                if result.detections:
                    detections_count += len(result.detections)
                
                if (i + 1) % 20 == 0:
                    print(f"  Processed {i + 1}/{self.test_frames} frames")
            
            # Calculate statistics
            avg_time = np.mean(processing_times)
            max_time = np.max(processing_times)
            min_time = np.min(processing_times)
            fps_capability = 1000 / avg_time if avg_time > 0 else 0
            
            print(f"✅ Object Detection Test Complete")
            print(f"   Average processing time: {avg_time:.1f}ms")
            print(f"   FPS capability: {fps_capability:.1f}")
            print(f"   Total detections: {detections_count}")
            
            return FeatureTestResult(
                feature_name="object_detection",
                success=True,
                avg_processing_time=avg_time,
                max_processing_time=max_time,
                min_processing_time=min_time,
                fps_capability=fps_capability,
                memory_usage=self.get_memory_usage(),
                model_size=model_path.stat().st_size / 1024 / 1024,
                accuracy_score=detections_count / self.test_frames
            )
            
        except Exception as e:
            return FeatureTestResult(
                feature_name="object_detection",
                success=False,
                avg_processing_time=0,
                max_processing_time=0,
                min_processing_time=0,
                fps_capability=0,
                memory_usage=0,
                error_message=str(e)
            )
    
    def test_hand_landmarks(self) -> FeatureTestResult:
        """Test MediaPipe hand landmark detection."""
        print("\n✋ Testing Hand Landmark Detection...")
        
        try:
            # Initialize MediaPipe hands
            mp_hands = mp.solutions.hands
            hands = mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            
            # Capture test image
            test_image = self.capture_test_image()
            if test_image is None:
                return FeatureTestResult(
                    feature_name="hand_landmarks",
                    success=False,
                    avg_processing_time=0,
                    max_processing_time=0,
                    min_processing_time=0,
                    fps_capability=0,
                    memory_usage=0,
                    error_message="Failed to capture test image"
                )
            
            # Run performance test
            processing_times = []
            hands_detected = 0
            
            print(f"Processing {self.test_frames} frames...")
            
            for i in range(self.test_frames):
                start_time = time.time()
                
                # Convert to RGB
                rgb_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
                
                # Process image
                results = hands.process(rgb_image)
                
                processing_time = (time.time() - start_time) * 1000
                processing_times.append(processing_time)
                
                if results.multi_hand_landmarks:
                    hands_detected += len(results.multi_hand_landmarks)
                
                if (i + 1) % 20 == 0:
                    print(f"  Processed {i + 1}/{self.test_frames} frames")
            
            # Calculate statistics
            avg_time = np.mean(processing_times)
            max_time = np.max(processing_times)
            min_time = np.min(processing_times)
            fps_capability = 1000 / avg_time if avg_time > 0 else 0
            
            print(f"✅ Hand Landmarks Test Complete")
            print(f"   Average processing time: {avg_time:.1f}ms")
            print(f"   FPS capability: {fps_capability:.1f}")
            print(f"   Hands detected: {hands_detected}")
            
            return FeatureTestResult(
                feature_name="hand_landmarks",
                success=True,
                avg_processing_time=avg_time,
                max_processing_time=max_time,
                min_processing_time=min_time,
                fps_capability=fps_capability,
                memory_usage=self.get_memory_usage(),
                accuracy_score=hands_detected / self.test_frames
            )
            
        except Exception as e:
            return FeatureTestResult(
                feature_name="hand_landmarks",
                success=False,
                avg_processing_time=0,
                max_processing_time=0,
                min_processing_time=0,
                fps_capability=0,
                memory_usage=0,
                error_message=str(e)
            )
    
    def get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except:
            return 0.0
    
    def run_all_tests(self) -> None:
        """Run all MediaPipe feature tests."""
        print("Starting comprehensive MediaPipe feature testing...")
        
        # Test each feature
        test_functions = [
            self.test_object_detection,
            self.test_hand_landmarks,
            # Add more test functions as they're implemented
        ]
        
        for test_func in test_functions:
            try:
                result = test_func()
                self.test_results.append(result)
            except Exception as e:
                print(f"❌ Test failed: {e}")
        
        # Generate report
        self.generate_report()
    
    def generate_report(self) -> None:
        """Generate comprehensive test report."""
        print("\n" + "="*60)
        print("MEDIAPIPE FEATURE TEST REPORT")
        print("="*60)
        
        successful_tests = [r for r in self.test_results if r.success]
        failed_tests = [r for r in self.test_results if not r.success]
        
        print(f"Total tests: {len(self.test_results)}")
        print(f"Successful: {len(successful_tests)}")
        print(f"Failed: {len(failed_tests)}")
        
        if successful_tests:
            print("\n✅ SUCCESSFUL TESTS:")
            for result in successful_tests:
                print(f"  {result.feature_name}:")
                print(f"    Processing time: {result.avg_processing_time:.1f}ms")
                print(f"    FPS capability: {result.fps_capability:.1f}")
                print(f"    Memory usage: {result.memory_usage:.1f}MB")
        
        if failed_tests:
            print("\n❌ FAILED TESTS:")
            for result in failed_tests:
                print(f"  {result.feature_name}: {result.error_message}")
        
        # Save detailed report
        report_file = self.results_dir / f'test_report_{int(time.time())}.json'
        with open(report_file, 'w') as f:
            json.dump([asdict(r) for r in self.test_results], f, indent=2)
        
        print(f"\nDetailed report saved to: {report_file}")


def main():
    """Main function for feature testing."""
    tester = MediaPipeFeatureTester()
    tester.run_all_tests()


if __name__ == '__main__':
    main()
