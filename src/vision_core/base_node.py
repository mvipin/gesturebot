#!/usr/bin/env python3
"""
MediaPipe Base Node for GestureBot Vision System
Provides shared functionality for all MediaPipe feature nodes.
"""

import time
import threading
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass
from collections import deque
from datetime import datetime

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from gesturebot.msg import VisionPerformance


class BufferedLogger:
    """
    Configurable logging buffer for high-performance diagnostic logging.

    Three operating modes:
    1. Disabled (enabled=False): No buffering, direct logging of critical errors only
    2. Circular (enabled=True, unlimited_mode=False): Circular buffer with auto-drop when full
    3. Unlimited (enabled=True, unlimited_mode=True): Unlimited buffer with timer-only flushing

    The mode names reflect behavior rather than use case - either mode can be used in
    production or debug scenarios depending on requirements.
    """

    def __init__(self, buffer_size: int = 200, logger=None, unlimited_mode: bool = False, enabled: bool = True):
        self.buffer_size = buffer_size
        self.logger = logger
        self.unlimited_mode = unlimited_mode
        self.enabled = enabled
        self.lock = threading.Lock()
        self.entry_count = 0

        # Configure buffer based on mode
        if not self.enabled:
            # Disabled mode: No buffer at all
            self.buffer = None
        elif self.unlimited_mode:
            # Unlimited mode: Unlimited buffer, timer-only flushing
            self.buffer = deque()  # No maxlen for unlimited growth
        else:
            # Circular mode: Circular buffer with auto-drop when full
            self.buffer = deque(maxlen=self.buffer_size)

    def log_event(self, event_type: str, message: str, **kwargs):
        """Add an event to the buffer with timestamp and optional metadata."""
        # Always count events for statistics
        with self.lock:
            self.entry_count += 1

        # If disabled, only log critical errors directly to main logger
        if not self.enabled:
            if event_type in ['PUBLISH_ERROR', 'CRITICAL_ERROR', 'INITIALIZATION_ERROR']:
                if self.logger:
                    self.logger.error(f"{event_type}: {message}")
            return

        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]  # millisecond precision

        entry = {
            'timestamp': timestamp,
            'event_type': event_type,
            'message': message,
            'metadata': kwargs
        }

        with self.lock:
            self.buffer.append(entry)
            # Note: In circular mode, the circular buffer (maxlen) automatically drops old entries
            # In unlimited mode, buffer grows unlimited and only flushes via timer

    def _flush_buffer(self):
        """Internal method to dump buffer contents (assumes lock is held)."""
        if not self.enabled or not self.buffer:
            return

        if self.logger:
            mode_str = "UNLIMITED" if self.unlimited_mode else "CIRCULAR"
            self.logger.info(f"=== BUFFERED LOG DUMP [{mode_str}] ({len(self.buffer)} entries) ===")
            for entry in self.buffer:
                metadata_str = ""
                if entry['metadata']:
                    metadata_str = " | " + " | ".join([f"{k}={v}" for k, v in entry['metadata'].items()])

                self.logger.info(f"[{entry['timestamp']}] {entry['event_type']}: {entry['message']}{metadata_str}")
            self.logger.info(f"=== END BUFFER DUMP (Total processed: {self.entry_count}) ===")

        self.buffer.clear()

    def flush(self):
        """Manually flush the current buffer contents."""
        if not self.enabled:
            if self.logger:
                self.logger.info("BufferedLogger is disabled - no buffer to flush")
            return

        with self.lock:
            self._flush_buffer()

    def get_stats(self):
        """Get buffer statistics."""
        with self.lock:
            if not self.enabled:
                return {
                    'enabled': False,
                    'mode': 'disabled',
                    'current_size': 0,
                    'max_size': 0,
                    'total_entries': self.entry_count
                }

            return {
                'enabled': True,
                'mode': 'unlimited' if self.unlimited_mode else 'circular',
                'current_size': len(self.buffer) if self.buffer else 0,
                'max_size': self.buffer_size if not self.unlimited_mode else 'unlimited',
                'total_entries': self.entry_count,
                'flush_strategy': 'timer_only' if self.unlimited_mode else 'circular_auto_drop'
            }


@dataclass
class ProcessingConfig:
    """Configuration for MediaPipe processing."""
    enabled: bool = True
    max_fps: float = 15.0
    frame_skip: int = 1
    confidence_threshold: float = 0.5
    max_results: int = 5
    priority: int = 1  # 0


class PipelineTimer:
    """High-precision timing tracker for the three main vision pipeline stages."""

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all timing measurements."""
        self.frame_start_time = 0.0
        self.preprocessing_start_time = 0.0
        self.mediapipe_start_time = 0.0
        self.postprocessing_start_time = 0.0
        self.frame_end_time = 0.0

        # Stage durations (in milliseconds)
        self.preprocessing_duration = 0.0
        self.mediapipe_duration = 0.0
        self.postprocessing_duration = 0.0
        self.total_duration = 0.0

    def start_frame(self):
        """Mark the start of frame processing (beginning of preprocessing)."""
        self.frame_start_time = time.perf_counter()
        self.preprocessing_start_time = self.frame_start_time

    def mark_mediapipe_start(self):
        """Mark the transition to MediaPipe processing."""
        current_time = time.perf_counter()
        self.preprocessing_duration = (current_time - self.preprocessing_start_time) * 1000
        self.mediapipe_start_time = current_time

    def mark_postprocessing_start(self):
        """Mark the start of post-processing (MediaPipe callback received)."""
        current_time = time.perf_counter()
        self.mediapipe_duration = (current_time - self.mediapipe_start_time) * 1000
        self.postprocessing_start_time = current_time

    def end_frame(self):
        """Mark the end of frame processing."""
        self.frame_end_time = time.perf_counter()
        self.postprocessing_duration = (self.frame_end_time - self.postprocessing_start_time) * 1000
        self.total_duration = (self.frame_end_time - self.frame_start_time) * 1000

    def get_summary(self) -> Dict[str, float]:
        """Get timing summary for this frame."""
        return {
            'preprocessing_ms': self.preprocessing_duration,
            'mediapipe_ms': self.mediapipe_duration,
            'postprocessing_ms': self.postprocessing_duration,
            'total_ms': self.total_duration
        }


@dataclass
class PerformanceStats:
    """Performance statistics for pipeline timing analysis."""
    # Frame counters
    frames_processed: int = 0
    frames_processed_period: int = 0  # Frames processed in current 5-second period

    # Overall timing
    current_fps: float = 0.0

    # Pipeline stage timing (in milliseconds, averaged over measurement period)
    avg_preprocessing_time: float = 0.0      # ROS message → MediaPipe submission
    avg_mediapipe_time: float = 0.0          # MediaPipe processing duration
    avg_postprocessing_time: float = 0.0     # MediaPipe callback → ROS publishing
    avg_total_pipeline_time: float = 0.0     # Sum of all pipeline stages

    # Timing metadata
    last_update: float = 0.0
    period_start_time: float = 0.0


class MediaPipeBaseNode(Node, ABC):
    """
    Abstract base class for all MediaPipe feature nodes.
    Provides common functionality for camera input, processing, and output.
    Includes configurable buffered logging for diagnostics.
    """

    def __init__(self, node_name: str, feature_name: str, config: ProcessingConfig,
                 enable_buffered_logging: bool = True, unlimited_buffer_mode: bool = False,
                 enable_performance_tracking: bool = False):
        super().__init__(node_name)

        self.feature_name = feature_name
        self.config = config

        # Initialize buffered logging
        self.buffered_logger = BufferedLogger(
            buffer_size=200,
            logger=self.get_logger(),
            unlimited_mode=unlimited_buffer_mode,
            enabled=enable_buffered_logging
        )

        # Performance tracking control
        self.enable_performance_tracking = enable_performance_tracking

        # Performance tracking (only if enabled)
        if self.enable_performance_tracking:
            self.stats = PerformanceStats()
            self.stats.period_start_time = time.perf_counter()
            self.pipeline_timer = PipelineTimer()
            self.timing_history = []  # Store recent timing measurements
            self.max_timing_history = 20  # Keep last 20 frame timings for 5-second averages
        else:
            self.stats = None
            self.pipeline_timer = None
            self.timing_history = []

        # Legacy performance tracking for compatibility
        self.processing_times = []
        self.frame_counter = 0
        self.last_fps_time = time.time()
        
        # Threading and synchronization
        self.processing_lock = threading.Lock()
        self.latest_results = None
        
        # ROS 2 components
        self.cv_bridge = CvBridge()
        
        # QoS profiles for different data types
        self.image_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        self.result_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        # Subscribers
        self.image_subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            self.image_qos
        )
        
        # Publishers
        self.performance_publisher = self.create_publisher(
            VisionPerformance,
            '/vision/performance',
            self.result_qos
        )
        
        # Initialize MediaPipe components
        self.initialize_mediapipe()

        # Performance monitoring timer
        self.performance_timer = self.create_timer(5.0, self.publish_performance_stats)

        # Buffer flush timer (10 seconds for all modes)
        self.buffer_flush_timer = self.create_timer(10.0, self._flush_buffered_logger)

        # Log buffered logger initialization
        buffer_stats = self.buffered_logger.get_stats()
        self.get_logger().info(f'BufferedLogger initialized: {buffer_stats}')

        self.get_logger().info(f'{self.feature_name} node initialized')
    
    @abstractmethod
    def initialize_mediapipe(self) -> bool:
        """Initialize MediaPipe components. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def process_frame(self, frame: np.ndarray, timestamp: float) -> Any:
        """Process a single frame. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def publish_results(self, results: Any, timestamp: float) -> None:
        """Publish processing results. Must be implemented by subclasses."""
        pass
    
    def image_callback(self, msg: Image) -> None:
        """Handle incoming camera frames with pipeline timing."""
        if not self.config.enabled:
            return

        # Start timing for this frame (only if performance tracking enabled)
        if self.enable_performance_tracking and self.pipeline_timer:
            self.pipeline_timer.start_frame()

        # Frame skipping for performance
        self.frame_counter += 1
        if self.frame_counter % (self.config.frame_skip + 1) != 0:
            return

        try:
            # Convert ROS image to OpenCV (preprocessing stage includes this conversion)
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, 'bgr8')
            timestamp = time.time()

            # Process frame asynchronously to avoid blocking
            # Note: This creates a reference copy for threading safety (Copy #3)
            threading.Thread(
                target=self._process_frame_async,
                args=(cv_image, timestamp),
                daemon=True
            ).start()

        except Exception as e:
            self.get_logger().error(f'[{self.feature_name}] Error in image callback: {e}')
    
    def _process_frame_async(self, frame: np.ndarray, timestamp: float) -> None:
        """Process frame in separate thread with detailed timing."""
        if not self.processing_lock.acquire(blocking=False):
            # Skip frame if still processing previous one
            self.stats.frames_skipped += 1
            return

        try:
            start_time = time.perf_counter()

            # Enhanced timing (only if performance tracking enabled)
            if self.enable_performance_tracking and self.pipeline_timer:
                self.pipeline_timer.mark_mediapipe_start()

            # Call subclass implementation (includes MediaPipe processing)
            results = self.process_frame(frame, timestamp)

            # Mark post-processing start (only if performance tracking enabled)
            if self.enable_performance_tracking and self.pipeline_timer:
                self.pipeline_timer.mark_postprocessing_start()

            # Publish results
            if results is not None:
                self.publish_results(results, timestamp)
                self.latest_results = results

                # Update frame counters (only if performance tracking enabled)
                if self.enable_performance_tracking and self.stats:
                    self.stats.frames_processed += 1

            # End frame timing and update statistics (only if performance tracking enabled)
            if self.enable_performance_tracking and self.pipeline_timer:
                self.pipeline_timer.end_frame()
                timing_summary = self.pipeline_timer.get_summary()
                self._update_enhanced_performance_stats(timing_summary)

            # Track legacy performance for compatibility
            processing_time = (time.perf_counter() - start_time) * 1000  # ms
            self._update_performance_stats(processing_time)

        except Exception as e:
            self.get_logger().error(f'[{self.feature_name}] Error processing frame: {e}')
        finally:
            self.processing_lock.release()
            # Reset timer for next frame (only if performance tracking enabled)
            if self.enable_performance_tracking and self.pipeline_timer:
                self.pipeline_timer.reset()
    
    def _update_enhanced_performance_stats(self, timing_summary: Dict[str, float]) -> None:
        """Update enhanced performance statistics with pipeline timing."""
        if not self.enable_performance_tracking or self.stats is None:
            return

        # Store timing history for rolling averages
        self.timing_history.append(timing_summary)
        if len(self.timing_history) > self.max_timing_history:
            self.timing_history.pop(0)

        # Update period frame count
        self.stats.frames_processed_period += 1

        # Calculate running averages for pipeline stages (last 20 frames)
        if self.timing_history:
            recent_timings = self.timing_history

            self.stats.avg_preprocessing_time = np.mean([t['preprocessing_ms'] for t in recent_timings])
            self.stats.avg_mediapipe_time = np.mean([t['mediapipe_ms'] for t in recent_timings])
            self.stats.avg_postprocessing_time = np.mean([t['postprocessing_ms'] for t in recent_timings])
            self.stats.avg_total_pipeline_time = np.mean([t['total_ms'] for t in recent_timings])

            # Calculate effective FPS based on recent processing
            current_time = time.perf_counter()
            period_duration = current_time - self.stats.period_start_time
            if period_duration > 0:
                self.stats.current_fps = self.stats.frames_processed_period / period_duration

    def _update_performance_stats(self, processing_time: float) -> None:
        """Update legacy performance statistics for compatibility."""
        self.processing_times.append(processing_time)
        if len(self.processing_times) > 100:
            self.processing_times.pop(0)

        # Only update legacy stats if performance tracking is disabled
        if not self.enable_performance_tracking:
            # Calculate FPS for legacy compatibility
            current_time = time.time()
            if current_time - self.last_fps_time >= 1.0:
                frames_in_period = len([t for t in self.processing_times
                                      if current_time - t/1000 <= 1.0])
                self.last_fps_time = current_time
    
    def publish_performance_stats(self) -> None:
        """Publish performance statistics (only if performance tracking is enabled)."""
        if not self.enable_performance_tracking or self.stats is None:
            return

        try:
            msg = VisionPerformance()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = 'base_link'
            msg.feature_name = str(self.feature_name)

            # Frame processing counts for this period
            msg.frames_processed_period = int(self.stats.frames_processed_period)
            msg.current_fps = float(self.stats.current_fps)

            # Pipeline stage timing (averaged over measurement period)
            msg.avg_preprocessing_time = float(self.stats.avg_preprocessing_time)
            msg.avg_mediapipe_time = float(self.stats.avg_mediapipe_time)
            msg.avg_postprocessing_time = float(self.stats.avg_postprocessing_time)
            msg.avg_total_pipeline_time = float(self.stats.avg_total_pipeline_time)

            self.performance_publisher.publish(msg)

            # Reset period counters
            self.stats.frames_processed_period = 0
            self.stats.period_start_time = time.perf_counter()

            self.get_logger().debug(f'Published performance stats for {self.feature_name}: '
                                  f'FPS={msg.current_fps:.1f}, '
                                  f'Pipeline={msg.avg_total_pipeline_time:.1f}ms')

        except Exception as e:
            self.get_logger().error(f'Error publishing performance stats: {e}')
    
    def get_latest_results(self) -> Optional[Any]:
        """Get the latest processing results."""
        return self.latest_results
    
    def update_config(self, new_config: ProcessingConfig) -> None:
        """Update processing configuration."""
        self.config = new_config
        self.get_logger().info(f'Configuration updated for {self.feature_name}')
    
    def enable_feature(self) -> None:
        """Enable processing for this feature."""
        self.config.enabled = True
        self.get_logger().info(f'{self.feature_name} enabled')
    
    def disable_feature(self) -> None:
        """Disable processing for this feature."""
        self.config.enabled = False
        self.get_logger().info(f'{self.feature_name} disabled')
    
    def _flush_buffered_logger(self) -> None:
        """Timer callback to flush the buffered logger."""
        try:
            self.buffered_logger.flush()
        except Exception as e:
            self.get_logger().error(f'Error flushing buffered logger: {e}')

    def log_buffered_event(self, event_type: str, message: str, **kwargs) -> None:
        """Log an event to the buffered logger."""
        try:
            self.buffered_logger.log_event(event_type, message, **kwargs)
        except Exception as e:
            self.get_logger().error(f'Error logging buffered event: {e}')

    def get_buffer_stats(self) -> Dict[str, Any]:
        """Get buffered logger statistics."""
        try:
            return self.buffered_logger.get_stats()
        except Exception as e:
            self.get_logger().error(f'Error getting buffer stats: {e}')
            return {'error': str(e)}

    def cleanup(self) -> None:
        """Cleanup resources when shutting down."""
        # Flush any remaining buffered logs
        try:
            self.buffered_logger.flush()
        except Exception as e:
            self.get_logger().error(f'Error flushing buffer during cleanup: {e}')

        self.get_logger().info(f'Shutting down {self.feature_name} node')


class MediaPipeCallbackMixin:
    """
    Mixin class for MediaPipe nodes that use callback-based processing.
    Provides common callback handling functionality.
    """
    
    def __init__(self):
        self.callback_results = None
        self.callback_lock = threading.Lock()
    
    def create_callback(self, result_type: str) -> Callable:
        """Create a callback function for MediaPipe processing."""
        def callback(result, output_image, timestamp_ms):
            with self.callback_lock:
                self.callback_results = {
                    'result': result,
                    'output_image': output_image,
                    'timestamp': timestamp_ms,
                    'type': result_type
                }

        return callback
    
    def get_callback_results(self) -> Optional[Dict]:
        """Get the latest callback results."""
        with self.callback_lock:
            results = self.callback_results
            self.callback_results = None  # Clear after reading
            return results
