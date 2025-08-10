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
    priority: int = 1  # 0=critical, 1=high, 2=medium, 3=low


@dataclass
class PerformanceStats:
    """Performance statistics for monitoring."""
    frames_processed: int = 0
    avg_processing_time: float = 0.0
    current_fps: float = 0.0
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    last_update: float = 0.0


class MediaPipeBaseNode(Node, ABC):
    """
    Abstract base class for all MediaPipe feature nodes.
    Provides common functionality for camera input, processing, and output.
    Includes configurable buffered logging for diagnostics.
    """

    def __init__(self, node_name: str, feature_name: str, config: ProcessingConfig,
                 enable_buffered_logging: bool = True, unlimited_buffer_mode: bool = False):
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
        
        # Performance tracking
        self.stats = PerformanceStats()
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
            f'/vision/{self.feature_name}/performance',
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
        """Handle incoming camera frames."""
        if not self.config.enabled:
            return

        # Frame skipping for performance
        self.frame_counter += 1
        if self.frame_counter % (self.config.frame_skip + 1) != 0:
            return



        try:
            # Convert ROS image to OpenCV
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, 'bgr8')
            timestamp = time.time()

            # Process frame asynchronously to avoid blocking
            threading.Thread(
                target=self._process_frame_async,
                args=(cv_image, timestamp),
                daemon=True
            ).start()

        except Exception as e:
            self.get_logger().error(f'[{self.feature_name}] Error in image callback: {e}')
    
    def _process_frame_async(self, frame: np.ndarray, timestamp: float) -> None:
        """Process frame in separate thread."""
        if not self.processing_lock.acquire(blocking=False):
            # Skip frame if still processing previous one
            return

        try:
            start_time = time.time()

            # Call subclass implementation
            results = self.process_frame(frame, timestamp)

            # Track performance
            processing_time = (time.time() - start_time) * 1000  # ms
            self._update_performance_stats(processing_time)

            # Publish results
            if results is not None:
                self.publish_results(results, timestamp)
                self.latest_results = results

        except Exception as e:
            self.get_logger().error(f'[{self.feature_name}] Error processing frame: {e}')
        finally:
            self.processing_lock.release()
    
    def _update_performance_stats(self, processing_time: float) -> None:
        """Update performance statistics."""
        self.processing_times.append(processing_time)
        if len(self.processing_times) > 100:
            self.processing_times.pop(0)
        
        self.stats.frames_processed += 1
        self.stats.avg_processing_time = np.mean(self.processing_times)
        
        # Calculate FPS
        current_time = time.time()
        if current_time - self.last_fps_time >= 1.0:
            frames_in_period = len([t for t in self.processing_times 
                                  if current_time - t/1000 <= 1.0])
            self.stats.current_fps = frames_in_period
            self.last_fps_time = current_time
    
    def publish_performance_stats(self) -> None:
        """Publish performance statistics."""
        try:
            msg = VisionPerformance()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.feature_name = str(self.feature_name)
            msg.frames_processed = int(self.stats.frames_processed)
            msg.avg_processing_time = float(self.stats.avg_processing_time)
            msg.current_fps = float(self.stats.current_fps)
            msg.cpu_usage = float(self.stats.cpu_usage)
            msg.memory_usage = float(self.stats.memory_usage)
            msg.enabled = bool(self.config.enabled)
            msg.priority = int(self.config.priority)
            msg.processing_healthy = bool(self.stats.avg_processing_time < 200.0)  # 200ms threshold
            msg.status_message = str("operational" if msg.processing_healthy else "degraded")

            self.performance_publisher.publish(msg)
            self.get_logger().debug(f'Published performance stats for {self.feature_name}')

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
