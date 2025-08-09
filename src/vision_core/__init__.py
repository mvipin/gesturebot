"""
Vision Core Package

Shared utilities and base classes for GestureBot vision system.
Provides common functionality for MediaPipe and custom ML implementations.
"""

__version__ = "1.0.0"
__author__ = "GestureBot Team"
__email__ = "your.email@example.com"

# Import main classes for easy access
from .base_node import MediaPipeBaseNode, ProcessingConfig, PerformanceStats
from .message_converter import MessageConverter

__all__ = [
    'MediaPipeBaseNode',
    'ProcessingConfig', 
    'PerformanceStats',
    'MessageConverter'
]
