#!/usr/bin/env python3
"""
MediaPipe Controller interfaces and concrete implementations.
Provides a composition-friendly abstraction so nodes own only ROS infra, while
controllers own MediaPipe-specific lifecycle, submission, and cleanup.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Callable

import mediapipe as mp
from mediapipe.tasks import python as mp_py
from mediapipe.tasks.python import vision as mp_vis


class MediaPipeController(ABC):
    """Abstract controller for MediaPipe pipelines."""

    @abstractmethod
    def is_ready(self) -> bool:
        """Return True when the underlying pipeline is initialized and ready."""
        raise NotImplementedError

    @abstractmethod
    def detect_async(self, mp_image: mp.Image, timestamp_ms: int) -> None:
        """Submit an image to the pipeline asynchronously."""
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        """Release underlying resources."""
        raise NotImplementedError


class ObjectDetectionController(MediaPipeController):
    """Controller for MediaPipe ObjectDetector (EfficientDet) using Tasks API."""

    def __init__(
        self,
        model_path: str,
        confidence_threshold: float,
        max_results: int,
        result_callback: Callable,
    ) -> None:
        base_options = mp_py.BaseOptions(model_asset_path=model_path)
        options = mp_vis.ObjectDetectorOptions(
            base_options=base_options,
            running_mode=mp_vis.RunningMode.LIVE_STREAM,
            max_results=max_results,
            score_threshold=confidence_threshold,
            result_callback=result_callback,
        )
        # Initialize detector upon construction
        self._detector = mp_vis.ObjectDetector.create_from_options(options)

    def is_ready(self) -> bool:
        return self._detector is not None

    def detect_async(self, mp_image: mp.Image, timestamp_ms: int) -> None:
        if self._detector is None:
            return
        self._detector.detect_async(mp_image, timestamp_ms)

    def close(self) -> None:
        if self._detector is not None:
            try:
                self._detector.close()
            finally:
                self._detector = None


class GestureRecognitionController(MediaPipeController):
    """Controller for MediaPipe GestureRecognizer using Tasks API."""

    def __init__(
        self,
        model_path: str,
        confidence_threshold: float,
        max_hands: int,
        result_callback: Callable,
    ) -> None:
        base_options = mp_py.BaseOptions(model_asset_path=model_path)
        options = mp_vis.GestureRecognizerOptions(
            base_options=base_options,
            running_mode=mp_vis.RunningMode.LIVE_STREAM,
            num_hands=max_hands,
            min_hand_detection_confidence=confidence_threshold,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            result_callback=result_callback,
        )
        # Initialize recognizer upon construction
        self._recognizer = mp_vis.GestureRecognizer.create_from_options(options)

    def is_ready(self) -> bool:
        return self._recognizer is not None

    def detect_async(self, mp_image: mp.Image, timestamp_ms: int) -> None:
        if self._recognizer is None:
            return
        self._recognizer.recognize_async(mp_image, timestamp_ms)

    def close(self) -> None:
        if self._recognizer is not None:
            try:
                self._recognizer.close()
            finally:
                self._recognizer = None

