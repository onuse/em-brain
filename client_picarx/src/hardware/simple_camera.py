#!/usr/bin/env python3
"""
Simple Camera Module for PiCar-X

Truly minimal camera access without heavy dependencies.
Just raw pixels for the brain to learn from.

Philosophy: The brain should learn what vision means,
not receive pre-processed "face detected" signals.
"""

import numpy as np
from typing import Optional, List

try:
    # Try picamera2 first (lightest option)
    from picamera2 import Picamera2
    CAMERA_TYPE = "picamera2"
    CAMERA_AVAILABLE = True
except ImportError:
    CAMERA_AVAILABLE = False
    CAMERA_TYPE = None
    print("⚠️ No camera available (picamera2 not installed)")


class SimpleCamera:
    """
    Minimal camera interface.
    
    No computer vision, no face detection, no preprocessing.
    Just raw pixels downsampled to brain-friendly features.
    """
    
    def __init__(self, resolution=(32, 32)):
        """
        Initialize camera with very low resolution.
        
        32x32 = 1024 pixels, still too much for brain.
        We'll downsample further to regions.
        """
        self.resolution = resolution
        self.camera = None
        self.enabled = False
        
        if CAMERA_AVAILABLE:
            self._init_camera()
    
    def _init_camera(self):
        """Initialize Raspberry Pi camera with minimal config."""
        try:
            self.camera = Picamera2()
            
            # Minimal configuration - no preview, no GPU processing
            config = self.camera.create_still_configuration(
                main={"size": self.resolution, "format": "RGB888"},
                buffer_count=1  # Minimal buffering
            )
            self.camera.configure(config)
            self.camera.start()
            
            self.enabled = True
            print(f"✅ Camera initialized at {self.resolution[0]}x{self.resolution[1]}")
            
        except Exception as e:
            print(f"❌ Camera init failed: {e}")
            self.enabled = False
    
    def get_brain_features(self) -> List[float]:
        """
        Get minimal vision features for brain.
        
        Returns 9 values: 3x3 grid of brightness.
        Brain will learn what these patterns mean.
        """
        if not self.enabled:
            return [0.5] * 9  # Neutral if no camera
        
        try:
            # Capture frame
            frame = self.camera.capture_array()
            
            # Convert to grayscale (average RGB)
            gray = np.mean(frame, axis=2)
            
            # Downsample to 3x3 regions
            h, w = gray.shape
            region_h = h // 3
            region_w = w // 3
            
            features = []
            for row in range(3):
                for col in range(3):
                    # Get average brightness of each region
                    region = gray[
                        row * region_h:(row + 1) * region_h,
                        col * region_w:(col + 1) * region_w
                    ]
                    brightness = np.mean(region) / 255.0
                    features.append(brightness)
            
            return features
            
        except Exception as e:
            print(f"Camera read error: {e}")
            return [0.5] * 9
    
    def cleanup(self):
        """Clean shutdown."""
        if self.enabled and self.camera:
            self.camera.stop()
            self.camera.close()
            self.enabled = False


# Alternative if we absolutely need to use vilib
class VilibWrapper:
    """
    If forced to use vilib, extract ONLY raw camera data.
    Ignore all the face detection, object detection, etc.
    """
    
    def __init__(self):
        """Initialize vilib but use ONLY camera, not CV features."""
        try:
            from vilib import Vilib
            Vilib.camera_start(vflip=False, hflip=False)
            self.enabled = True
            print("✅ Using vilib for camera (raw mode only)")
        except:
            self.enabled = False
            print("❌ vilib not available")
    
    def get_brain_features(self) -> List[float]:
        """Get raw pixels, ignore vilib's CV features."""
        if not self.enabled:
            return [0.5] * 9
        
        try:
            from vilib import Vilib
            # Get raw frame, don't use face_detect() etc
            frame = Vilib.img  # Raw image access
            
            if frame is None:
                return [0.5] * 9
            
            # Same downsampling as SimpleCamera
            gray = np.mean(frame, axis=2)
            # ... rest same as above
            
        except:
            return [0.5] * 9


if __name__ == "__main__":
    # Test the simple camera
    print("Testing simple camera...")
    
    cam = SimpleCamera(resolution=(32, 32))
    
    if cam.enabled:
        import time
        for i in range(5):
            features = cam.get_brain_features()
            print(f"Frame {i}: {features}")
            time.sleep(0.5)
        
        cam.cleanup()
    else:
        print("No camera to test")