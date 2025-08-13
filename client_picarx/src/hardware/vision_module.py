#!/usr/bin/env python3
"""
Future Camera Module for PiCar-X

This is a TEMPLATE for adding camera support later.
Not currently used, but shows how we could add vision
without any complex dependencies.

Philosophy: Convert visual input to simple features
that fit in the brain's unused input channels (5-23).
"""

import numpy as np
from typing import List, Optional

try:
    # Option 1: Using picamera2 (Raspberry Pi specific)
    from picamera2 import Picamera2
    CAMERA_AVAILABLE = True
    CAMERA_TYPE = "picamera2"
except ImportError:
    try:
        # Option 2: Using OpenCV (heavier but universal)
        import cv2
        CAMERA_AVAILABLE = True
        CAMERA_TYPE = "opencv"
    except ImportError:
        CAMERA_AVAILABLE = False
        CAMERA_TYPE = None
        print("⚠️ No camera libraries available")


class SimpleVisionModule:
    """
    Convert camera input to brain-friendly features.
    
    No complex computer vision - just basic features
    the brain can learn to interpret.
    """
    
    def __init__(self, resolution=(64, 64)):
        """Initialize camera with low resolution for speed."""
        self.resolution = resolution
        self.camera = None
        self.enabled = False
        
        if CAMERA_AVAILABLE:
            self._init_camera()
    
    def _init_camera(self):
        """Initialize camera based on available library."""
        try:
            if CAMERA_TYPE == "picamera2":
                self.camera = Picamera2()
                config = self.camera.create_preview_configuration(
                    main={"size": self.resolution, "format": "RGB888"}
                )
                self.camera.configure(config)
                self.camera.start()
                self.enabled = True
                print(f"✅ Camera initialized with picamera2 at {self.resolution}")
                
            elif CAMERA_TYPE == "opencv":
                self.camera = cv2.VideoCapture(0)
                self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
                self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
                self.enabled = True
                print(f"✅ Camera initialized with OpenCV at {self.resolution}")
                
        except Exception as e:
            print(f"❌ Camera initialization failed: {e}")
            self.enabled = False
    
    def get_vision_features(self) -> List[float]:
        """
        Get simple vision features for brain input channels.
        
        Returns 14 features that could map to brain channels 5-18:
        - 9 region brightnesses (3x3 grid)
        - 2 color channels (red vs blue dominance)
        - 1 overall brightness
        - 1 contrast measure
        - 1 motion estimate (if previous frame available)
        """
        
        if not self.enabled:
            # Return neutral values if no camera
            return [0.5] * 14
        
        try:
            # Capture frame
            frame = self._capture_frame()
            if frame is None:
                return [0.5] * 14
            
            # Convert to grayscale for brightness analysis
            gray = np.mean(frame, axis=2)
            
            # Feature 1-9: 3x3 region brightness
            regions = self._extract_regions(gray, 3, 3)
            
            # Feature 10-11: Color dominance (R-G, B-G)
            red_dominance = np.mean(frame[:,:,0] - frame[:,:,1]) / 255.0 + 0.5
            blue_dominance = np.mean(frame[:,:,2] - frame[:,:,1]) / 255.0 + 0.5
            
            # Feature 12: Overall brightness
            overall_brightness = np.mean(gray) / 255.0
            
            # Feature 13: Contrast (standard deviation)
            contrast = np.std(gray) / 128.0  # Normalize to ~0-1
            
            # Feature 14: Motion (placeholder - would need previous frame)
            motion = 0.0  # Could implement frame differencing
            
            features = (
                regions +  # 9 values
                [red_dominance, blue_dominance,
                 overall_brightness, contrast, motion]
            )
            
            # Ensure all values are 0-1 range
            features = [max(0.0, min(1.0, f)) for f in features]
            
            return features
            
        except Exception as e:
            print(f"Vision processing error: {e}")
            return [0.5] * 14
    
    def _capture_frame(self) -> Optional[np.ndarray]:
        """Capture a frame from camera."""
        if CAMERA_TYPE == "picamera2":
            return self.camera.capture_array()
        elif CAMERA_TYPE == "opencv":
            ret, frame = self.camera.read()
            if ret:
                # Convert BGR to RGB
                return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return None
    
    def _extract_regions(self, image: np.ndarray, rows: int, cols: int) -> List[float]:
        """
        Divide image into grid and get average brightness per region.
        
        Returns normalized brightness values (0-1) for each region.
        """
        h, w = image.shape
        region_h = h // rows
        region_w = w // cols
        
        regions = []
        for r in range(rows):
            for c in range(cols):
                region = image[r*region_h:(r+1)*region_h,
                              c*region_w:(c+1)*region_w]
                brightness = np.mean(region) / 255.0
                regions.append(brightness)
        
        return regions
    
    def cleanup(self):
        """Clean up camera resources."""
        if self.enabled:
            if CAMERA_TYPE == "picamera2":
                self.camera.stop()
            elif CAMERA_TYPE == "opencv":
                self.camera.release()
            self.enabled = False


# Example integration with brainstem:
"""
# In brainstem.py, you could add:

def sensors_to_brain_format(self, raw: RawSensorData, 
                           vision: Optional[SimpleVisionModule] = None) -> List[float]:
    brain_input = [0.5] * 24
    
    # Existing sensors (channels 0-4)
    # ... current code ...
    
    # Add vision if available (channels 5-18)
    if vision and vision.enabled:
        vision_features = vision.get_vision_features()
        brain_input[5:19] = vision_features
    
    return brain_input
"""


if __name__ == "__main__":
    # Test the vision module
    print("Testing vision module...")
    
    vision = SimpleVisionModule()
    
    if vision.enabled:
        print("\nCapturing vision features...")
        for i in range(5):
            features = vision.get_vision_features()
            print(f"Frame {i}: regions={features[:9]}")
            print(f"  Colors: R={features[9]:.2f} B={features[10]:.2f}")
            print(f"  Brightness={features[11]:.2f} Contrast={features[12]:.2f}")
            import time
            time.sleep(0.5)
        
        vision.cleanup()
    else:
        print("No camera available for testing")