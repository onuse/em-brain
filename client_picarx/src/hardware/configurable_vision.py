#!/usr/bin/env python3
"""
Configurable Vision Module

Uses robot_config.json to set resolution.
Defaults to 640x480 as the bare minimum for real vision.

The brain server will automatically adapt its neural field
to handle whatever resolution we send!
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

try:
    from picamera2 import Picamera2
    CAMERA_AVAILABLE = True
except ImportError:
    CAMERA_AVAILABLE = False
    print("⚠️ Camera not available")


class ConfigurableVision:
    """
    Vision module that reads resolution from config.
    
    640x480 minimum for real vision tasks.
    The brain's nn.Linear projection handles any input size!
    """
    
    def __init__(self, config_path: str = "config/robot_config.json"):
        """Load config and initialize camera."""
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Extract vision settings
        vision_config = self.config.get('vision', {})
        self.enabled = vision_config.get('enabled', True) and CAMERA_AVAILABLE
        self.resolution = tuple(vision_config.get('resolution', [640, 480]))
        self.fps = vision_config.get('fps', 30)
        self.format = vision_config.get('format', 'grayscale')
        
        # Calculate data dimensions
        self.pixels = self.resolution[0] * self.resolution[1]
        if self.format == 'rgb':
            self.output_dim = self.pixels * 3
        else:
            self.output_dim = self.pixels
        
        # Initialize camera
        self.camera = None
        if self.enabled and CAMERA_AVAILABLE:
            self._init_camera()
        
        self._print_info()
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from JSON file."""
        config_file = Path(config_path)
        if config_file.exists():
            with open(config_file, 'r') as f:
                return json.load(f)
        else:
            print(f"⚠️ Config not found at {config_path}, using defaults")
            return {
                'vision': {
                    'enabled': True,
                    'resolution': [640, 480],
                    'fps': 30,
                    'format': 'grayscale'
                }
            }
    
    def _init_camera(self):
        """Initialize camera with configured resolution."""
        try:
            self.camera = Picamera2()
            
            # Configure for specified resolution and format
            config = self.camera.create_video_configuration(
                main={
                    "size": self.resolution,
                    "format": "RGB888"  # Always capture RGB, convert if needed
                },
                controls={
                    "FrameRate": float(self.fps),
                },
                buffer_count=2  # Small buffer for low latency
            )
            self.camera.configure(config)
            self.camera.start()
            
            print(f"✅ Camera initialized successfully")
            
        except Exception as e:
            print(f"❌ Camera initialization failed: {e}")
            self.enabled = False
    
    def _print_info(self):
        """Print vision configuration info."""
        if self.enabled:
            bandwidth_mbps = (self.output_dim * self.fps * 4) / 1_000_000  # float32
            print(f"📷 VISION CONFIGURATION")
            print(f"   Resolution: {self.resolution[0]}×{self.resolution[1]}")
            print(f"   Format: {self.format}")
            print(f"   Pixels: {self.pixels:,}")
            print(f"   Output dimension: {self.output_dim:,}")
            print(f"   FPS target: {self.fps}")
            print(f"   Data rate: {self.pixels * self.fps:,} pixels/sec")
            print(f"   Network bandwidth: {bandwidth_mbps:.1f} MB/s")
            
            if self.pixels >= 1_000_000:
                print(f"   🔥 EXTREME MODE: Over 1 MILLION pixels!")
            elif self.pixels >= 300_000:
                print(f"   💪 REAL VISION: This will challenge the brain!")
        else:
            print("📷 Vision disabled or unavailable")
    
    def get_frame(self) -> np.ndarray:
        """
        Get current frame as numpy array.
        
        Returns:
            Grayscale: (height, width) normalized to 0-1
            RGB: (height, width, 3) normalized to 0-1
        """
        if not self.enabled or not self.camera:
            # Return neutral gray frame
            if self.format == 'rgb':
                return np.ones((*self.resolution, 3)) * 0.5
            else:
                return np.ones(self.resolution) * 0.5
        
        try:
            # Capture RGB frame
            frame = self.camera.capture_array()
            
            # Normalize to 0-1
            frame = frame.astype(np.float32) / 255.0
            
            # Convert to grayscale if configured
            if self.format == 'grayscale':
                frame = np.mean(frame, axis=2)
            
            return frame
            
        except Exception as e:
            print(f"Frame capture error: {e}")
            if self.format == 'rgb':
                return np.ones((*self.resolution, 3)) * 0.5
            else:
                return np.ones(self.resolution) * 0.5
    
    def get_flattened_frame(self) -> List[float]:
        """
        Get frame as flattened list for brain input.
        
        This is what gets sent to the brain server.
        At 640x480, this is 307,200 values!
        """
        frame = self.get_frame()
        return frame.flatten().tolist()
    
    def get_downsampled_frame(self, target_size: Tuple[int, int] = (64, 64)) -> np.ndarray:
        """
        Get downsampled frame if needed for visualization or debugging.
        
        NOT for brain input - brain gets full resolution!
        """
        frame = self.get_frame()
        
        if frame.shape[:2] == target_size:
            return frame
        
        # Simple averaging downsample
        h_ratio = frame.shape[0] // target_size[0]
        w_ratio = frame.shape[1] // target_size[1]
        
        if self.format == 'rgb':
            downsampled = np.zeros((*target_size, 3))
            for i in range(target_size[0]):
                for j in range(target_size[1]):
                    region = frame[i*h_ratio:(i+1)*h_ratio, 
                                  j*w_ratio:(j+1)*w_ratio]
                    downsampled[i, j] = np.mean(region, axis=(0, 1))
        else:
            downsampled = np.zeros(target_size)
            for i in range(target_size[0]):
                for j in range(target_size[1]):
                    region = frame[i*h_ratio:(i+1)*h_ratio,
                                  j*w_ratio:(j+1)*w_ratio]
                    downsampled[i, j] = np.mean(region)
        
        return downsampled
    
    def cleanup(self):
        """Clean shutdown."""
        if self.camera:
            self.camera.stop()
            self.camera.close()
            self.enabled = False


if __name__ == "__main__":
    # Test the configurable vision
    import time
    
    print("Testing configurable vision module...")
    
    # Create config for testing
    test_config = {
        'vision': {
            'enabled': True,
            'resolution': [640, 480],
            'fps': 30,
            'format': 'grayscale'
        }
    }
    
    # Save test config
    with open('/tmp/test_vision_config.json', 'w') as f:
        json.dump(test_config, f)
    
    # Test vision module
    vision = ConfigurableVision('/tmp/test_vision_config.json')
    
    if vision.enabled:
        print("\nCapturing frames...")
        
        for i in range(3):
            start = time.time()
            
            # Get full resolution frame
            frame_flat = vision.get_flattened_frame()
            
            elapsed = time.time() - start
            
            print(f"Frame {i}: {len(frame_flat)} values in {elapsed*1000:.1f}ms")
            
            # Also test downsampled for visualization
            small = vision.get_downsampled_frame((8, 8))
            print(f"  Downsampled to 8x8 for viz: shape={small.shape}")
            
            time.sleep(0.5)
        
        vision.cleanup()
    else:
        print("No camera available for testing")