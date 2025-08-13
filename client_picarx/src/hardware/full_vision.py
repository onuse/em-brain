#!/usr/bin/env python3
"""
Full Vision Module for PiCar-X

NO DOWNSAMPLING - Give the brain the full visual stream.
Let the brain learn to handle the bandwidth.

Philosophy: Intelligence emerges from computational pressure.
If we make it easy, why would the brain become intelligent?
"""

import numpy as np
from typing import List, Optional

try:
    from picamera2 import Picamera2
    CAMERA_AVAILABLE = True
except ImportError:
    CAMERA_AVAILABLE = False
    print("⚠️ Camera not available")


class FullVision:
    """
    Full resolution vision for the brain.
    
    No preprocessing, no downsampling, no "help".
    The brain gets the full fire hose and must learn to cope.
    """
    
    def __init__(self, resolution=(64, 64)):
        """
        Initialize with full resolution.
        
        64x64 = 4096 pixels. Yes, it's a lot.
        That's the point - force the brain to be intelligent.
        """
        self.resolution = resolution
        self.camera = None
        self.enabled = False
        
        if CAMERA_AVAILABLE:
            self._init_camera()
    
    def _init_camera(self):
        """Initialize camera for full resolution streaming."""
        try:
            self.camera = Picamera2()
            
            # Configure for speed, not quality
            config = self.camera.create_video_configuration(
                main={
                    "size": self.resolution,
                    "format": "RGB888"
                },
                controls={
                    "FrameRate": 30.0,  # Push for high FPS
                },
                buffer_count=1  # Minimal latency
            )
            self.camera.configure(config)
            self.camera.start()
            
            self.enabled = True
            print(f"✅ Full vision initialized: {self.resolution[0]}x{self.resolution[1]} = {self.resolution[0]*self.resolution[1]} pixels")
            print(f"   Brain must handle {self.resolution[0]*self.resolution[1]*30:.1f} pixels/second!")
            
        except Exception as e:
            print(f"❌ Camera init failed: {e}")
            self.enabled = False
    
    def get_full_frame(self) -> List[float]:
        """
        Get FULL frame as flat array for brain.
        
        Returns 4096 grayscale values (for 64x64).
        No mercy, no downsampling. Brain deals with it all.
        """
        if not self.enabled:
            # Even mock data is full resolution
            return [0.5] * (self.resolution[0] * self.resolution[1])
        
        try:
            # Get raw frame
            frame = self.camera.capture_array()
            
            # Convert to grayscale (or keep RGB for even more data!)
            gray = np.mean(frame, axis=2)
            
            # Flatten and normalize
            # This is 4096 values for 64x64
            flat = gray.flatten() / 255.0
            
            return flat.tolist()
            
        except Exception as e:
            print(f"Vision error: {e}")
            return [0.5] * (self.resolution[0] * self.resolution[1])
    
    def get_rgb_frame(self) -> List[float]:
        """
        Get FULL RGB frame - 3x the data!
        
        Returns 12,288 values for 64x64 RGB.
        Maximum bandwidth pressure on the brain.
        """
        if not self.enabled:
            return [0.5] * (self.resolution[0] * self.resolution[1] * 3)
        
        try:
            frame = self.camera.capture_array()
            
            # Flatten RGB and normalize
            # This is 12,288 values for 64x64x3
            flat = frame.flatten() / 255.0
            
            return flat.tolist()
            
        except Exception as e:
            print(f"Vision error: {e}")
            return [0.5] * (self.resolution[0] * self.resolution[1] * 3)
    
    def cleanup(self):
        """Clean shutdown."""
        if self.enabled and self.camera:
            self.camera.stop()
            self.camera.close()


# Let's also update our brainstem concept
class HighBandwidthBrainstem:
    """
    Concept: Don't protect the brain from data.
    
    Send EVERYTHING and let the brain develop
    intelligence to handle it.
    """
    
    def sensors_to_brain(self, vision, audio, sensors):
        """
        Build massive sensory vector.
        
        No downsampling, no selection, no filtering.
        """
        brain_input = []
        
        # Basic sensors (5 channels)
        brain_input.extend(sensors)
        
        # Full vision (4096 channels for 64x64)
        brain_input.extend(vision.get_full_frame())
        
        # Full audio spectrum (512 FFT bins?)
        brain_input.extend(audio.get_full_spectrum())
        
        # Total: 4000+ channels
        # Brain must learn to handle this!
        
        return brain_input


"""
Why this is RIGHT:

1. Biological brains handle millions of photoreceptors
   - Human eye: ~120 million rods, 6 million cones
   - Brain learns to compress, attend, extract
   
2. Computational pressure creates intelligence
   - Easy problems don't need intelligence
   - Hard problems force innovation
   
3. The brain WILL adapt
   - Field dynamics will evolve compression
   - Attention mechanisms will emerge
   - Feature extraction will self-organize
   
4. This is how real intelligence works
   - Not by us pre-selecting "important" features
   - But by learning what's important

The brain server's GPU can handle it.
The network can handle it (local network).
Let's stop babying the brain and give it a real challenge!
"""