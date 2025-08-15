#!/usr/bin/env python3
"""
Vision Field Injector - Direct MJPEG to field injection.

This is THE critical sensor - currently takes 98% of processing time.
Moving to parallel thread enables reasonable resolutions again!
"""

import socket
import struct
import threading
import time
import torch
import numpy as np
from typing import Dict, Optional, Tuple
import io


class VisionFieldInjector:
    """
    Vision sensor thread with direct field injection.
    Receives MJPEG frames and injects into visual cortex region.
    """
    
    def __init__(self, brain_field: torch.Tensor, port: int = 10002, 
                 target_resolution: Tuple[int, int] = (320, 240)):
        """
        Args:
            brain_field: Reference to brain's field tensor
            port: UDP port to listen on for video frames
            target_resolution: Expected frame resolution (width, height)
        """
        self.field = brain_field
        self.port = port
        self.target_resolution = target_resolution
        self.running = False
        self.thread = None
        
        # Vision gets a large region - the "visual cortex"
        self.field_region = self._allocate_region()
        
        # Frame buffer for reassembly (MJPEG frames can be chunked)
        self.frame_buffer = bytearray()
        self.expected_frame_size = 0
        
        # Stats
        self.frames_received = 0
        self.frames_injected = 0
        self.last_frame_time = 0
        self.avg_frame_size = 0
        
    def _allocate_region(self) -> Dict:
        """
        Allocate field region for visual processing.
        
        Vision needs significant space:
        - Spatial dimensions for retinotopic mapping
        - Multiple channels for features (edges, colors, motion)
        """
        # Use a significant portion of field for vision
        # Real brains dedicate ~30% of cortex to vision!
        field_shape = self.field.shape
        
        # Use half the spatial dimensions for vision
        spatial_end = field_shape[0] // 2
        
        return {
            'spatial': (slice(0, spatial_end), 
                       slice(0, spatial_end), 
                       slice(0, spatial_end)),
            'channels': slice(0, 16),  # First 16 channels for visual features
            'decay_rate': 0.9,  # Fast decay for dynamic vision
            'injection_strength': 0.3,
            # Mapping from image coords to field coords
            'resize_factor': spatial_end / max(self.target_resolution)
        }
    
    def start(self):
        """Start the vision injection thread."""
        if self.running:
            return
            
        self.running = True
        self.thread = threading.Thread(target=self._injection_loop, daemon=True)
        self.thread.start()
        print(f"👁️  Vision field injector started on port {self.port}")
        print(f"   Resolution: {self.target_resolution[0]}x{self.target_resolution[1]}")
        print(f"   Field region: {self.field_region['spatial']}")
    
    def stop(self):
        """Stop the injection thread."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)
        print(f"👁️  Vision injector stopped. Frames: {self.frames_received}, "
              f"Injected: {self.frames_injected}")
    
    def _injection_loop(self):
        """
        Main loop - receives MJPEG frames and injects into field.
        This removes vision processing from main brain thread!
        """
        # Setup UDP socket with larger buffer for video
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        # Increase receive buffer for video frames
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024*1024)  # 1MB
        
        sock.bind(('', self.port))
        sock.settimeout(0.1)  # 100ms timeout
        
        print(f"👁️  Vision injector listening on UDP:{self.port}")
        
        while self.running:
            try:
                # Receive frame chunk
                data, addr = sock.recvfrom(65536)  # Max UDP packet
                
                if len(data) < 8:
                    continue
                
                # Parse header (frame_id, chunk_id, total_chunks, data_length)
                frame_id, chunk_id, total_chunks, data_len = struct.unpack('!IIHH', data[:12])
                chunk_data = data[12:12+data_len]
                
                # Simple case: single chunk (small frames)
                if total_chunks == 1:
                    self._process_frame(chunk_data)
                else:
                    # Multi-chunk: reassemble (for larger frames)
                    # TODO: Implement frame reassembly for larger resolutions
                    pass
                    
            except socket.timeout:
                # No frame received - apply decay
                self._inject_decay()
                
            except Exception as e:
                if self.frames_received % 100 == 0:
                    print(f"⚠️  Vision injector error: {e}")
        
        sock.close()
    
    def _process_frame(self, jpeg_data: bytes):
        """
        Process MJPEG frame and inject into field.
        This is where 98% of CPU was being used in main thread!
        """
        self.frames_received += 1
        
        try:
            # Decode JPEG to numpy array
            # In production, use turbojpeg for speed
            import cv2
            nparr = np.frombuffer(jpeg_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is None:
                return
            
            # Convert to grayscale for simplicity (can do color later)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Resize to field dimensions
            spatial = self.field_region['spatial']
            field_h = spatial[0].stop - spatial[0].start
            field_w = spatial[1].stop - spatial[1].start
            field_d = spatial[2].stop - spatial[2].start
            
            # Resize image to fit field spatial dimensions
            resized = cv2.resize(gray, (field_w, field_h))
            
            # Normalize to [0, 1]
            normalized = resized.astype(np.float32) / 255.0
            
            # Inject into field
            self._inject_into_field(normalized)
            
            self.frames_injected += 1
            self.last_frame_time = time.time()
            
            # Log periodically
            if self.frames_injected % 30 == 0:  # Every ~2 seconds at 15fps
                print(f"👁️  Frame #{self.frames_injected}: "
                      f"{frame.shape[1]}x{frame.shape[0]} → "
                      f"field[{field_w}x{field_h}x{field_d}]")
                
        except Exception as e:
            print(f"⚠️  Frame processing error: {e}")
    
    def _inject_into_field(self, image: np.ndarray):
        """
        Inject visual data into field with feature extraction.
        
        This is where visual processing happens in parallel!
        No longer blocks the main brain thread.
        """
        spatial = self.field_region['spatial']
        channels = self.field_region['channels']
        decay = self.field_region['decay_rate']
        strength = self.field_region['injection_strength']
        
        with torch.no_grad():
            # Decay old visual activation
            self.field[spatial][..., channels] *= decay
            
            # Get field dimensions
            field_h = spatial[0].stop - spatial[0].start
            field_w = spatial[1].stop - spatial[1].start
            field_d = spatial[2].stop - spatial[2].start
            
            # Create visual feature tensor
            features = torch.zeros(field_h, field_w, field_d, 16)
            
            # Channel 0-3: Raw intensity at different depths
            for d in range(min(4, field_d)):
                features[:, :, d, 0] = torch.from_numpy(image) * (1.0 - d*0.2)
            
            # Channel 4-7: Edge detection (simple gradients)
            if image.shape[0] > 1 and image.shape[1] > 1:
                dy = np.diff(image, axis=0)
                dx = np.diff(image, axis=1)
                
                # Pad to match original size
                dy = np.pad(dy, ((0,1), (0,0)), mode='edge')
                dx = np.pad(dx, ((0,0), (0,1)), mode='edge')
                
                # Inject gradients
                features[:, :, 0, 4] = torch.from_numpy(np.abs(dy))
                features[:, :, 0, 5] = torch.from_numpy(np.abs(dx))
                
                # Magnitude
                magnitude = np.sqrt(dy**2 + dx**2)
                features[:, :, 0, 6] = torch.from_numpy(magnitude)
            
            # Channel 8-11: Motion (difference from last frame)
            if hasattr(self, '_last_image'):
                motion = np.abs(image - self._last_image)
                features[:, :, 0, 8] = torch.from_numpy(motion)
                
                # Directional motion
                motion_x = image - self._last_image
                features[:, :, 0, 9] = torch.from_numpy(np.maximum(motion_x, 0))  # Rightward
                features[:, :, 0, 10] = torch.from_numpy(np.maximum(-motion_x, 0))  # Leftward
            
            self._last_image = image.copy()
            
            # Channel 12-15: Summary statistics
            features[0, 0, 0, 12] = image.mean()  # Overall brightness
            features[0, 0, 0, 13] = image.std()   # Contrast
            features[0, 0, 0, 14] = image.max()   # Brightest point
            features[0, 0, 0, 15] = image.min()   # Darkest point
            
            # Inject features into field
            self.field[spatial][..., channels] += features * strength
            
            # Clamp to prevent explosion
            self.field[spatial][..., channels] = torch.clamp(
                self.field[spatial][..., channels], -10, 10
            )
    
    def _inject_decay(self):
        """
        Apply decay when no frames received.
        Represents visual memory fading.
        """
        spatial = self.field_region['spatial']
        channels = self.field_region['channels']
        
        with torch.no_grad():
            # Faster decay when no input (like closing eyes)
            self.field[spatial][..., channels] *= 0.8


# Test standalone
if __name__ == "__main__":
    print("VISION FIELD INJECTOR TEST")
    print("=" * 50)
    
    # Create test field
    field = torch.zeros(32, 32, 32, 64)  # Larger field for vision
    print(f"Created field: {field.shape}")
    
    # Create vision injector
    injector = VisionFieldInjector(field, port=10002, target_resolution=(320, 240))
    injector.start()
    
    print("\nVision injector ready.")
    print("Send MJPEG frames to UDP port 10002")
    print("Or run: python3 client_picarx/examples/udp_vision_stream.py")
    
    # Monitor field changes
    try:
        for i in range(60):  # 60 seconds
            spatial = injector.field_region['spatial']
            channels = injector.field_region['channels']
            
            region = field[spatial][..., channels]
            mean_val = region.mean().item()
            max_val = region.max().item()
            
            print(f"t={i}s: frames={injector.frames_received}, "
                  f"injected={injector.frames_injected}, "
                  f"field: mean={mean_val:.3f}, max={max_val:.3f}")
            
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nStopping...")
    
    injector.stop()
    print("\n✅ Vision injector test complete!")