"""
Simple Video Stream Handler

Video has completely different requirements:
- High bandwidth (640x480 @ 30fps = ~27MB/s raw)
- Can drop frames without catastrophe
- Should never block motor control
- Needs continuous flow

This runs on a separate UDP port and injects directly into the field.
"""

import socket
import struct
import threading
import time
import numpy as np
import torch
from typing import Optional, Callable
import cv2


class SimpleVideoStream:
    """
    Separate video stream handler that injects directly into the brain field.
    
    Key design:
    - UDP port 10002 (fire and forget)
    - Drops old frames if processing is slow
    - Never blocks the main control loop
    - Direct field injection (no sensor vector)
    """
    
    def __init__(self, brain, port: int = 10002):
        """
        Initialize video stream.
        
        Args:
            brain: UnifiedFieldBrain instance to inject into
            port: UDP port for video stream
        """
        self.brain = brain
        self.port = port
        self.running = False
        self.thread = None
        
        # Stats
        self.frame_count = 0
        self.dropped_frames = 0
        self.last_frame_time = 0
        
        # Processing state
        self.processing = False
        self.latest_frame = None
        
        # Pre-allocate injection regions (visual cortex area)
        self._setup_injection_regions()
    
    def _setup_injection_regions(self):
        """Setup where in the field video gets injected."""
        # Use channels 8-15 for vision (8 channels)
        self.vision_channels = slice(8, 16)
        
        # Use front half of the field for vision
        field_size = self.brain.spatial_size
        self.vision_region = (
            slice(0, field_size // 2),  # Front X
            slice(0, field_size),        # Full Y
            slice(0, field_size // 2),  # Front Z
        )
        
        # Pre-compute downsampling target size
        self.target_size = (
            field_size // 2,  # Height
            field_size        # Width
        )
        
        print(f"📹 Vision injection region: {self.target_size[0]}×{self.target_size[1]} "
              f"→ channels {self.vision_channels}")
    
    def _listen_loop(self):
        """Main UDP listening loop for video frames."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(('', self.port))
        sock.settimeout(0.1)  # Non-blocking with timeout
        
        print(f"📹 Video stream listening on UDP port {self.port}")
        
        frame_buffer = []
        
        while self.running:
            try:
                # Receive frame data
                data, addr = sock.recvfrom(65536)  # Max UDP packet
                
                if len(data) > 8:
                    # Parse header: [timestamp:8][frame_data:rest]
                    timestamp = struct.unpack('!Q', data[:8])[0]
                    frame_data = data[8:]
                    
                    # Check if it's JPEG or raw
                    if frame_data[:2] == b'\xff\xd8':  # JPEG magic number
                        # Decode JPEG
                        frame = cv2.imdecode(
                            np.frombuffer(frame_data, np.uint8),
                            cv2.IMREAD_GRAYSCALE
                        )
                    else:
                        # Raw frame with dimensions in header
                        if len(frame_data) > 4:
                            h, w = struct.unpack('!HH', frame_data[:4])
                            raw_data = frame_data[4:4 + h*w]
                            if len(raw_data) == h * w:
                                frame = np.frombuffer(raw_data, np.uint8).reshape(h, w)
                            else:
                                continue
                        else:
                            continue
                    
                    if frame is not None:
                        self.frame_count += 1
                        
                        # Drop frame if still processing previous one
                        if self.processing:
                            self.dropped_frames += 1
                            if self.dropped_frames % 30 == 0:
                                print(f"⚠️ Dropped {self.dropped_frames} frames (processing too slow)")
                        else:
                            # Process this frame
                            self.latest_frame = frame
                            self.last_frame_time = time.time()
                            self._inject_frame(frame)
                        
                        # Status every second (assuming ~30fps)
                        if self.frame_count % 30 == 0:
                            fps = 30.0 * (1.0 - self.dropped_frames / max(self.frame_count, 1))
                            print(f"📹 Frame {self.frame_count}, "
                                  f"effective FPS: {fps:.1f}, "
                                  f"shape: {frame.shape}")
                            
            except socket.timeout:
                # No frame received, that's OK
                continue
            except Exception as e:
                print(f"📹 Video stream error: {e}")
        
        sock.close()
        print("📹 Video stream stopped")
    
    def _inject_frame(self, frame: np.ndarray):
        """
        Inject video frame directly into brain field.
        
        This is where the magic happens - video becomes field activity.
        """
        self.processing = True
        
        try:
            # Downsample frame to fit field region
            if frame.shape != self.target_size:
                frame_resized = cv2.resize(
                    frame, 
                    (self.target_size[1], self.target_size[0]),
                    interpolation=cv2.INTER_AREA  # Good for downsampling
                )
            else:
                frame_resized = frame
            
            # Normalize to [-1, 1]
            frame_normalized = (frame_resized.astype(np.float32) / 127.5) - 1.0
            
            # Convert to torch tensor
            frame_tensor = torch.from_numpy(frame_normalized).to(self.brain.device)
            
            # Inject into multiple channels with different features
            with torch.no_grad():
                x_slice, y_slice, z_slice = self.vision_region
                
                # Channel 8: Raw intensity
                self.brain.field[x_slice, y_slice, z_slice, 8] += frame_tensor.unsqueeze(0) * 0.3
                
                # Channel 9: Edges (simple gradient)
                if frame_tensor.shape[0] > 1 and frame_tensor.shape[1] > 1:
                    dy = torch.diff(frame_tensor, dim=0, prepend=frame_tensor[:1])
                    dx = torch.diff(frame_tensor, dim=1, prepend=frame_tensor[:, :1])
                    edges = torch.sqrt(dx**2 + dy[:-1]**2)
                    self.brain.field[x_slice, y_slice, z_slice, 9] += edges.unsqueeze(0) * 0.2
                
                # Channel 10: Temporal difference (motion)
                if hasattr(self, 'last_frame_tensor'):
                    motion = frame_tensor - self.last_frame_tensor
                    self.brain.field[x_slice, y_slice, z_slice, 10] += motion.unsqueeze(0) * 0.25
                self.last_frame_tensor = frame_tensor.clone()
                
                # Channels 11-15: Multi-scale features (like a simple CNN)
                current = frame_tensor
                for i, ch in enumerate(range(11, min(16, 11 + 5))):
                    if current.shape[0] > 2 and current.shape[1] > 2:
                        # Simple pooling for multi-scale
                        current = torch.nn.functional.avg_pool2d(
                            current.unsqueeze(0).unsqueeze(0), 
                            kernel_size=2, 
                            stride=1, 
                            padding=0
                        ).squeeze()
                        
                        # Inject at decreasing strength
                        strength = 0.15 / (i + 1)
                        
                        # Pad or crop to fit
                        h_diff = self.target_size[0] - current.shape[0] 
                        w_diff = self.target_size[1] - current.shape[1]
                        
                        if h_diff > 0 or w_diff > 0:
                            # Pad with zeros
                            current = torch.nn.functional.pad(
                                current, 
                                (0, max(0, w_diff), 0, max(0, h_diff))
                            )
                        elif h_diff < 0 or w_diff < 0:
                            # Crop
                            current = current[:self.target_size[0], :self.target_size[1]]
                        
                        self.brain.field[x_slice, y_slice, z_slice, ch] += current.unsqueeze(0) * strength
                
        except Exception as e:
            print(f"📹 Frame injection error: {e}")
        finally:
            self.processing = False
    
    def start(self):
        """Start video stream listener."""
        if self.running:
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._listen_loop, daemon=True)
        self.thread.start()
        print("📹 Video stream started")
    
    def stop(self):
        """Stop video stream listener."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        print(f"📹 Video stream stopped. Processed {self.frame_count} frames, "
              f"dropped {self.dropped_frames}")
    
    def get_status(self) -> dict:
        """Get stream status."""
        return {
            'running': self.running,
            'frames_received': self.frame_count,
            'frames_dropped': self.dropped_frames,
            'drop_rate': self.dropped_frames / max(self.frame_count, 1),
            'last_frame_age': time.time() - self.last_frame_time if self.last_frame_time else None
        }