#!/usr/bin/env python3
"""
Vision UDP Streamer - Sends vision frames to brain via UDP.

This enables 640x480 vision without TCP buffer overflow!
Vision is processed in parallel thread on brain side.
"""

import socket
import struct
import numpy as np
import threading
import time
from typing import Optional, Tuple
import queue

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("⚠️  OpenCV not available - vision streaming disabled")


class VisionStreamer:
    """
    Streams vision frames to brain via UDP.
    Enables high-resolution vision without blocking.
    """
    
    def __init__(self, brain_ip: str, port: int = 10002, 
                 resolution: Tuple[int, int] = (640, 480),
                 fps_target: int = 15):
        """
        Args:
            brain_ip: IP address of brain server
            port: UDP port for vision stream
            resolution: Target resolution (width, height)
            fps_target: Target frames per second
        """
        self.brain_ip = brain_ip
        self.port = port
        self.resolution = resolution
        self.fps_target = fps_target
        
        # UDP socket
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        
        # Frame queue (latest frame only)
        self.frame_queue = queue.Queue(maxsize=1)
        
        # Threading
        self.running = False
        self.thread = None
        
        # Stats
        self.frames_sent = 0
        self.last_frame_time = 0
        
    def start(self):
        """Start the vision streaming thread."""
        if self.running:
            return
            
        self.running = True
        self.thread = threading.Thread(target=self._stream_loop, daemon=True)
        self.thread.start()
        print(f"📹 Vision streamer started: {self.resolution[0]}x{self.resolution[1]} @ {self.fps_target}fps")
        print(f"   Streaming to: {self.brain_ip}:{self.port}")
    
    def stop(self):
        """Stop the streaming thread."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)
        self.sock.close()
        print(f"📹 Vision streamer stopped. Sent {self.frames_sent} frames")
    
    def send_frame(self, frame: np.ndarray):
        """
        Queue a frame for streaming.
        Only keeps latest frame (drops old frames if not sent yet).
        """
        if not self.running:
            return
            
        # Drop old frame if queue is full
        try:
            self.frame_queue.put_nowait(frame)
        except queue.Full:
            # Replace old frame with new one
            try:
                self.frame_queue.get_nowait()
                self.frame_queue.put_nowait(frame)
            except:
                pass
    
    def _stream_loop(self):
        """Main streaming loop - sends frames via UDP."""
        frame_interval = 1.0 / self.fps_target
        
        while self.running:
            try:
                # Get frame from queue (with timeout)
                frame = self.frame_queue.get(timeout=0.1)
                
                # Enforce frame rate limit
                now = time.time()
                if now - self.last_frame_time < frame_interval:
                    time.sleep(frame_interval - (now - self.last_frame_time))
                
                # Send frame
                self._send_frame_udp(frame)
                self.last_frame_time = time.time()
                
            except queue.Empty:
                # No frame available
                continue
            except Exception as e:
                if self.frames_sent % 100 == 0:
                    print(f"⚠️  Vision stream error: {e}")
    
    def _send_frame_udp(self, frame: np.ndarray):
        """
        Send frame via UDP.
        
        For large frames, may need to chunk into multiple packets.
        For now, we'll compress as JPEG and send.
        """
        try:
            if CV2_AVAILABLE:
                # Encode as JPEG for compression
                _, jpeg_data = cv2.imencode('.jpg', frame, 
                    [cv2.IMWRITE_JPEG_QUALITY, 80])  # 80% quality
                jpeg_bytes = jpeg_data.tobytes()
            else:
                # No CV2 - send raw grayscale (much larger!)
                if len(frame.shape) == 3:
                    # Convert to grayscale if color
                    frame = np.mean(frame, axis=2).astype(np.uint8)
                jpeg_bytes = frame.flatten().tobytes()
            
            # Simple protocol for single-packet frames
            # For larger frames, would need chunking
            if len(jpeg_bytes) < 65000:  # UDP packet size limit
                # Single packet: frame_id, chunk_id=0, total_chunks=1, data_len, data
                packet = struct.pack('!IIHH', 
                    self.frames_sent,  # frame_id
                    0,                  # chunk_id
                    1,                  # total_chunks
                    len(jpeg_bytes)     # data_len
                ) + jpeg_bytes
                
                self.sock.sendto(packet, (self.brain_ip, self.port))
                self.frames_sent += 1
                
                # Log periodically
                if self.frames_sent % 30 == 0:
                    print(f"📹 Sent frame #{self.frames_sent} ({len(jpeg_bytes)} bytes)")
            else:
                # Need chunking for large frames
                # TODO: Implement multi-packet chunking
                print(f"⚠️  Frame too large for single packet: {len(jpeg_bytes)} bytes")
                
        except Exception as e:
            print(f"⚠️  Failed to send frame: {e}")


class VisionStreamAdapter:
    """
    Adapter to integrate vision streaming with existing brainstem.
    Intercepts vision data and streams it via UDP instead of TCP.
    """
    
    def __init__(self, brain_ip: str, enabled: bool = True):
        """
        Args:
            brain_ip: IP address of brain server
            enabled: Whether to enable UDP streaming
        """
        self.enabled = enabled
        self.streamer = None
        
        if enabled:
            self.streamer = VisionStreamer(brain_ip)
            self.streamer.start()
    
    def process_vision(self, vision_data: list) -> list:
        """
        Process vision data - stream via UDP and return empty list.
        
        Args:
            vision_data: Normalized vision pixels (flattened)
            
        Returns:
            Empty list to prevent sending via TCP
        """
        if not self.enabled or not self.streamer:
            # Fallback to TCP
            return vision_data
        
        # Reconstruct frame from flattened data
        # Assuming grayscale 640x480
        if len(vision_data) == 307200:  # 640x480
            frame = np.array(vision_data).reshape(480, 640)
            frame = (frame * 255).astype(np.uint8)
        elif len(vision_data) == 76800:  # 320x240
            frame = np.array(vision_data).reshape(240, 320)
            frame = (frame * 255).astype(np.uint8)
        elif len(vision_data) == 3072:  # 64x48
            frame = np.array(vision_data).reshape(48, 64)
            frame = (frame * 255).astype(np.uint8)
        else:
            # Unknown resolution
            return vision_data
        
        # Send via UDP
        self.streamer.send_frame(frame)
        
        # Return empty list so TCP doesn't send vision
        # Brain will get it via UDP instead!
        return []
    
    def stop(self):
        """Stop vision streaming."""
        if self.streamer:
            self.streamer.stop()


# Test standalone
if __name__ == "__main__":
    print("VISION STREAMER TEST")
    print("=" * 50)
    
    # Create streamer
    streamer = VisionStreamer("localhost", port=10002)
    streamer.start()
    
    # Send test frames
    print("\nSending test frames...")
    for i in range(10):
        # Create synthetic frame
        frame = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
        
        # Add some pattern
        frame[100:200, 100:200] = 255  # White square
        frame[300:400, 400:500] = 0    # Black square
        
        streamer.send_frame(frame)
        print(f"  Queued frame {i}")
        time.sleep(0.1)
    
    # Wait for sending
    time.sleep(2)
    
    streamer.stop()
    print("\n✅ Test complete!")