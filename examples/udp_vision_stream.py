#!/usr/bin/env python3
"""
Example: MJPEG Video Stream over UDP
Perfect for Pi Zero 2 W - hardware JPEG, no H.264 encoder needed!
"""

import socket
import struct
import time
import cv2
import numpy as np

# Robot side - stream MJPEG video over UDP
class MJPEGVideoStreamer:
    def __init__(self, brain_ip: str, port: int = 10002):
        self.brain_ip = brain_ip
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.frame_num = 0
        
        # Configure camera for MJPEG
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 15)  # 15fps is plenty for Pi Zero
        
    def start_streaming(self):
        """Stream MJPEG frames - each frame independent!"""
        quality = 60  # JPEG quality (lower = smaller packets)
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                continue
            
            # Convert to grayscale (optional, saves bandwidth)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Encode as JPEG (hardware accelerated on Pi Camera!)
            _, jpeg = cv2.imencode('.jpg', gray, 
                                  [cv2.IMWRITE_JPEG_QUALITY, quality])
            
            # Simple packet header
            timestamp = int(time.time() * 1e6)  # microseconds
            header = struct.pack('!QII', 
                timestamp,
                self.frame_num,
                len(jpeg)
            )
            
            packet = header + jpeg.tobytes()
            
            # Send if fits in UDP packet (usually does at 640x480)
            if len(packet) < 65000:
                self.sock.sendto(packet, (self.brain_ip, self.port))
                self.frame_num += 1
            else:
                # Reduce quality if too big
                quality = max(30, quality - 10)
            
            # ~15 fps rate limiting
            time.sleep(0.066)


# Brain side - decode MJPEG stream  
class BrainMJPEGReceiver:
    def __init__(self, port: int = 10002):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(('', port))
        self.sock.settimeout(0.001)  # Non-blocking
        self.last_frame_num = -1
        
    def receive_frames(self):
        """Receive and decode MJPEG frames"""
        while True:
            try:
                data, addr = self.sock.recvfrom(65536)
                
                # Parse header
                timestamp, frame_num, jpeg_size = struct.unpack('!QII', data[:16])
                jpeg_data = data[16:16+jpeg_size]
                
                # Detect dropped frames (good for learning!)
                if self.last_frame_num >= 0:
                    dropped = frame_num - self.last_frame_num - 1
                    if dropped > 0:
                        # Brain learns to predict through gaps!
                        pass
                
                self.last_frame_num = frame_num
                
                # Decode JPEG to numpy array
                frame = cv2.imdecode(
                    np.frombuffer(jpeg_data, np.uint8),
                    cv2.IMREAD_GRAYSCALE
                )
                
                if frame is not None:
                    yield {
                        'timestamp': timestamp,
                        'frame_num': frame_num,
                        'image': frame  # 640x480 grayscale
                    }
                    
            except socket.timeout:
                # No frame? Brain uses prediction
                yield None


# Example usage
if __name__ == "__main__":
    print("MJPEG VIDEO STREAMING OVER UDP!")
    print("="*50)
    
    print("\nROBOT SIDE (Pi Zero 2 W):")
    print("-------------------------")
    print("streamer = MJPEGVideoStreamer('192.168.1.100')")
    print("streamer.start_streaming()  # Hardware JPEG, no H.264 needed!")
    
    print("\nBRAIN SIDE:")
    print("-----------")
    print("receiver = BrainMJPEGReceiver(10002)")
    print("for frame_data in receiver.receive_frames():")
    print("    if frame_data:")
    print("        brain.process_vision(frame_data['image'])  # 640x480")
    print("    else:")
    print("        # Missing frame - brain predicts!")
    
    print("\nWHY MJPEG FOR PI ZERO 2 W:")
    print("✓ NO H.264 encoder on Pi Zero 2 W")
    print("✓ Hardware JPEG in Pi Camera Module")
    print("✓ Each frame independent (perfect for UDP)")
    print("✓ ~30KB per frame at 640x480")
    print("✓ 3.6 Mbps bandwidth (WiFi handles easily)")
    
    print("\nBIOLOGICAL REALISM:")
    print("• Frame drops = temporal prediction pressure")
    print("• Variable framerate = real-world timing")
    print("• Compression artifacts = noisy sensors")
    print("• Async arrival = biological messiness")
    
    print("\nPROGRESSIVE ENHANCEMENT:")
    print("Start:  320x240 @ 15fps = 1.2 Mbps")
    print("Good:   640x480 @ 15fps = 3.6 Mbps")  
    print("Dream:  1280x720 @ 10fps = 6.4 Mbps")
    
    print("\nTHE KEY INSIGHT:")
    print("MJPEG's frame independence matches UDP's")
    print("packet independence. Lose a packet? Lose")
    print("a frame. Next frame still works perfectly!")