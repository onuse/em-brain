#!/usr/bin/env python3
"""
Example: HD Video Stream over UDP
Proper H.264 video streaming, not individual frames!
"""

import socket
import struct
import time
import subprocess
import threading

# Robot side - stream H.264 video over UDP
class HDVideoStreamer:
    def __init__(self, brain_ip: str, port: int = 10002):
        self.brain_ip = brain_ip
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        
    def start_streaming(self):
        """Stream H.264 video directly from camera to brain"""
        # Use hardware encoder on Raspberry Pi for efficiency
        # This streams compressed video at ~2-5 Mbps for HD
        cmd = [
            'ffmpeg',
            '-f', 'v4l2',                    # Video4Linux2 input
            '-video_size', '1920x1080',      # Full HD!
            '-framerate', '30',               # 30 FPS
            '-i', '/dev/video0',              # Camera device
            '-c:v', 'h264_omx',              # Hardware H.264 encoder
            '-b:v', '4M',                     # 4 Mbps bitrate
            '-f', 'h264',                     # Output format
            'udp://{}:{}?pkt_size=1316'.format(self.brain_ip, self.port)
        ]
        
        # Fire and forget - ffmpeg handles the streaming
        subprocess.run(cmd)
        
    def start_streaming_with_rtp(self):
        """Alternative: Use RTP for better video streaming"""
        # RTP is designed for real-time video/audio
        cmd = [
            'ffmpeg',
            '-f', 'v4l2',
            '-video_size', '1920x1080',
            '-framerate', '30',
            '-i', '/dev/video0',
            '-c:v', 'h264_omx',
            '-b:v', '4M',
            '-f', 'rtp',                     # RTP protocol
            'rtp://{}:{}'.format(self.brain_ip, self.port)
        ]
        subprocess.run(cmd)


# Brain side - decode H.264 stream
class BrainVideoReceiver:
    def __init__(self, port: int = 10002):
        self.port = port
        
    def start_receiving(self):
        """Decode H.264 stream using ffmpeg"""
        # Decode to raw frames for brain processing
        cmd = [
            'ffmpeg',
            '-i', 'udp://0.0.0.0:{}'.format(self.port),  # Listen on UDP
            '-f', 'rawvideo',                             # Output raw frames
            '-pix_fmt', 'gray',                           # Grayscale for brain
            '-video_size', '1920x1080',
            'pipe:1'                                       # Output to stdout
        ]
        
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE)
        
        # Read decoded frames
        frame_size = 1920 * 1080  # pixels
        while True:
            raw_frame = process.stdout.read(frame_size)
            if len(raw_frame) == frame_size:
                # Feed to brain - some frames may be corrupted/missing
                # That's OK! Brain learns robustness
                yield raw_frame
    
    def start_receiving_opencv(self):
        """Alternative: Use OpenCV for decoding"""
        import cv2
        
        # OpenCV can directly read from UDP stream
        cap = cv2.VideoCapture('udp://0.0.0.0:{}?overrun_nonfatal=1'.format(self.port))
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimal buffering
        
        while True:
            ret, frame = cap.read()
            if ret:
                # Convert to grayscale for brain
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                yield gray
            # Missing frames? Brain fills gaps with prediction!


# Example usage
if __name__ == "__main__":
    print("HD VIDEO STREAMING OVER UDP!")
    print("="*50)
    
    print("\nROBOT SIDE:")
    print("-----------")
    print("# Stream HD video from camera:")
    print("streamer = HDVideoStreamer('192.168.1.100')")
    print("streamer.start_streaming()  # Uses hardware H.264 encoder")
    print()
    print("# Or with RTP for better streaming:")
    print("streamer.start_streaming_with_rtp()")
    
    print("\nBRAIN SIDE:")
    print("-----------")
    print("# Decode H.264 stream:")
    print("receiver = BrainVideoReceiver(10002)")
    print("for frame in receiver.start_receiving():")
    print("    brain.process_vision(frame)  # 1920x1080 grayscale")
    print()
    print("# Or use OpenCV:")
    print("for frame in receiver.start_receiving_opencv():")
    print("    brain.process_vision(frame)")
    
    print("\nKEY BENEFITS:")
    print("✓ Full HD video stream (1920x1080 @ 30fps)")
    print("✓ Hardware H.264 encoding (efficient)")
    print("✓ ~4 Mbps bandwidth (vs 500 Mbps raw)")
    print("✓ Could go 4K with H.265")
    print("✓ Packet loss/jitter handled by codec")
    print("✓ Standard video streaming protocols")
    
    print("\nBIOLOGICAL REALISM:")
    print("• Eyes don't send every photon - they compress")
    print("• Retina does edge detection before sending")
    print("• Optic nerve is bandwidth-limited")
    print("• Brain reconstructs from compressed signal")
    print("• Missing data filled by prediction")
    
    print("\nNO MORE COMPROMISES:")
    print("64x48 was a TCP buffer workaround")
    print("With UDP video streams: FULL HD!")