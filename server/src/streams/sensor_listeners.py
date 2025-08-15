#!/usr/bin/env python3
"""
Sensor stream listeners for the brain server.
Each sensor gets its own port and listener thread.
"""

import socket
import struct
import threading
import time
import numpy as np
from typing import Optional, Callable
import cv2


class BatteryListener:
    """Listens for battery voltage updates on UDP/TCP port."""
    
    def __init__(self, port: int = 10004, use_tcp: bool = False, 
                 callback: Optional[Callable] = None):
        self.port = port
        self.use_tcp = use_tcp
        self.callback = callback
        self.running = False
        self.thread = None
        self.latest_voltage = 7.4  # Default voltage
        self.last_update = 0
        self.packet_count = 0
        
    def _listen_loop(self):
        """Main listening loop."""
        try:
            if self.use_tcp:
                # TCP server
                server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                server_sock.bind(('', self.port))
                server_sock.listen(1)
                print(f"🔋 Battery listener on TCP port {self.port}")
                
                conn, addr = server_sock.accept()
                print(f"🔋 Battery stream connected from {addr}")
                
                while self.running:
                    data = conn.recv(4)
                    if len(data) == 4:
                        voltage = struct.unpack('!f', data)[0]
                        self._process_voltage(voltage)
                    elif len(data) == 0:
                        print("🔋 Battery stream disconnected")
                        break
            else:
                # UDP server
                sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                sock.bind(('', self.port))
                sock.settimeout(0.1)  # Non-blocking with timeout
                print(f"🔋 Battery listener on UDP port {self.port}")
                
                while self.running:
                    try:
                        data, addr = sock.recvfrom(1024)
                        if len(data) == 4:
                            voltage = struct.unpack('!f', data)[0]
                            self._process_voltage(voltage)
                    except socket.timeout:
                        continue
                        
        except Exception as e:
            print(f"✗ Battery listener error: {e}")
        finally:
            print("🔋 Battery listener stopped")
    
    def _process_voltage(self, voltage: float):
        """Process received voltage value."""
        self.latest_voltage = voltage
        self.last_update = time.time()
        self.packet_count += 1
        
        # Callback to brain if provided
        if self.callback:
            self.callback('battery', voltage)
        
        # Log every 10th packet
        if self.packet_count % 10 == 0:
            print(f"🔋 Battery: {voltage:.2f}V (packet #{self.packet_count})")
        
        # Warn on low battery
        if voltage < 6.5:
            print(f"⚠️  LOW BATTERY: {voltage:.2f}V")
    
    def start(self):
        """Start listening in background thread."""
        if self.running:
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._listen_loop, daemon=True)
        self.thread.start()
    
    def stop(self):
        """Stop listening."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)
    
    def get_latest(self) -> tuple[float, float]:
        """Get latest voltage and age in seconds."""
        age = time.time() - self.last_update if self.last_update > 0 else float('inf')
        return self.latest_voltage, age


class UltrasonicListener:
    """Listens for ultrasonic distance updates."""
    
    def __init__(self, port: int = 10003, callback: Optional[Callable] = None):
        self.port = port
        self.callback = callback
        self.running = False
        self.thread = None
        self.latest_distance = 100.0  # cm
        self.last_update = 0
        
    def _listen_loop(self):
        """UDP listener for ultrasonic data."""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(('', self.port))
            sock.settimeout(0.1)
            print(f"📏 Ultrasonic listener on UDP port {self.port}")
            
            while self.running:
                try:
                    data, addr = sock.recvfrom(1024)
                    if len(data) == 4:
                        distance = struct.unpack('!f', data)[0]
                        self.latest_distance = distance
                        self.last_update = time.time()
                        
                        if self.callback:
                            self.callback('ultrasonic', distance)
                        
                        # Warn on obstacles
                        if distance < 10:
                            print(f"⚠️  OBSTACLE: {distance:.1f}cm")
                            
                except socket.timeout:
                    continue
                    
        except Exception as e:
            print(f"✗ Ultrasonic listener error: {e}")
    
    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._listen_loop, daemon=True)
        self.thread.start()
    
    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)


class VideoListener:
    """Listens for MJPEG video stream."""
    
    def __init__(self, port: int = 10002, callback: Optional[Callable] = None):
        self.port = port
        self.callback = callback
        self.running = False
        self.thread = None
        self.latest_frame = None
        self.last_frame_time = 0
        self.frame_count = 0
        
    def _listen_loop(self):
        """UDP listener for MJPEG frames."""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(('', self.port))
            sock.settimeout(0.1)
            print(f"📹 Video listener on UDP port {self.port}")
            
            while self.running:
                try:
                    data, addr = sock.recvfrom(65536)
                    
                    if len(data) > 8:
                        # Parse header
                        timestamp = struct.unpack('!Q', data[:8])[0]
                        jpeg_data = data[8:]
                        
                        # Decode JPEG
                        frame = cv2.imdecode(
                            np.frombuffer(jpeg_data, np.uint8),
                            cv2.IMREAD_GRAYSCALE
                        )
                        
                        if frame is not None:
                            self.latest_frame = frame
                            self.last_frame_time = time.time()
                            self.frame_count += 1
                            
                            if self.callback:
                                self.callback('video', frame)
                            
                            # Log every 30th frame (roughly once per second at 30fps)
                            if self.frame_count % 30 == 0:
                                print(f"📹 Frame #{self.frame_count}, shape: {frame.shape}")
                                
                except socket.timeout:
                    continue
                    
        except Exception as e:
            print(f"✗ Video listener error: {e}")
    
    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._listen_loop, daemon=True)
        self.thread.start()
    
    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)


class StreamManager:
    """Manages all sensor stream listeners."""
    
    def __init__(self, brain=None):
        self.brain = brain
        self.listeners = {}
        
    def start_all(self):
        """Start all sensor listeners."""
        # Battery on port 10004
        self.listeners['battery'] = BatteryListener(
            port=10004,
            callback=self._sensor_callback
        )
        self.listeners['battery'].start()
        
        # Ultrasonic on port 10003
        self.listeners['ultrasonic'] = UltrasonicListener(
            port=10003,
            callback=self._sensor_callback
        )
        self.listeners['ultrasonic'].start()
        
        # Video on port 10002
        self.listeners['video'] = VideoListener(
            port=10002,
            callback=self._sensor_callback
        )
        self.listeners['video'].start()
        
        print("✓ All sensor stream listeners started")
    
    def stop_all(self):
        """Stop all listeners."""
        for name, listener in self.listeners.items():
            print(f"Stopping {name} listener...")
            listener.stop()
        print("✓ All listeners stopped")
    
    def _sensor_callback(self, sensor_type: str, data):
        """Called when sensor data arrives."""
        if self.brain:
            # Update brain with async sensor data
            if sensor_type == 'battery':
                self.brain.battery_voltage = data
            elif sensor_type == 'ultrasonic':
                self.brain.obstacle_distance = data
            elif sensor_type == 'video':
                self.brain.latest_vision = data
        # If no brain, listeners just track latest values


# Standalone test
if __name__ == "__main__":
    print("Starting sensor stream listeners")
    print("=" * 50)
    
    manager = StreamManager()
    manager.start_all()
    
    print("\nListening for sensor streams...")
    print("Start the robot streams to see data flow!")
    print("Press Ctrl+C to stop\n")
    
    try:
        while True:
            time.sleep(1)
            
            # Show latest values
            if 'battery' in manager.listeners:
                voltage, age = manager.listeners['battery'].get_latest()
                if age < 10:
                    print(f"Battery: {voltage:.2f}V (age: {age:.1f}s)")
                    
    except KeyboardInterrupt:
        print("\nStopping...")
    
    manager.stop_all()
    print("Done!")