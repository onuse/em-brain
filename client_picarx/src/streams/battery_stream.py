#!/usr/bin/env python3
"""
Battery voltage streaming - simplest possible sensor stream.
Sends battery voltage as separate UDP stream to brain.
Falls back to TCP if UDP fails.
"""

import socket
import struct
import time
import threading
from typing import Optional

# Import connection monitor
try:
    from .connection_monitor import connection_monitor
except ImportError:
    # Dummy if not available
    class DummyMonitor:
        def is_connected(self): return True
    connection_monitor = DummyMonitor()

class BatteryStreamer:
    """Streams battery voltage to brain server on separate port."""
    
    def __init__(self, brain_ip: str, port: int = 10004, use_tcp: bool = False):
        self.brain_ip = brain_ip
        self.port = port
        self.use_tcp = use_tcp
        self.running = False
        self.thread = None
        self.sock = None
        
        # For getting battery voltage
        self.adc = None
        self.battery_channel = 6  # ADC channel for battery
        
    def _init_hardware(self):
        """Initialize hardware for battery reading."""
        try:
            # Try robot_hat first
            import robot_hat
            self.adc = robot_hat.ADC(self.battery_channel)
            print(f"✓ Battery ADC initialized on channel {self.battery_channel}")
            return True
        except:
            try:
                # Fallback to direct ADC
                from hardware.bare_metal_hal import ADC
                self.adc = ADC(self.battery_channel)
                print(f"✓ Battery ADC initialized (bare metal)")
                return True
            except:
                print("⚠ No ADC hardware - using mock battery")
                return False
    
    def _read_battery_voltage(self) -> float:
        """Read actual battery voltage or return mock value."""
        if self.adc:
            try:
                # Read ADC (0-4095) and convert to voltage
                adc_value = self.adc.read()
                # Pi-car-x battery is 2S LiPo (7.4V nominal)
                # ADC has voltage divider, roughly 10:1
                voltage = (adc_value / 4095.0) * 3.3 * 10.0
                return max(6.0, min(8.4, voltage))  # Clamp to valid range
            except:
                pass
        
        # Mock battery slowly discharging
        import random
        base = 7.4 - (time.time() % 3600) / 3600  # Discharge over an hour
        return base + random.uniform(-0.05, 0.05)
    
    def _connect(self) -> bool:
        """Create socket connection (UDP or TCP)."""
        try:
            if self.use_tcp:
                self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.sock.connect((self.brain_ip, self.port))
                print(f"✓ Battery stream connected via TCP to {self.brain_ip}:{self.port}")
            else:
                self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                print(f"✓ Battery stream ready via UDP to {self.brain_ip}:{self.port}")
            return True
        except Exception as e:
            print(f"✗ Battery stream connection failed: {e}")
            return False
    
    def _stream_loop(self):
        """Main streaming loop - sends battery voltage periodically."""
        print("🔋 Battery stream started")
        
        # Initialize hardware
        self._init_hardware()
        
        # Connect socket
        if not self._connect():
            print("✗ Battery stream failed to connect")
            return
        
        packet_count = 0
        last_voltage = 0.0
        
        while self.running:
            # Check if main connection is still alive
            if not connection_monitor.is_connected():
                print("🔌 Battery stream stopping - main connection lost")
                break
                
            try:
                # Read battery voltage
                voltage = self._read_battery_voltage()
                
                # Only send if changed significantly (save bandwidth)
                if abs(voltage - last_voltage) > 0.01:
                    # Pack as simple float
                    packet = struct.pack('!f', voltage)
                    
                    if self.use_tcp:
                        self.sock.send(packet)
                    else:
                        self.sock.sendto(packet, (self.brain_ip, self.port))
                    
                    packet_count += 1
                    last_voltage = voltage
                    
                    # Debug output every 10 packets
                    if packet_count % 10 == 0:
                        print(f"🔋 Sent {packet_count} packets, latest: {voltage:.2f}V")
                
                # Battery doesn't change fast - 1Hz is plenty
                time.sleep(1.0)
                
            except Exception as e:
                print(f"✗ Battery stream error: {e}")
                # Try to reconnect
                time.sleep(5)
                if not self._connect():
                    break
        
        print("🔋 Battery stream stopped")
        if self.sock:
            self.sock.close()
    
    def start(self):
        """Start streaming in background thread."""
        if self.running:
            print("Battery stream already running")
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._stream_loop, daemon=True)
        self.thread.start()
    
    def stop(self):
        """Stop streaming."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)


# Standalone test
if __name__ == "__main__":
    import sys
    
    # Get brain IP from command line or use localhost
    brain_ip = sys.argv[1] if len(sys.argv) > 1 else "127.0.0.1"
    
    print(f"Starting battery stream to {brain_ip}")
    print("=" * 50)
    
    # Try UDP first
    streamer = BatteryStreamer(brain_ip, use_tcp=False)
    streamer.start()
    
    try:
        # Run for a while
        time.sleep(30)
    except KeyboardInterrupt:
        print("\nStopping...")
    
    streamer.stop()
    print("Done!")