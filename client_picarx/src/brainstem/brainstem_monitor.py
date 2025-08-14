#!/usr/bin/env python3
"""
Brainstem Monitoring Server

Provides telemetry and metrics from the brainstem for debugging and monitoring.
Runs on port 9997 by default (brain monitor uses 9998).
"""

import socket
import json
import time
import threading
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict


@dataclass
class BrainstemMetrics:
    """Current brainstem metrics snapshot."""
    # Lifecycle
    uptime_seconds: float = 0.0
    total_cycles: int = 0
    
    # Connection status
    brain_connected: bool = False
    brain_timeouts: int = 0
    reconnect_attempts: int = 0
    
    # Safety & reflexes
    reflex_active: bool = False
    collision_triggers: int = 0
    cliff_triggers: int = 0
    battery_triggers: int = 0
    emergency_stops: int = 0
    
    # Current sensor values
    distance_cm: float = 999.0
    battery_voltage: float = 0.0
    grayscale_values: list = None
    
    # Performance metrics
    avg_cycle_time_ms: float = 0.0
    max_cycle_time_ms: float = 0.0
    sensor_read_time_ms: float = 0.0
    brain_comm_time_ms: float = 0.0
    action_exec_time_ms: float = 0.0
    
    # Motor state
    left_motor_speed: float = 0.0
    right_motor_speed: float = 0.0
    steering_angle: float = 0.0
    camera_angle: float = 0.0
    
    def __post_init__(self):
        if self.grayscale_values is None:
            self.grayscale_values = [0, 0, 0]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with metadata."""
        return {
            'timestamp': time.time(),
            'type': 'brainstem_telemetry',
            **asdict(self)
        }


class BrainstemMonitor:
    """
    Monitoring server for brainstem telemetry.
    
    Provides:
    - TCP server on configurable port (default 9997)
    - JSON telemetry endpoint
    - Metrics history buffer
    - Thread-safe metric updates
    """
    
    def __init__(self, port: int = 9997, host: str = '0.0.0.0'):
        """Initialize monitoring server."""
        self.port = port
        self.host = host
        self.metrics = BrainstemMetrics()
        self.metrics_lock = threading.Lock()
        
        # Server state
        self.server_socket = None
        self.server_thread = None
        self.running = False
        
        # Performance tracking
        self.start_time = time.time()
        self.cycle_times = []  # Rolling buffer of last 100 cycle times
        self.max_history = 100
        
        print(f"📊 Brainstem monitor initialized (port {port})")
    
    def start(self):
        """Start the monitoring server."""
        if self.running:
            return
            
        self.running = True
        self.server_thread = threading.Thread(target=self._run_server, daemon=True)
        self.server_thread.start()
        print(f"📊 Monitoring server started on {self.host}:{self.port}")
    
    def stop(self):
        """Stop the monitoring server."""
        self.running = False
        if self.server_socket:
            try:
                self.server_socket.close()
            except:
                pass
        print("📊 Monitoring server stopped")
    
    def _run_server(self):
        """Run the TCP server loop."""
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(5)
            self.server_socket.settimeout(1.0)  # Allow periodic checks
            
            while self.running:
                try:
                    conn, addr = self.server_socket.accept()
                    # Handle connection in separate thread
                    threading.Thread(
                        target=self._handle_client,
                        args=(conn, addr),
                        daemon=True
                    ).start()
                except socket.timeout:
                    continue
                except Exception as e:
                    if self.running:
                        print(f"⚠️ Monitor server error: {e}")
                        
        except Exception as e:
            print(f"❌ Failed to start monitor server: {e}")
        finally:
            if self.server_socket:
                self.server_socket.close()
    
    def _handle_client(self, conn: socket.socket, addr: tuple):
        """Handle a monitoring client connection."""
        try:
            conn.settimeout(5.0)
            
            # Read request (if any)
            try:
                request = conn.recv(1024).decode().strip()
            except:
                request = ""
            
            # Get current metrics
            with self.metrics_lock:
                response_data = self.metrics.to_dict()
                # Add computed values
                response_data['uptime_seconds'] = time.time() - self.start_time
            
            # Send JSON response
            response = json.dumps(response_data, indent=2) + "\n"
            conn.sendall(response.encode())
            
        except Exception as e:
            print(f"⚠️ Error handling monitor client: {e}")
        finally:
            conn.close()
    
    # === Metric Update Methods ===
    
    def update_cycle_time(self, cycle_time_ms: float):
        """Update cycle timing metrics."""
        with self.metrics_lock:
            self.cycle_times.append(cycle_time_ms)
            if len(self.cycle_times) > self.max_history:
                self.cycle_times.pop(0)
            
            self.metrics.avg_cycle_time_ms = sum(self.cycle_times) / len(self.cycle_times)
            self.metrics.max_cycle_time_ms = max(self.cycle_times)
            self.metrics.total_cycles += 1
    
    def update_sensors(self, distance_cm: float, battery_v: float, grayscale: list):
        """Update current sensor readings."""
        with self.metrics_lock:
            self.metrics.distance_cm = distance_cm
            self.metrics.battery_voltage = battery_v
            self.metrics.grayscale_values = grayscale.copy() if grayscale else [0, 0, 0]
    
    def update_motors(self, left: float, right: float, steering: float, camera: float):
        """Update motor state."""
        with self.metrics_lock:
            self.metrics.left_motor_speed = left
            self.metrics.right_motor_speed = right
            self.metrics.steering_angle = steering
            self.metrics.camera_angle = camera
    
    def update_connection(self, connected: bool, timeout: bool = False):
        """Update brain connection status."""
        with self.metrics_lock:
            self.metrics.brain_connected = connected
            if timeout:
                self.metrics.brain_timeouts += 1
            if not connected:
                self.metrics.reconnect_attempts += 1
    
    def trigger_reflex(self, reflex_type: str):
        """Record a reflex trigger."""
        with self.metrics_lock:
            self.metrics.reflex_active = True
            if reflex_type == 'collision':
                self.metrics.collision_triggers += 1
            elif reflex_type == 'cliff':
                self.metrics.cliff_triggers += 1
            elif reflex_type == 'battery':
                self.metrics.battery_triggers += 1
            elif reflex_type == 'emergency':
                self.metrics.emergency_stops += 1
    
    def clear_reflex(self):
        """Clear reflex state."""
        with self.metrics_lock:
            self.metrics.reflex_active = False
    
    def update_timing(self, sensor_ms: float = 0, comm_ms: float = 0, action_ms: float = 0):
        """Update component timing metrics."""
        with self.metrics_lock:
            if sensor_ms > 0:
                self.metrics.sensor_read_time_ms = sensor_ms
            if comm_ms > 0:
                self.metrics.brain_comm_time_ms = comm_ms
            if action_ms > 0:
                self.metrics.action_exec_time_ms = action_ms
    
    def get_summary(self) -> str:
        """Get a text summary of current metrics."""
        with self.metrics_lock:
            m = self.metrics
            uptime = time.time() - self.start_time
            
            return f"""
Brainstem Telemetry Summary
===========================
Uptime: {uptime:.1f}s
Cycles: {m.total_cycles} ({m.avg_cycle_time_ms:.1f}ms avg)
Brain: {'Connected' if m.brain_connected else f'Disconnected (timeouts: {m.brain_timeouts})'}
Safety: {'REFLEX ACTIVE' if m.reflex_active else 'Normal'}
  Collisions: {m.collision_triggers}
  Cliffs: {m.cliff_triggers}
  Battery: {m.battery_triggers}
  E-Stops: {m.emergency_stops}
Sensors:
  Distance: {m.distance_cm:.1f}cm
  Battery: {m.battery_voltage:.2f}V
  Grayscale: {m.grayscale_values}
Performance:
  Sensor read: {m.sensor_read_time_ms:.1f}ms
  Brain comm: {m.brain_comm_time_ms:.1f}ms
  Action exec: {m.action_exec_time_ms:.1f}ms
"""


# === Standalone Test ===

def test_monitor():
    """Test the monitoring server standalone."""
    monitor = BrainstemMonitor(port=9997)
    monitor.start()
    
    # Simulate some metrics
    import random
    for i in range(100):
        monitor.update_cycle_time(random.uniform(40, 60))
        monitor.update_sensors(
            distance_cm=random.uniform(5, 100),
            battery_v=random.uniform(6.5, 7.5),
            grayscale=[random.randint(0, 4095) for _ in range(3)]
        )
        if i % 20 == 0:
            monitor.trigger_reflex('collision')
        time.sleep(0.1)
    
    print(monitor.get_summary())
    
    # Test connection
    print("\nTesting connection...")
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect(('localhost', 9997))
    sock.send(b"status\n")
    response = sock.recv(4096).decode()
    print("Received:", response[:200], "...")
    sock.close()
    
    monitor.stop()


if __name__ == "__main__":
    test_monitor()