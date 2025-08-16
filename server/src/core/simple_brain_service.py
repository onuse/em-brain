"""
Simple Brain Service - TCP server for one brain.

Just handles connections and processes sensor/motor data.
No complex abstractions.
"""

import socket
import struct
import threading
import time
import json
from typing import Optional, Tuple

from .simple_factory import create_brain
from .simple_telemetry import SimpleTelemetry
from .simple_errors import ConnectionError, ProcessingError

# Import multi-stream support
try:
    from ..streams.field_injection_threads import SensorFieldInjectionManager
    STREAMS_AVAILABLE = True
except ImportError:
    STREAMS_AVAILABLE = False
    print("⚠️ Multi-stream support not available")


class SimpleBrainService:
    """Minimal TCP server for brain."""
    
    def __init__(self, port=9999, brain_config='balanced', enable_streams=True, telemetry_port=9998):
        self.port = port
        self.telemetry_port = telemetry_port
        self.brain = create_brain(brain_config)
        self.telemetry = SimpleTelemetry()
        self.running = False
        self.client = None
        
        # Telemetry broadcasting
        self.telemetry_clients = []
        self.telemetry_thread = None
        
        # Initialize multi-stream support if available
        self.stream_manager = None
        if enable_streams and STREAMS_AVAILABLE:
            try:
                self.stream_manager = SensorFieldInjectionManager(self.brain)
                print("✅ Multi-stream support initialized")
            except Exception as e:
                print(f"⚠️ Could not initialize streams: {e}")
        
    def start(self, enable_vision=True, enable_audio=False, vision_resolution=(320, 240)):
        """
        Start TCP server and optional stream listeners.
        
        Args:
            enable_vision: Start vision stream on port 10002
            enable_audio: Start audio stream on port 10006
            vision_resolution: Resolution for vision stream
        """
        self.running = True
        
        # Start multi-stream listeners if available
        if self.stream_manager:
            started = []
            
            # Always start basic sensors (battery, ultrasonic)
            if self.stream_manager.start_battery_injector():
                started.append("battery(10004)")
            if self.stream_manager.start_ultrasonic_injector():
                started.append("ultrasonic(10003)")
            
            # Start vision if requested
            if enable_vision:
                if self.stream_manager.start_vision_injector(port=10002, resolution=vision_resolution):
                    started.append(f"vision(10002@{vision_resolution[0]}x{vision_resolution[1]})")
            
            # Start audio if requested
            if enable_audio:
                if self.stream_manager.start_audio_injector():
                    started.append("audio(10006)")
            
            if started:
                print(f"📡 Stream listeners started: {', '.join(started)}")
        
        # Start telemetry server
        self._start_telemetry_server()
        
        # Start main TCP server
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server.bind(('0.0.0.0', self.port))
        self.server.listen(1)
        
        print(f"🧠 Brain service listening on port {self.port}")
        
        # Accept connections
        while self.running:
            try:
                self.server.settimeout(1.0)
                client, addr = self.server.accept()
                print(f"✅ Robot connected from {addr}")
                self.handle_client(client)
            except socket.timeout:
                continue
            except Exception as e:
                print(f"Connection error: {e}")
    
    def handle_client(self, client: socket.socket):
        """Handle robot connection."""
        self.client = client
        
        # Handle handshake first
        if not self._handle_handshake(client):
            print("❌ Handshake failed, closing connection")
            client.close()
            self.client = None
            return
        
        client.settimeout(0.1)  # Non-blocking after handshake
        
        # Initialize tracking for heartbeats
        self.last_motivation = ""
        
        try:
            while self.running:
                # Receive sensor data
                sensors = self.receive_sensors(client)
                if sensors is None:
                    time.sleep(0.001)
                    continue
                
                # Process with brain
                motors, brain_telemetry = self.brain.process(sensors)
                
                # Update telemetry
                metrics = self.telemetry.update(brain_telemetry)
                
                # Broadcast telemetry
                self._broadcast_telemetry(sensors, motors, brain_telemetry, metrics)
                
                # Send motor commands
                self.send_motors(client, motors)
                
                # Periodic status with field evolution
                if self.telemetry.cycles % 100 == 0:
                    field_energy = brain_telemetry.get('energy', 0)
                    field_variance = brain_telemetry.get('variance', 0)
                    comfort = brain_telemetry.get('comfort', 0)
                    
                    # Detect interesting changes
                    if not hasattr(self, 'last_energy'):
                        self.last_energy = field_energy
                        self.last_comfort = comfort
                    
                    energy_delta = field_energy - self.last_energy
                    comfort_delta = comfort - self.last_comfort
                    
                    # Status line
                    print(f"  Cycle {self.telemetry.cycles}: {metrics['motivation']}, "
                          f"{metrics['hz']:.0f} Hz, Energy: {field_energy:.3f} "
                          f"(Δ{energy_delta:+.3f}), Comfort: {comfort:.2f} "
                          f"(Δ{comfort_delta:+.2f})")
                    
                    # Alert on significant changes
                    if abs(energy_delta) > 0.1:
                        print(f"    ⚡ ENERGY SPIKE: Field is {('agitated' if energy_delta > 0 else 'calming')}")
                    if abs(comfort_delta) > 0.5:
                        print(f"    😰 COMFORT SHIFT: Brain is {('relaxing' if comfort_delta > 0 else 'stressed')}")
                    if 'STARVED' in metrics['motivation'] and self.last_motivation != 'STARVED':
                        print(f"    🍽️ STARVATION: Brain needs input!")
                    if 'ACTIVE' in metrics['motivation']:
                        print(f"    🎯 ACTIVE: Brain is engaged and learning!")
                    
                    self.last_energy = field_energy
                    self.last_comfort = comfort
                    self.last_motivation = metrics['motivation']
                
                # More frequent heartbeat when something interesting happens
                if self.telemetry.cycles % 20 == 0:
                    if brain_telemetry.get('exploring', False):
                        print(f"    🔍 Exploring... (cycle {self.telemetry.cycles})")
                    
        except Exception as e:
            print(f"Client error: {e}")
        finally:
            client.close()
            self.client = None
            print("Robot disconnected")
    
    def _handle_handshake(self, client: socket.socket) -> bool:
        """Handle initial handshake with robot."""
        try:
            print("🤝 Starting handshake...")
            client.settimeout(5.0)  # 5 second timeout for handshake
            
            # Read handshake header (13 bytes) - magic(4) + length(4) + type(1) + vector_length(4)
            header = client.recv(13)
            if len(header) < 13:
                print(f"❌ Handshake failed: incomplete header (got {len(header)} bytes)")
                return False
            
            # Parse header - client uses network byte order (!)
            magic, message_length, msg_type, vector_length = struct.unpack('!IIBI', header)
            
            # Check magic (0x524F424F = 'ROBO')
            if magic != 0x524F424F:
                print(f"❌ Handshake failed: wrong magic (got 0x{magic:08X}, expected 0x524F424F 'ROBO')")
                return False
            
            if msg_type != 2:  # HANDSHAKE type in client protocol
                print(f"❌ Handshake failed: wrong type (got {msg_type}, expected 2)")
                return False
            
            # Read handshake data
            data_size = vector_length * 4
            data = client.recv(data_size)
            if len(data) < data_size:
                print(f"❌ Handshake failed: incomplete data")
                return False
            
            # Parse capabilities (client sends floats in native byte order)
            capabilities = struct.unpack(f'{vector_length}f', data)
            version = capabilities[0] if vector_length > 0 else 1.0
            sensory_dim = int(capabilities[1]) if vector_length > 1 else 16
            motor_dim = int(capabilities[2]) if vector_length > 2 else 5
            
            print(f"📡 Robot capabilities: v{version}, sensors={sensory_dim}, motors={motor_dim}")
            
            # Send response using client's protocol format
            response_capabilities = [
                1.0,  # Version
                float(self.brain.sensory_dim),
                float(self.brain.motor_dim)
            ]
            
            # Encode response in client's format
            vector_data = struct.pack(f'{len(response_capabilities)}f', *response_capabilities)
            message_length = 1 + 4 + len(vector_data)  # type + vector_length + data
            response_header = struct.pack('!IIBI', 
                0x524F424F,  # Magic ('ROBO')
                message_length,  # Length
                2,  # MSG_HANDSHAKE
                len(response_capabilities)  # Vector length
            )
            client.sendall(response_header + vector_data)
            
            print(f"✅ Handshake complete - sensors={sensory_dim}, motors={motor_dim}")
            return True
            
        except socket.timeout:
            print("❌ Handshake timeout")
            return False
        except Exception as e:
            print(f"❌ Handshake error: {e}")
            return False
    
    def receive_sensors(self, client: socket.socket) -> Optional[list]:
        """Receive sensor data from robot."""
        try:
            # Read header (13 bytes) - magic(4) + length(4) + type(1) + vector_length(4)
            header = client.recv(13)
            if len(header) < 13:
                return None
            
            # Parse header - client uses network byte order (!)
            magic, message_length, msg_type, vector_length = struct.unpack('!IIBI', header)
            
            # Validate magic
            if magic != 0x524F424F:  # 'ROBO'
                return None
            
            if msg_type != 0:  # MSG_SENSORY_INPUT in client protocol
                return None
            
            # Read sensor values
            data_size = vector_length * 4
            data = client.recv(data_size)
            if len(data) < data_size:
                return None
            
            # Unpack floats (native byte order for data)
            sensors = list(struct.unpack(f'{vector_length}f', data))
            return sensors
            
        except socket.timeout:
            return None
        except Exception:
            return None
    
    def send_motors(self, client: socket.socket, motors: list):
        """Send motor commands to robot."""
        try:
            # Encode in client's protocol format
            vector_data = struct.pack(f'{len(motors)}f', *motors)
            message_length = 1 + 4 + len(vector_data)  # type + vector_length + data
            header = struct.pack('!IIBI', 
                0x524F424F,  # Magic ('ROBO')
                message_length,  # Length
                1,  # MSG_ACTION_OUTPUT in client protocol
                len(motors)  # Vector length
            )
            
            # Send
            client.sendall(header + vector_data)
        except Exception:
            pass  # Robot might have disconnected
    
    def stop(self):
        """Stop service."""
        self.running = False
        if self.client:
            self.client.close()
        self.server.close()
        
        # Close telemetry connections
        for client in self.telemetry_clients:
            try:
                client.close()
            except:
                pass
        if hasattr(self, 'telemetry_server'):
            self.telemetry_server.close()
        
        print(f"Service stopped. {self.telemetry.get_summary()}")
    
    def _start_telemetry_server(self):
        """Start telemetry broadcasting server."""
        try:
            self.telemetry_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.telemetry_server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.telemetry_server.bind(('0.0.0.0', self.telemetry_port))
            self.telemetry_server.listen(5)
            self.telemetry_server.settimeout(0.1)
            
            # Start accept thread
            self.telemetry_thread = threading.Thread(
                target=self._telemetry_accept_loop, 
                daemon=True
            )
            self.telemetry_thread.start()
            
            print(f"📊 Telemetry broadcasting on port {self.telemetry_port}")
        except Exception as e:
            print(f"⚠️ Could not start telemetry server: {e}")
    
    def _telemetry_accept_loop(self):
        """Accept telemetry client connections."""
        while self.running:
            try:
                client, addr = self.telemetry_server.accept()
                self.telemetry_clients.append(client)
                print(f"📊 Telemetry client connected from {addr}")
            except socket.timeout:
                continue
            except:
                break
    
    def _broadcast_telemetry(self, sensors, motors, brain_telemetry, metrics):
        """Broadcast telemetry to all connected monitors."""
        if not self.telemetry_clients:
            return
        
        # Combine all telemetry
        telemetry = {
            **brain_telemetry,
            **metrics,
            'sensors': sensors[:16] if len(sensors) > 16 else sensors,
            'motors': motors[:5] if len(motors) > 5 else motors
        }
        
        # Send to all clients
        message = json.dumps(telemetry) + '\n'
        dead_clients = []
        
        for client in self.telemetry_clients:
            try:
                client.send(message.encode('utf-8'))
            except:
                dead_clients.append(client)
        
        # Remove dead clients
        for client in dead_clients:
            self.telemetry_clients.remove(client)
            try:
                client.close()
            except:
                pass


# Simple entry point
def run_brain_service(port=9999, config='balanced'):
    """Run the brain service."""
    service = SimpleBrainService(port, config)
    
    try:
        service.start()
    except KeyboardInterrupt:
        print("\nShutting down...")
        service.stop()