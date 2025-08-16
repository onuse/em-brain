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
    
    def __init__(self, port=9999, enable_streams=True, telemetry_port=9998):
        self.port = port
        self.telemetry_port = telemetry_port
        self.brain = None  # Create brain after handshake
        self.telemetry = SimpleTelemetry()
        self.running = False
        self.client = None
        
        # Telemetry broadcasting
        self.telemetry_clients = []
        self.telemetry_thread = None
        
        # Initialize multi-stream support if available (brain will be set later)
        self.stream_manager = None
        if enable_streams and STREAMS_AVAILABLE:
            try:
                self.stream_manager = SensorFieldInjectionManager(None)  # Brain set after handshake
                print("✅ Multi-stream support initialized")
            except Exception as e:
                print(f"⚠️ Could not initialize streams: {e}")
        
    def start(self):
        """
        Start TCP server. Stream listeners will be started after handshake
        based on client capabilities.
        """
        self.running = True
        
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
        
        # Log connection details
        try:
            peer = client.getpeername()
            local = client.getsockname()
            print(f"   Connection established: {peer} -> {local}")
        except Exception as e:
            print(f"   ⚠️ Could not get socket details: {e}")
        
        # Handle handshake first
        if not self._handle_handshake(client):
            print("❌ Handshake failed, closing connection")
            client.close()
            self.client = None
            return
        
        # Give client time to process handshake response
        print("⏳ Waiting for client to process handshake...")
        time.sleep(0.5)  # 500ms for client to receive and process response
        
        client.settimeout(0.1)  # Non-blocking after handshake
        
        # Initialize tracking for heartbeats and error counts
        self.last_motivation = ""
        self.incomplete_header_count = 0  # Track incomplete headers to avoid spam
        self.bad_magic_count = 0  # Track bad magic bytes to avoid spam
        
        try:
            while self.running:
                # Receive sensor data
                try:
                    sensors = self.receive_sensors(client)
                    if sensors is None:
                        # Check if client is still connected
                        try:
                            # Try to peek at the socket
                            client.setblocking(False)
                            data = client.recv(1, socket.MSG_PEEK)
                            client.setblocking(True)
                            if not data:
                                # Connection closed cleanly
                                break
                        except BlockingIOError:
                            # No data available but socket is still open
                            pass
                        except (ConnectionResetError, ConnectionAbortedError, OSError):
                            # Connection lost
                            break
                        except:
                            # Any other error, assume disconnected
                            break
                        
                        time.sleep(0.001)
                        continue
                except ConnectionError as e:
                    # Stream corrupted (bad magic, etc) - disconnect
                    print(f"🔌 Disconnecting due to protocol error: {e}")
                    break
                
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
                    
        except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError):
            # Normal disconnection - client closed connection
            pass
        except OSError as e:
            # Windows-specific connection errors
            if e.errno not in [10054, 10053, 10038, 10035, 104]:
                # Only print if it's not a normal disconnect error or WOULDBLOCK
                print(f"Network error: {e}")
        except Exception as e:
            # Unexpected errors
            print(f"Unexpected error: {e}")
        finally:
            # Clean up client connection
            client.close()
            self.client = None
            
            # Stop all stream listeners when main connection drops
            if self.stream_manager:
                print("🛑 Stopping all stream listeners...")
                try:
                    self.stream_manager.stop_all()
                    print("✅ All stream listeners stopped")
                except Exception as e:
                    print(f"⚠️ Error stopping streams: {e}")
            
            # CRITICAL: Clean up brain to free GPU memory
            if self.brain is not None:
                print("🧹 Cleaning up brain (freeing GPU memory)...")
                try:
                    # Delete tensors to free CUDA memory
                    del self.brain.field
                    del self.brain.field_momentum
                    if hasattr(self.brain, 'dynamics'):
                        del self.brain.dynamics
                    if hasattr(self.brain, 'tensions'):
                        del self.brain.tensions
                    if hasattr(self.brain, 'prediction'):
                        del self.brain.prediction
                    if hasattr(self.brain, 'learning'):
                        del self.brain.learning
                    if hasattr(self.brain, 'motor'):
                        del self.brain.motor
                    
                    # Clear the brain reference
                    del self.brain
                    self.brain = None
                    
                    # Force CUDA memory cleanup
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                    
                    print("✅ Brain cleaned up, GPU memory freed")
                except Exception as e:
                    print(f"⚠️ Error cleaning up brain: {e}")
                    self.brain = None  # At least clear the reference
            
            print("Robot disconnected")
    
    def _handle_handshake(self, client: socket.socket) -> bool:
        """Handle initial handshake with robot."""
        try:
            print("🤝 Starting handshake...")
            client.settimeout(5.0)  # 5 second timeout for handshake
            
            # Read handshake header (13 bytes) - magic(4) + length(4) + type(1) + vector_length(4)
            print("   Waiting for handshake header (13 bytes)...")
            header = client.recv(13)
            if len(header) == 0:
                print("❌ Handshake failed: Client closed connection before sending handshake")
                return False
            if len(header) < 13:
                print(f"❌ Handshake failed: incomplete header (got {len(header)} bytes, expected 13)")
                print(f"   Received: {header.hex()}")
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
            print(f"   Waiting for handshake data ({data_size} bytes for {vector_length} floats)...")
            data = client.recv(data_size)
            if len(data) < data_size:
                print(f"❌ Handshake failed: incomplete data (got {len(data)}/{data_size} bytes)")
                return False
            
            # Parse capabilities (client sends floats in native byte order)
            capabilities = struct.unpack(f'{vector_length}f', data)
            print(f"   Received capabilities: {capabilities}")
            version = capabilities[0] if vector_length > 0 else 1.0
            sensory_dim = int(capabilities[1]) if vector_length > 1 else 16
            motor_dim = int(capabilities[2]) if vector_length > 2 else 5
            hardware_type = capabilities[3] if vector_length > 3 else 1.0
            capability_mask = int(capabilities[4]) if vector_length > 4 else 0
            
            # Extended capabilities for vision (if present)
            vision_width = int(capabilities[5]) if vector_length > 5 else 320
            vision_height = int(capabilities[6]) if vector_length > 6 else 240
            self.vision_resolution = (vision_width, vision_height)
            
            print(f"📡 Robot capabilities: v{version}, sensors={sensory_dim}, motors={motor_dim}")
            
            # Parse capability mask to understand what streams the client has
            has_vision = bool(capability_mask & 1)
            has_audio = bool(capability_mask & 2)
            
            if has_vision:
                print(f"   Vision: {vision_width}x{vision_height} resolution")
            if has_audio:
                print(f"   Audio: Enabled")
            
            # Create brain with negotiated dimensions if not already created
            if self.brain is None:
                print(f"🧠 Creating brain with negotiated dimensions: {sensory_dim} sensors, {motor_dim} motors")
                from .simple_factory import create_brain
                from ..brains.field.auto_config import get_optimal_config
                
                # Get config
                config = get_optimal_config()  # No parameter needed - auto-detects
                
                # Print hardware detection
                print(f"\n{'='*60}")
                print(f"HARDWARE DETECTION")
                print(f"{'='*60}")
                print(f"Device: {config['gpu_name']}")
                if config['device'].type == 'cuda':
                    print(f"Memory: {config['available_memory_gb']:.1f}/{config['total_memory_gb']:.1f} GB available")
                print(f"\nBRAIN CONFIGURATION")
                print(f"{'='*60}")
                print(f"Size: {config['spatial_size']}³×{config['channels']}")
                print(f"Parameters: {config['parameters']:,}")
                print(f"Memory usage: {config['memory_gb']:.2f} GB")
                print(f"Estimated speed: {config['estimated_hz']:,} Hz")
                print(f"{'='*60}\n")
                
                # Create brain with correct dimensions
                from ..brains.field.truly_minimal_brain import TrulyMinimalBrain
                self.brain = TrulyMinimalBrain(
                    sensory_dim=sensory_dim,
                    motor_dim=motor_dim,
                    spatial_size=config['spatial_size'],
                    channels=config['channels'],
                    device=config['device'],
                    quiet_mode=True  # We already printed info above
                )
                
                # Update stream manager with the new brain
                if self.stream_manager:
                    self.stream_manager.brain = self.brain
                
                # Load brain state if requested
                if hasattr(self, 'brain_state_to_load') and self.brain_state_to_load:
                    if self.brain.load(self.brain_state_to_load):
                        print(f"✅ Loaded brain state from {self.brain_state_to_load}")
                    else:
                        print(f"❌ Failed to load {self.brain_state_to_load}, starting fresh")
            
            # Start stream listeners based on CLIENT capabilities
            if self.stream_manager and self.brain:
                started = []
                
                # Basic sensors (always try these)
                if self.stream_manager.start_battery_injector():
                    started.append("battery(10004)")
                if self.stream_manager.start_ultrasonic_injector():
                    started.append("ultrasonic(10003)")
                
                # Vision stream - only if client has vision
                if has_vision:
                    if self.stream_manager.start_vision_injector(port=10002, resolution=self.vision_resolution):
                        started.append(f"vision(10002@{vision_width}x{vision_height})")
                
                # Audio stream - only if client has audio  
                if has_audio:
                    if self.stream_manager.start_audio_injector():
                        started.append("audio(10006)")
                
                if started:
                    print(f"📡 Stream listeners started: {', '.join(started)}")
            
            # Send response using client's protocol format
            # Client expects: [brain_version, accepted_sensory, accepted_action, gpu_available, brain_capabilities]
            import torch
            gpu_available = 1.0 if torch.cuda.is_available() else 0.0
            
            response_capabilities = [
                1.0,  # Brain version
                float(sensory_dim),  # Accept client's sensory dimensions
                float(motor_dim),    # Accept client's motor dimensions
                gpu_available,       # GPU available (1.0 = yes, 0.0 = no)
                15.0                 # Brain capabilities (1=visual, 2=audio, 4=manipulation, 8=multi-agent)
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
            response_msg = response_header + vector_data
            
            print(f"📤 Preparing handshake response: {len(response_msg)} bytes")
            print(f"   Header: magic=0x{0x524F424F:08X}, msg_len={message_length}, type=2, vec_len={len(response_capabilities)}")
            print(f"   Capabilities being sent: {response_capabilities}")
            print(f"   Full response bytes: {response_msg.hex()[:60]}..." if len(response_msg) > 30 else f"   Full response: {response_msg.hex()}")
            
            # Use sendall to ensure all bytes are sent
            try:
                # Log socket state before sending
                print(f"   Socket state before send:")
                print(f"     - Connected: {client.getpeername()}")
                print(f"     - Timeout: {client.gettimeout()}")
                
                # First disable Nagle algorithm for immediate sending
                client.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                print(f"     - TCP_NODELAY set (Nagle disabled)")
                
                # Try different sending methods
                print(f"   Attempting to send {len(response_msg)} bytes...")
                
                # Send the response
                bytes_sent = 0
                try:
                    # Use sendall to ensure all bytes are sent
                    client.sendall(response_msg)
                    bytes_sent = len(response_msg)
                    print(f"   ✅ sendall() succeeded - {bytes_sent} bytes sent")
                    
                    # DO NOT use SO_LINGER with timeout 0 - it causes immediate close!
                    # Instead, ensure data is sent by disabling buffering
                    client.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                    
                    # Give TCP stack time to actually send the data
                    time.sleep(0.01)  # 10ms to ensure data goes out
                    
                except Exception as e:
                    print(f"   ❌ sendall() failed: {e}")
                    raise
                
                # Verify send buffer is empty
                import select
                readable, writable, exceptional = select.select([], [client], [], 0)
                if writable:
                    print(f"     - Socket is writable (send buffer has space)")
                else:
                    print(f"     - Socket NOT writable (send buffer may be full)")
                
                print(f"   ✅ Total {bytes_sent} bytes sent successfully")
                
                # Debug: Verify client is still connected and data was sent
                try:
                    peer = client.getpeername()
                    print(f"   ✅ Client still connected at {peer}")
                    
                    # Try to receive 1 byte with MSG_PEEK to check if client sent anything
                    import select
                    readable, _, _ = select.select([client], [], [], 0.1)
                    if readable:
                        try:
                            # Set non-blocking for peek
                            client.setblocking(False)
                            peek_data = client.recv(1, socket.MSG_PEEK)
                            client.setblocking(True)
                            if peek_data:
                                print(f"   🔍 Client has data ready (peeked: 0x{peek_data[0]:02X})")
                            else:
                                print(f"   ⚠️ Socket readable but no data - connection may be closing")
                        except (BlockingIOError, socket.error):
                            # Would block - no data available yet
                            print(f"   ⏳ No immediate data from client (normal)")
                    else:
                        print(f"   ⏳ Waiting for client to process response (normal)")
                        
                except Exception as e:
                    print(f"   ⚠️ Post-send check failed: {e}")
                
            except socket.error as e:
                print(f"   ❌ Socket error sending response: {e}")
                if hasattr(e, 'errno'):
                    print(f"   Error code: {e.errno}")
                return False
            except Exception as e:
                print(f"   ❌ Unexpected error sending response: {e}")
                import traceback
                traceback.print_exc()
                return False
            
            print(f"✅ Handshake complete - sensors={sensory_dim}, motors={motor_dim}")
            return True
            
        except socket.timeout:
            print("❌ Handshake timeout: Client didn't send handshake within 5 seconds")
            print("   Possible causes:")
            print("   - Client crashed or hung")
            print("   - Network connectivity issues")
            print("   - Client using wrong protocol version")
            return False
        except struct.error as e:
            print(f"❌ Handshake protocol error: {e}")
            print("   The data format doesn't match expected protocol")
            return False
        except Exception as e:
            print(f"❌ Unexpected handshake error: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def receive_sensors(self, client: socket.socket) -> Optional[list]:
        """Receive sensor data from robot."""
        try:
            # Read header (13 bytes) - magic(4) + length(4) + type(1) + vector_length(4)
            header = client.recv(13)
            if len(header) < 13:
                # Only print warning occasionally to avoid spam
                if len(header) > 0:  # Got partial data
                    self.incomplete_header_count += 1
                    if self.incomplete_header_count <= 3:  # Only print first 3 times
                        print(f"⚠️ Incomplete sensor header: {len(header)} bytes")
                    elif self.incomplete_header_count == 10:  # And once more after 10
                        print(f"⚠️ Continuing to receive incomplete headers (suppressing further messages)")
                return None
            
            # Parse header - client uses network byte order (!)
            magic, message_length, msg_type, vector_length = struct.unpack('!IIBI', header)
            
            # Validate magic
            if magic != 0x524F424F:  # 'ROBO'
                # Only print error for first few occurrences to avoid spam
                if not hasattr(self, 'bad_magic_count'):
                    self.bad_magic_count = 0
                self.bad_magic_count += 1
                
                if self.bad_magic_count <= 3:
                    print(f"❌ Invalid magic in sensor message: 0x{magic:08X}")
                    if self.bad_magic_count == 3:
                        print("   (suppressing further magic errors - connection likely corrupted)")
                
                # Bad magic usually means connection is corrupted, trigger disconnect
                raise ConnectionError("Invalid magic bytes - stream corrupted")
            
            if msg_type != 0:  # MSG_SENSORY_INPUT in client protocol
                print(f"❌ Wrong message type: {msg_type}, expected 0 (SENSORY_INPUT)")
                return None
            
            # Read sensor values
            data_size = vector_length * 4
            data = client.recv(data_size)
            if len(data) < data_size:
                print(f"❌ Incomplete sensor data: {len(data)}/{data_size} bytes")
                return None
            
            # Unpack floats (native byte order for data)
            sensors = list(struct.unpack(f'{vector_length}f', data))
            
            # Only log first successful receive, then periodically
            if not hasattr(self, 'sensor_receive_count'):
                self.sensor_receive_count = 0
            self.sensor_receive_count += 1
            
            if self.sensor_receive_count == 1:
                print(f"✅ Receiving sensor data: {len(sensors)} values per message")
            elif self.sensor_receive_count % 1000 == 0:  # Every 1000 messages (~50 seconds at 20Hz)
                print(f"📊 Processed {self.sensor_receive_count} sensor messages")
            
            # Reset error counters on successful receive
            self.incomplete_header_count = 0
            self.bad_magic_count = 0
            
            return sensors
            
        except socket.timeout:
            return None
        except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError) as e:
            # Connection closed by client - this is normal when client disconnects
            # Don't spam the log, just return None to trigger cleanup
            return None
        except OSError as e:
            # Windows-specific connection errors (10054, 10053, etc.)
            if e.errno in [10054, 10053, 10038, 10035, 104]:  # Common connection/non-blocking errors
                # Silent - these are normal
                return None
            else:
                print(f"❌ Network error: {e}")
                return None
        except Exception as e:
            print(f"❌ Unexpected error receiving sensors: {e}")
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
def run_brain_service(port=9999):
    """Run the brain service."""
    service = SimpleBrainService(port)
    
    try:
        service.start()
    except KeyboardInterrupt:
        print("\nShutting down...")
        service.stop()