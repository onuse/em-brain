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
    
    def __init__(self, port=9999, enable_streams=True, telemetry_port=9998, debug=False, 
                 brain_type='unified', target='balanced', auto_save_interval=300,
                 save_on_exit=True, brain_state_path=None):
        self.port = port
        self.telemetry_port = telemetry_port
        self.brain = None  # Create brain after handshake
        self.brain_type = brain_type  # Store brain type for creation
        self.target = target  # Store optimization target
        self.telemetry = SimpleTelemetry()
        self.running = False
        self.client = None
        self.debug = debug  # Debug logging flag
        
        # Auto-save configuration
        self.auto_save_interval = auto_save_interval  # Save every N seconds (0 = disabled)
        self.save_on_exit = save_on_exit  # Save when disconnecting
        self.brain_state_path = brain_state_path  # Path to load/save brain state
        self.last_save_time = 0
        self.last_save_cycle = 0
        
        # Telemetry broadcasting
        self.telemetry_clients = []
        self.telemetry_thread = None
        
        # Initialize multi-stream support if available (brain will be set later)
        self.enable_streams = enable_streams
        self.stream_manager = None  # Will be created after brain initialization
        
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
        # Also set SO_REUSEPORT if available (helps with socket reuse)
        if hasattr(socket, 'SO_REUSEPORT'):
            self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        # Bind to all interfaces (0.0.0.0) to accept external connections
        self.server.bind(('0.0.0.0', self.port))
        
        # Get and display the actual IP address
        import socket as s
        hostname = s.gethostname()
        try:
            # Get local IP
            local_ip = s.gethostbyname(hostname)
            print(f"📡 Accepting connections on:")
            print(f"   - localhost:{self.port}")
            print(f"   - {local_ip}:{self.port}")
            print(f"   - 0.0.0.0:{self.port} (all interfaces)")
        except:
            print(f"📡 Accepting connections on 0.0.0.0:{self.port}")
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
        
        # Enable TCP keepalive to prevent Windows from closing the connection
        client.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        
        # Windows-specific keepalive settings
        if hasattr(socket, 'TCP_KEEPIDLE'):
            client.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, 1)
        if hasattr(socket, 'TCP_KEEPINTVL'):
            client.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, 1)
        if hasattr(socket, 'TCP_KEEPCNT'):
            client.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPCNT, 3)
        
        # Log connection details
        try:
            peer = client.getpeername()
            local = client.getsockname()
            if self.debug:
                print(f"   Connection established: {peer} -> {local}")
        except Exception as e:
            if self.debug:
                print(f"   ⚠️ Could not get socket details: {e}")
        
        # Handle handshake first
        if not self._handle_handshake(client):
            print("❌ Handshake failed, closing connection")
            client.close()
            self.client = None
            return
        
        # Give client time to process handshake and start sensor loop
        if self.debug:
            print("⏳ Waiting for client to start sensor stream...")
        time.sleep(1.0)  # 1 second for client to process handshake and start sending
        
        # Be patient for first sensor data
        client.settimeout(5.0)  # Long timeout for first sensor packet
        first_sensor_received = False
        
        # Initialize tracking for heartbeats and error counts
        self.last_motivation = ""
        self.incomplete_header_count = 0  # Track incomplete headers to avoid spam
        self.bad_magic_count = 0  # Track bad magic bytes to avoid spam
        
        try:
            while self.running:
                # Receive sensor data
                try:
                    if self.debug:
                        print(f"DEBUG: Waiting for sensor data (timeout={client.gettimeout()}s)...")
                    sensors = self.receive_sensors(client)
                    if self.debug:
                        print(f"DEBUG: receive_sensors returned: {sensors is not None}")
                    
                    # After first successful receive, reduce timeout
                    if sensors and not first_sensor_received:
                        first_sensor_received = True
                        client.settimeout(0.1)  # Now we can use short timeout
                        if self.debug:
                            print("✅ First sensor data received, switching to fast mode")
                    
                    if sensors is None:
                        # Don't check connection immediately - just continue waiting
                        # The recv() in receive_sensors already has a timeout
                        if not first_sensor_received:
                            # Be patient for first sensor packet
                            if self.debug:
                                print("DEBUG: Still waiting for first sensor packet...")
                            time.sleep(0.1)  # Wait 100ms before trying again
                            continue
                        else:
                            # After first packet, we can be less patient
                            time.sleep(0.001)
                            continue
                except ConnectionError as e:
                    # Stream corrupted (bad magic, etc) - disconnect
                    print(f"🔌 Disconnecting due to protocol error: {e}")
                    break
                
                # Process with brain
                try:
                    if self.debug:
                        print(f"DEBUG: Processing {len(sensors)} sensors with brain...")
                    
                    # For first few cycles, warn that it might take time
                    if self.telemetry.cycles < 5:
                        print(f"  ⏳ Processing cycle {self.telemetry.cycles + 1} (may take 30-60s initially)...")
                    
                    motors, brain_telemetry = self.brain.process(sensors)
                    
                    if self.debug:
                        print(f"DEBUG: Brain returned {len(motors)} motor commands")
                except Exception as e:
                    print(f"❌ Brain processing error: {e}")
                    import traceback
                    traceback.print_exc()
                    break
                
                # Update telemetry
                try:
                    metrics = self.telemetry.update(brain_telemetry)
                except Exception as e:
                    print(f"❌ Telemetry error: {e}")
                    metrics = {}
                
                # Broadcast telemetry
                try:
                    self._broadcast_telemetry(sensors, motors, brain_telemetry, metrics)
                except Exception as e:
                    print(f"⚠️ Telemetry broadcast error: {e}")
                
                # Send motor commands
                try:
                    if self.debug:
                        print(f"DEBUG: Sending {len(motors)} motor commands...")
                    self.send_motors(client, motors)
                    if self.debug:
                        print(f"DEBUG: Motor commands sent")
                except Exception as e:
                    print(f"❌ Error sending motors: {e}")
                    import traceback
                    traceback.print_exc()
                    break
                
                # Auto-save check
                if self.auto_save_interval > 0 and self.brain:
                    current_time = time.time()
                    if current_time - self.last_save_time >= self.auto_save_interval:
                        self._auto_save_brain()
                        self.last_save_time = current_time
                
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
            # Save brain state on exit if configured
            if self.save_on_exit and self.brain and hasattr(self.brain, 'save_state'):
                print("💾 Saving brain state before disconnecting...")
                self._save_brain_on_exit()
            
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
            if self.debug:
                print("🤝 Starting handshake...")
            client.settimeout(5.0)  # 5 second timeout for handshake
            
            # Read handshake header (13 bytes) - magic(4) + length(4) + type(1) + vector_length(4)
            if self.debug:
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
            if self.debug:
                print(f"   Waiting for handshake data ({data_size} bytes for {vector_length} floats)...")
            data = client.recv(data_size)
            if len(data) < data_size:
                print(f"❌ Handshake failed: incomplete data (got {len(data)}/{data_size} bytes)")
                return False
            
            # Parse capabilities (client sends floats in native byte order)
            capabilities = struct.unpack(f'{vector_length}f', data)
            if self.debug:
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
                
                # Print brain type being created
                print(f"\n{'='*60}")
                print(f"BRAIN CREATION")
                print(f"{'='*60}")
                print(f"Brain Type: {self.brain_type.upper()}")
                print(f"Target: {self.target}")
                
                # Create brain using factory with selected type
                self.brain = create_brain(
                    brain_type=self.brain_type,
                    target=self.target,
                    quiet=False  # Let factory print its info
                )
                
                # For Critical Mass Brain, it already handles sensor/motor dims internally
                # For UnifiedFieldBrain, we need to check compatibility
                if self.brain_type == 'unified':
                    # UnifiedFieldBrain needs dimension compatibility check
                    if hasattr(self.brain, 'sensory_dim') and self.brain.sensory_dim != sensory_dim:
                        print(f"⚠️ Adjusting sensory dimensions: {self.brain.sensory_dim} -> {sensory_dim}")
                    if hasattr(self.brain, 'motor_dim') and self.brain.motor_dim != motor_dim:
                        print(f"⚠️ Adjusting motor dimensions: {self.brain.motor_dim} -> {motor_dim}")
                
                print(f"{'='*60}\n")
                
                # Initialize stream manager now that brain exists
                if self.enable_streams and STREAMS_AVAILABLE and self.stream_manager is None:
                    try:
                        from ..streams.field_injection_threads import SensorFieldInjectionManager
                        
                        # Check if brain has field tensor for injection
                        if hasattr(self.brain, 'field'):
                            self.stream_manager = SensorFieldInjectionManager(self.brain)
                            print("✅ Multi-stream support initialized (vision/audio → field injection)")
                        else:
                            print("ℹ️ Brain type doesn't support direct field injection")
                    except Exception as e:
                        # Not critical - streams are optional
                        if self.debug:
                            print(f"Stream initialization: {e}")
                
                # Load brain state if specified
                if self.brain_state_path and hasattr(self.brain, 'load_state'):
                    import os
                    if os.path.exists(self.brain_state_path):
                        if self.brain.load_state(self.brain_state_path):
                            print(f"✅ Loaded brain state from {self.brain_state_path}")
                        else:
                            print(f"❌ Failed to load {self.brain_state_path}, starting fresh")
                    else:
                        print(f"ℹ️ No saved brain state found at {self.brain_state_path}, starting fresh")
                
                # Initialize auto-save
                if self.auto_save_interval > 0:
                    self.last_save_time = time.time()
                    print(f"⏰ Auto-save enabled every {self.auto_save_interval} seconds")
            
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
            # Use native byte order for floats (not network order)
            if self.debug:
                print(f"   DEBUG: About to pack {len(response_capabilities)} floats: {response_capabilities}")
            pack_format = str(len(response_capabilities)) + 'f'  # Creates '5f' for 5 floats
            try:
                vector_data = struct.pack(pack_format, *response_capabilities)
                if self.debug:
                    print(f"   DEBUG: Packed {len(vector_data)} bytes successfully")
            except Exception as e:
                print(f"   ERROR: struct.pack failed: {e}")
                return False
            message_length = 1 + 4 + len(vector_data)  # type + vector_length + data
            response_header = struct.pack('!IIBI', 
                0x524F424F,  # Magic ('ROBO')
                message_length,  # Length
                2,  # MSG_HANDSHAKE
                len(response_capabilities)  # Vector length
            )
            response_msg = response_header + vector_data
            
            if self.debug:
                print(f"📤 Preparing handshake response: {len(response_msg)} bytes")
                print(f"   Header: magic=0x{0x524F424F:08X}, msg_len={message_length}, type=2, vec_len={len(response_capabilities)}")
                print(f"   Capabilities being sent: {response_capabilities}")
                print(f"   Full response ({len(response_msg)} bytes): {response_msg.hex()}")
            
            if self.debug:
                # Verify it matches expected
                expected = bytes.fromhex("524f424f0000001902000000050000803f000040410000c0400000803f00007041")
                if response_msg == expected:
                    print(f"   ✅ Response matches expected format")
                else:
                    print(f"   ⚠️ Response differs from expected!")
                    print(f"   Expected: {expected.hex()}")
                    print(f"   Got:      {response_msg.hex()}")
            
            # Send with verification
            try:
                if self.debug:
                    print(f"   Sending {len(response_msg)} bytes...")
                
                # Disable Nagle for immediate send
                client.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                
                # Send and verify
                bytes_sent = client.send(response_msg)
                if bytes_sent < len(response_msg):
                    if self.debug:
                        print(f"   ⚠️ Only sent {bytes_sent}/{len(response_msg)} bytes!")
                    remaining = response_msg[bytes_sent:]
                    client.sendall(remaining)
                    if self.debug:
                        print(f"   Sent remaining {len(remaining)} bytes")
                else:
                    if self.debug:
                        print(f"   ✅ Sent all {bytes_sent} bytes")
                
                # Force a flush by sending empty data (sometimes helps)
                try:
                    client.send(b'')
                except:
                    pass
                
                
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
    
    def _auto_save_brain(self):
        """Auto-save brain state periodically."""
        if not self.brain or not hasattr(self.brain, 'save_state'):
            return
        
        # Generate filename with timestamp if no path specified
        if not self.brain_state_path:
            import os
            os.makedirs("brain_states", exist_ok=True)
            self.brain_state_path = f"brain_states/{self.brain_type}_autosave.brain"
        
        # Save state
        if self.brain.save_state(self.brain_state_path):
            cycles = self.telemetry.cycles
            delta_cycles = cycles - self.last_save_cycle
            print(f"💾 Auto-saved brain state (cycle {cycles}, +{delta_cycles} since last save)")
            self.last_save_cycle = cycles
        else:
            print(f"⚠️ Failed to auto-save brain state")
    
    def _save_brain_on_exit(self):
        """Save brain state when exiting."""
        if not self.brain or not hasattr(self.brain, 'save_state'):
            return
        
        # Generate exit filename
        import os
        from datetime import datetime
        
        os.makedirs("brain_states", exist_ok=True)
        
        # Use configured path or generate one
        if self.brain_state_path:
            # Add _exit suffix if using auto-save path
            base, ext = os.path.splitext(self.brain_state_path)
            exit_path = f"{base}_exit{ext}"
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            exit_path = f"brain_states/{self.brain_type}_{timestamp}_exit.brain"
        
        # Save state
        if self.brain.save_state(exit_path):
            cycles = self.telemetry.cycles
            print(f"✅ Brain state saved on exit: {exit_path}")
            print(f"   Total cycles: {cycles}")
            if hasattr(self.brain, 'metrics'):
                concepts = self.brain.metrics.get('concepts_formed', 0)
                print(f"   Concepts formed: {concepts}")
                if hasattr(self.brain, 'enhanced_metrics'):
                    chains = self.brain.enhanced_metrics.get('causal_chains_learned', 0)
                    meanings = self.brain.enhanced_metrics.get('semantic_meanings', 0)
                    print(f"   Causal chains: {chains}")
                    print(f"   Semantic meanings: {meanings}")
        else:
            print(f"⚠️ Failed to save brain state on exit")


# Simple entry point
def run_brain_service(port=9999):
    """Run the brain service."""
    service = SimpleBrainService(port)
    
    try:
        service.start()
    except KeyboardInterrupt:
        print("\nShutting down...")
        service.stop()