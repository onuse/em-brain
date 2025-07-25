"""
Minimal TCP Server

Simple TCP server that hosts the minimal brain and accepts connections
from robot clients (Pi Zero brainstems, simulators, etc.)
"""

import socket
import threading
import time
import datetime
from typing import Dict, Any, Optional, Callable
import uuid

from .protocol import MessageProtocol, MessageProtocolError
from .error_codes import BrainErrorCode, create_brain_error, log_brain_error
from .sensor_buffer import get_sensor_buffer
from ..brain_factory import BrainFactory


class MinimalTCPServer:
    """
    TCP server for the minimal brain.
    
    Accepts connections from robot clients and processes sensory input
    through the brain to generate action outputs.
    """
    
    def __init__(self, brain: BrainFactory, host: str = '0.0.0.0', port: int = 9999):
        """
        Initialize the TCP server.
        
        Args:
            brain: The minimal brain instance
            host: Server host address
            port: Server port
        """
        self.brain = brain
        self.host = host
        self.port = port
        
        # Networking
        self.server_socket = None
        self.running = False
        
        # Protocol and client management
        self.protocol = MessageProtocol()
        self.clients = {}  # client_id -> client_info
        self.client_counter = 0
        
        # Experience storage tracking per client
        self.client_experiences = {}  # client_id -> previous experience data
        
        # Sensor buffer for decoupling brain from socket I/O
        self.sensor_buffer = get_sensor_buffer()
        
        # Performance tracking
        self.total_requests = 0
        self.total_clients_served = 0
        self.start_time = None
        
        print(f"🌐 MinimalTCPServer initialized")
        print(f"   Host: {host}:{port}")
        print(f"   Brain: {brain}")
    
    def start(self):
        """Start the TCP server."""
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"\n[{timestamp}] 🚀 Starting TCP server on {self.host}:{self.port}")
        
        try:
            # Create server socket
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(5)  # Allow up to 5 pending connections
            
            self.running = True
            self.start_time = time.time()
            
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print(f"[{timestamp}] ✅ Server listening on {self.host}:{self.port}")
            print("   Waiting for robot clients to connect...")
            
            # Main server loop
            while self.running:
                try:
                    client_socket, client_address = self.server_socket.accept()
                    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    print(f"[{timestamp}] 🤖 New client connected from {client_address}")
                    
                    # Handle client in separate thread
                    client_thread = threading.Thread(
                        target=self._handle_client,
                        args=(client_socket, client_address),
                        daemon=True
                    )
                    client_thread.start()
                    
                except socket.error as e:
                    if self.running:  # Only log if not shutting down
                        print(f"⚠️  Server socket error: {e}")
                        break
        
        except Exception as e:
            print(f"❌ Failed to start server: {e}")
            self.running = False
        
        finally:
            self._cleanup()
    
    def stop(self):
        """Stop the TCP server."""
        print("🛑 Stopping TCP server...")
        self.running = False
        
        if self.server_socket:
            try:
                self.server_socket.close()
            except:
                pass
    
    def _handle_client(self, client_socket: socket.socket, client_address: tuple):
        """Handle individual client connection."""
        
        # Generate unique client ID
        client_id = f"client_{self.client_counter}_{uuid.uuid4().hex[:8]}"
        self.client_counter += 1
        
        # Track client info
        client_info = {
            'id': client_id,
            'address': client_address,
            'connected_at': time.time(),
            'requests_served': 0,
            'last_activity': time.time(),
            'capabilities': None
        }
        self.clients[client_id] = client_info
        self.total_clients_served += 1
        
        print(f"📋 Client {client_id} registered from {client_address}")
        
        try:
            # Set socket timeout for client communication
            # Use longer timeout for biological timescale tests (5+ minutes consolidation)
            client_socket.settimeout(300.0)  # 5 minute timeout
            
            # Main client communication loop
            while self.running:
                try:
                    # Receive message from client  
                    # Use longer receive timeout for biological tests
                    msg_type, vector_data = self.protocol.receive_message(client_socket, timeout=60.0)
                    client_info['last_activity'] = time.time()
                    
                    # Process message based on type
                    if msg_type == MessageProtocol.MSG_HANDSHAKE:
                        response = self._handle_handshake(client_info, vector_data)
                    elif msg_type == MessageProtocol.MSG_SENSORY_INPUT:
                        response = self._handle_sensory_input(client_info, vector_data)
                    else:
                        error = create_brain_error(
                            BrainErrorCode.UNKNOWN_MESSAGE_TYPE,
                            f"Unknown message type {msg_type}",
                            context={'message_type': msg_type}
                        )
                        log_brain_error(error, client_id)
                        response = self.protocol.encode_error(error.error_code.value)
                    
                    # Send response
                    if not self.protocol.send_message(client_socket, response):
                        print(f"💔 Failed to send response to {client_id}")
                        break
                    
                    client_info['requests_served'] += 1
                    self.total_requests += 1
                
                except TimeoutError:
                    print(f"⏰ Client {client_id} timed out")
                    break
                except MessageProtocolError as e:
                    error = create_brain_error(
                        BrainErrorCode.PROTOCOL_ERROR,
                        f"Protocol error: {e}",
                        context={'protocol_error': str(e)},
                        exception=e
                    )
                    log_brain_error(error, client_id)
                    error_response = self.protocol.encode_error(error.error_code.value)
                    self.protocol.send_message(client_socket, error_response)
                except ConnectionError:
                    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    print(f"[{timestamp}] 💔 Client {client_id} disconnected")
                    break
                except Exception as e:
                    print(f"❌ Unexpected error handling {client_id}: {e}")
                    break
        
        finally:
            # Cleanup client connection
            try:
                client_socket.close()
            except:
                pass
            
            if client_id in self.clients:
                del self.clients[client_id]
            
            # Cleanup client experience data
            if client_id in self.client_experiences:
                del self.client_experiences[client_id]
            
            # Cleanup sensor buffer data
            self.sensor_buffer.clear_client_data(client_id)
            
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print(f"[{timestamp}] 👋 Client {client_id} disconnected (served {client_info['requests_served']} requests)")
    
    def _handle_handshake(self, client_info: dict, capabilities: list) -> bytes:
        """Handle handshake message from client with dynamic dimension negotiation."""
        print(f"🤝 Handshake from {client_info['id']}: {len(capabilities)} capabilities")
        
        # Parse client capabilities 
        client_version = capabilities[0] if len(capabilities) > 0 else 1.0
        client_sensory_dim = int(capabilities[1]) if len(capabilities) > 1 else 16
        client_action_dim = int(capabilities[2]) if len(capabilities) > 2 else 4
        client_hardware_type = capabilities[3] if len(capabilities) > 3 else 0.0
        client_capabilities_mask = int(capabilities[4]) if len(capabilities) > 4 else 0
        
        print(f"   Client v{client_version}: {client_sensory_dim}D sensors → {client_action_dim}D actions")
        print(f"   Hardware type: {client_hardware_type}, Capabilities: {client_capabilities_mask}")
        
        # Decode capability flags
        has_visual = bool(client_capabilities_mask & 1)
        has_audio = bool(client_capabilities_mask & 2)
        has_manipulation = bool(client_capabilities_mask & 4)
        print(f"   Features: visual={has_visual}, audio={has_audio}, manipulation={has_manipulation}")
        
        # Get current server configuration
        server_sensory_dim = self.brain.brain.expected_sensory_dim
        server_action_dim = self.brain.brain.expected_motor_dim
        
        # Dynamic dimension negotiation
        negotiated_sensory_dim = client_sensory_dim
        negotiated_action_dim = client_action_dim
        
        # If dimensions differ, adapt the brain
        if client_sensory_dim != server_sensory_dim or client_action_dim != server_action_dim:
            print(f"   🔄 Dynamic dimension adaptation needed:")
            print(f"      Server expects: {server_sensory_dim}D sensors → {server_action_dim}D actions")
            print(f"      Client provides: {client_sensory_dim}D sensors → {client_action_dim}D actions")
            
            # Store negotiated dimensions for this client
            client_info['sensory_dim'] = negotiated_sensory_dim
            client_info['action_dim'] = negotiated_action_dim
            client_info['needs_adaptation'] = True
            
            # For UnifiedFieldBrain, we'll handle dimension adaptation in process_sensory_input
            print(f"   ✅ Will adapt dimensions per request")
        else:
            print(f"   ✅ Dimensions match (no adaptation needed)")
            client_info['needs_adaptation'] = False
        
        # Store all client capabilities
        client_info['capabilities'] = {
            'version': client_version,
            'sensory_dim': negotiated_sensory_dim,
            'action_dim': negotiated_action_dim,
            'hardware_type': client_hardware_type,
            'capabilities_mask': client_capabilities_mask,
            'has_visual': has_visual,
            'has_audio': has_audio,
            'has_manipulation': has_manipulation
        }
        
        # Server capabilities
        brain_capabilities = 15  # 1+2+4+8 = all capabilities
        
        # Respond with negotiated dimensions
        server_capabilities = [
            4.0,  # Brain version
            float(negotiated_sensory_dim),  # Accepted sensory vector size
            float(negotiated_action_dim),   # Accepted action vector size
            1.0,  # GPU acceleration available
            float(brain_capabilities)  # Brain capabilities mask
        ]
        
        return self.protocol.encode_handshake(server_capabilities)
    
    def _handle_sensory_input(self, client_info: dict, sensory_vector: list) -> bytes:
        """Handle sensory input from client and generate action response with dynamic dimensions."""
        
        # Validate sensory input
        if len(sensory_vector) == 0:
            error = create_brain_error(
                BrainErrorCode.EMPTY_SENSORY_INPUT,
                context={'client_id': client_info['id']}
            )
            log_brain_error(error, client_info['id'])
            return self.protocol.encode_error(error.error_code.value)
        
        if len(sensory_vector) > 64:  # Increased limit for flexibility
            error = create_brain_error(
                BrainErrorCode.SENSORY_INPUT_TOO_LARGE,
                f"Sensory input size {len(sensory_vector)} exceeds limit of 64",
                context={'input_size': len(sensory_vector), 'limit': 64}
            )
            log_brain_error(error, client_info['id'])
            return self.protocol.encode_error(error.error_code.value)
        
        try:
            client_id = client_info['id']
            
            # Add sensor input to buffer (step toward decoupled brain loop)
            self.sensor_buffer.add_sensor_input(client_id, sensory_vector)
            
            # Adapt dimensions if needed
            adapted_sensory_vector = sensory_vector
            if client_info.get('needs_adaptation', False):
                server_sensory_dim = self.brain.brain.expected_sensory_dim
                client_sensory_dim = len(sensory_vector)
                
                if client_sensory_dim != server_sensory_dim:
                    # Adapt sensory vector to server dimensions
                    if client_sensory_dim < server_sensory_dim:
                        # Pad with zeros
                        adapted_sensory_vector = sensory_vector + [0.0] * (server_sensory_dim - client_sensory_dim)
                        if client_info['requests_served'] < 3:
                            print(f"   📏 Padded {client_sensory_dim}D → {server_sensory_dim}D sensory input")
                    else:
                        # Truncate
                        adapted_sensory_vector = sensory_vector[:server_sensory_dim]
                        if client_info['requests_served'] < 3:
                            print(f"   ✂️  Truncated {client_sensory_dim}D → {server_sensory_dim}D sensory input")
            
            # Store previous experience if this is not the first request
            if client_id in self.client_experiences:
                prev_experience = self.client_experiences[client_id]
                
                # Store complete experience: prev_sensory → prev_action → current_sensory (outcome)
                experience_id = self.brain.store_experience(
                    sensory_input=prev_experience['sensory_input'],
                    action_taken=prev_experience['action_taken'],
                    outcome=adapted_sensory_vector,  # Current sensory input is the outcome
                    predicted_action=prev_experience['predicted_action']
                )
                
                # Log vector stream learning (first few times)
                if client_info['requests_served'] < 5:
                    brain_stats = self.brain.get_brain_stats()
                    total_cycles = brain_stats['brain_summary']['total_cycles']
                    print(f"💾 {client_id}: Stored in vector streams {experience_id} "
                          f"(total cycles: {total_cycles})")
            
            # Process current sensory input through brain
            action_vector, brain_state = self.brain.process_sensory_input(adapted_sensory_vector)
            
            # CRITICAL: Run maintenance operations for field brain health
            if hasattr(self.brain, 'run_recommended_maintenance'):
                maintenance_performed = self.brain.run_recommended_maintenance()
                if any(maintenance_performed.values()):
                    # Note: Maintenance messages are already printed by the scheduler
                    pass
            
            # Adapt action dimensions if needed
            final_action_vector = action_vector
            if client_info.get('needs_adaptation', False):
                client_action_dim = client_info['capabilities']['action_dim']
                server_action_dim = len(action_vector)
                
                if client_action_dim != server_action_dim:
                    if client_action_dim < server_action_dim:
                        # Truncate to client's expected dimensions
                        final_action_vector = action_vector[:client_action_dim]
                        if client_info['requests_served'] < 3:
                            print(f"   ✂️  Truncated {server_action_dim}D → {client_action_dim}D action output")
                    else:
                        # Pad with zeros
                        final_action_vector = action_vector + [0.0] * (client_action_dim - server_action_dim)
                        if client_info['requests_served'] < 3:
                            print(f"   📏 Padded {server_action_dim}D → {client_action_dim}D action output")
            
            # Store current experience data for next cycle
            self.client_experiences[client_id] = {
                'sensory_input': adapted_sensory_vector.copy(),
                'action_taken': action_vector.copy(),
                'predicted_action': action_vector.copy(),  # For now, predicted = taken
                'timestamp': time.time()
            }
            
            # Debug output for first few requests
            if client_info['requests_served'] < 5:
                print(f"🧠 {client_id}: {len(sensory_vector)}D sensors → "
                      f"{len(final_action_vector)}D action ({brain_state['prediction_method']}, "
                      f"conf: {brain_state['prediction_confidence']:.3f})")
            
            return self.protocol.encode_action_output(final_action_vector)
            
        except Exception as e:
            # Determine specific error type based on exception
            error_code = self._classify_brain_error(e)
            
            error = create_brain_error(
                error_code,
                f"Brain processing failed: {e}",
                context={
                    'client_id': client_info['id'],
                    'sensory_input_size': len(sensory_vector),
                    'requests_served': client_info['requests_served'],
                    'exception_type': type(e).__name__
                },
                exception=e
            )
            log_brain_error(error, client_info['id'])
            return self.protocol.encode_error(error.error_code.value)
    
    def _classify_brain_error(self, exception: Exception) -> BrainErrorCode:
        """Classify exception into specific brain error code."""
        exception_str = str(exception).lower()
        exception_type = type(exception).__name__
        
        # Check for specific error patterns
        if 'similarity' in exception_str:
            return BrainErrorCode.SIMILARITY_ENGINE_FAILURE
        elif 'prediction' in exception_str:
            return BrainErrorCode.PREDICTION_ENGINE_FAILURE
        elif 'experience' in exception_str:
            return BrainErrorCode.EXPERIENCE_STORAGE_FAILURE
        elif 'activation' in exception_str:
            return BrainErrorCode.ACTIVATION_DYNAMICS_FAILURE
        elif 'pattern' in exception_str:
            return BrainErrorCode.PATTERN_ANALYSIS_FAILURE
        elif 'memory' in exception_str or 'memoryerror' in exception_type.lower():
            return BrainErrorCode.MEMORY_PRESSURE_ERROR
        elif 'cuda' in exception_str or 'gpu' in exception_str or 'mps' in exception_str:
            return BrainErrorCode.GPU_PROCESSING_ERROR
        elif 'timeout' in exception_str:
            return BrainErrorCode.SYSTEM_OVERLOAD
        elif 'dimension' in exception_str:
            return BrainErrorCode.INVALID_ACTION_DIMENSIONS
        else:
            # Default to generic brain processing error
            return BrainErrorCode.BRAIN_PROCESSING_ERROR
    
    def get_server_stats(self) -> dict:
        """Get comprehensive server statistics."""
        uptime = time.time() - self.start_time if self.start_time else 0
        
        # Active client info
        active_clients = []
        for client_info in self.clients.values():
            active_clients.append({
                'id': client_info['id'],
                'address': client_info['address'],
                'requests_served': client_info['requests_served'],
                'connected_duration': time.time() - client_info['connected_at'],
                'last_activity': time.time() - client_info['last_activity']
            })
        
        # Brain statistics
        brain_stats = self.brain.get_brain_stats()
        
        return {
            'server': {
                'host': self.host,
                'port': self.port,
                'running': self.running,
                'uptime_seconds': uptime,
                'total_requests': self.total_requests,
                'total_clients_served': self.total_clients_served,
                'requests_per_second': self.total_requests / max(1, uptime),
                'active_clients': len(self.clients)
            },
            'clients': active_clients,
            'brain': brain_stats
        }
    
    def _cleanup(self):
        """Cleanup server resources."""
        if self.server_socket:
            try:
                self.server_socket.close()
            except:
                pass
        
        # Close all client connections
        for client_info in list(self.clients.values()):
            # Connections will be closed by their handler threads
            pass
        
        self.clients.clear()
        print("🧹 TCP server cleanup complete")
    
    def __str__(self) -> str:
        if self.running:
            return f"MinimalTCPServer(running on {self.host}:{self.port}, {len(self.clients)} clients)"
        else:
            return f"MinimalTCPServer(stopped, served {self.total_clients_served} total clients)"
    
    def __repr__(self) -> str:
        return self.__str__()