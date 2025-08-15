"""
Clean TCP Server - Transport Layer Only

This TCP server ONLY handles network communication. All business logic
is delegated to the ConnectionHandler.
"""

import socket
import threading
import time
import datetime
from typing import Optional, Callable
import uuid

try:
    # Try chunked protocol for large messages
    from .protocol_chunked import ChunkedProtocol as MessageProtocol
    print("   Using chunked protocol (handles large messages with limited buffers)")
except ImportError:
    try:
        # Try backward-compatible protocol
        from .protocol_compat import BackwardCompatibleProtocol as MessageProtocol
        print("   Using backward-compatible protocol (handles old and new clients)")
    except ImportError:
        # Fall back to standard protocol
        from .protocol import MessageProtocol
        print("   Using standard protocol (requires magic bytes)")

from .protocol import MessageProtocolError
from ..core.interfaces import IConnectionHandler


class CleanTCPServer:
    """
    TCP server that ONLY handles network transport.
    
    All business logic is delegated to the ConnectionHandler,
    maintaining clean separation of concerns.
    """
    
    def __init__(self, connection_handler: IConnectionHandler, 
                 host: str = '0.0.0.0', port: int = 9999):
        """
        Initialize the TCP server.
        
        Args:
            connection_handler: Handler for all business logic
            host: Server host address
            port: Server port
        """
        self.connection_handler = connection_handler
        self.host = host
        self.port = port
        
        # Networking
        self.server_socket = None
        self.running = False
        
        # Protocol class (each client gets its own instance)
        self.protocol_class = MessageProtocol
        
        # Client tracking (for network management only)
        self.active_clients = {}  # client_id -> socket
        self.client_counter = 0
        
        # Statistics
        self.start_time = None
        self.total_connections = 0
        
        # Silent initialization - details shown at server ready
    
    def start(self):
        """Start the TCP server."""
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"\n[{timestamp}] 🚀 Starting TCP server on {self.host}:{self.port}")
        
        try:
            # Create server socket
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(5)
            
            self.running = True
            self.start_time = time.time()
            
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print(f"[{timestamp}] ✅ Server listening on {self.host}:{self.port}")
            print("   Waiting for robot clients to connect...")
            
            # Main server loop
            while self.running:
                try:
                    client_socket, client_address = self.server_socket.accept()
                    # Increase receive buffer for large vision data (1.2MB messages)
                    client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 2 * 1024 * 1024)  # 2MB
                    # Also disable Nagle's for real-time communication
                    client_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    print(f"[{timestamp}] 🤖 New connection from {client_address}")
                    
                    # Handle client in separate thread
                    client_thread = threading.Thread(
                        target=self._handle_client_connection,
                        args=(client_socket, client_address),
                        daemon=True
                    )
                    client_thread.start()
                    
                except socket.error as e:
                    if self.running:
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
    
    def _handle_client_connection(self, client_socket: socket.socket, client_address: tuple):
        """Handle individual client connection - transport only."""
        
        # Generate unique client ID
        client_id = f"client_{self.client_counter}_{uuid.uuid4().hex[:8]}"
        self.client_counter += 1
        self.total_connections += 1
        
        # Track socket
        self.active_clients[client_id] = client_socket
        
        # Create protocol instance for this client
        protocol = self.protocol_class()
        
        print(f"📋 Assigned ID {client_id} to connection from {client_address}")
        
        try:
            # Set socket timeout
            client_socket.settimeout(300.0)  # 5 minutes
            
            # Main communication loop
            while self.running:
                try:
                    # Receive message
                    msg_type, vector_data = protocol.receive_message(client_socket, timeout=60.0)
                    
                    # Route to handler based on message type
                    if msg_type == protocol.MSG_HANDSHAKE:
                        response_data = self.connection_handler.handle_handshake(client_id, vector_data)
                        response = protocol.encode_handshake(response_data)
                        
                    elif msg_type == protocol.MSG_SENSORY_INPUT:
                        # Only log every 100th message to reduce spam
                        if not hasattr(self, '_msg_count'):
                            self._msg_count = {}
                        if client_id not in self._msg_count:
                            self._msg_count[client_id] = 0
                        self._msg_count[client_id] += 1
                        
                        response_data = self.connection_handler.handle_sensory_input(client_id, vector_data)
                        response = protocol.encode_action_output(response_data)
                        
                        # Occasional status with brain insights
                        if self._msg_count[client_id] % 50 == 0:
                            # Get brain state for interesting telemetry
                            brain_state = self._get_brain_insights(client_id)
                            print(f"   🧠 Cycle {self._msg_count[client_id]}: {brain_state}")
                        elif self._msg_count[client_id] % 25 == 0:
                            # Show sensor summary
                            num_sensors = len(vector_data) if vector_data else 0
                            print(f"   📡 Processing {num_sensors:,} sensor values")
                        elif self._msg_count[client_id] % 10 == 0:
                            # Just a pulse to show it's alive
                            print(f"   ⚡ Active", end="\r")  # Carriage return to overwrite
                        
                    else:
                        # Unknown message type
                        response = protocol.encode_error(1.0)  # Unknown message type error
                    
                    # Send response (silently - we already show status above)
                    if not protocol.send_message(client_socket, response):
                        print(f"💔 Failed to send response to {client_id}")
                        break
                
                except TimeoutError:
                    print(f"⏰ Client {client_id} timed out")
                    break
                    
                except MessageProtocolError as e:
                    print(f"⚠️  Protocol error from {client_id}: {e}")
                    error_response = protocol.encode_error(2.0)  # Protocol error
                    protocol.send_message(client_socket, error_response)
                    
                except ConnectionError:
                    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    print(f"[{timestamp}] 💔 Client {client_id} disconnected")
                    break
                    
                except Exception as e:
                    print(f"❌ Unexpected error handling {client_id}: {e}")
                    break
        
        finally:
            # Cleanup connection
            try:
                client_socket.close()
            except:
                pass
            
            # Remove from tracking
            if client_id in self.active_clients:
                del self.active_clients[client_id]
            
            # Notify handler of disconnect
            self.connection_handler.handle_disconnect(client_id)
            
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print(f"[{timestamp}] 👋 Connection closed for {client_id}")
    
    def get_server_stats(self) -> dict:
        """Get server statistics."""
        uptime = time.time() - self.start_time if self.start_time else 0
        
        return {
            'host': self.host,
            'port': self.port,
            'running': self.running,
            'uptime_seconds': uptime,
            'total_connections': self.total_connections,
            'active_connections': len(self.active_clients)
        }
    
    def _cleanup(self):
        """Cleanup server resources."""
        if self.server_socket:
            try:
                self.server_socket.close()
            except:
                pass
        
        # Close all client connections
        for client_id, client_socket in list(self.active_clients.items()):
            try:
                client_socket.close()
            except:
                pass
        
        self.active_clients.clear()
        print("🧹 TCP server cleanup complete")
    
    def _get_brain_insights(self, client_id: str) -> str:
        """Get interesting insights about brain state."""
        try:
            # Access brain through connection handler
            brain = None
            if hasattr(self.connection_handler, 'brain_service'):
                if hasattr(self.connection_handler.brain_service, 'brain_pool'):
                    brain = self.connection_handler.brain_service.brain_pool.brain
            
            if brain and hasattr(brain, 'field'):
                import torch
                field = brain.field
                with torch.no_grad():
                    # Calculate various metrics
                    energy = torch.abs(field).mean().item()
                    max_activation = torch.abs(field).max().item()
                    active_percent = (torch.abs(field) > 0.1).sum().item() / field.numel() * 100
                    
                    # Get motor outputs if available
                    motor_str = ""
                    if hasattr(brain, '_last_motor_commands'):
                        motors = brain._last_motor_commands
                        if motors and len(motors) >= 2:
                            thrust = motors[0]
                            turn = motors[1]
                            motor_str = f" | thrust={thrust:+.2f} turn={turn:+.2f}"
                    
                    # Determine brain state
                    if energy < 0.01:
                        state = "🧊 dormant"
                    elif energy < 0.05:
                        state = "😴 waking"
                    elif energy < 0.1:
                        state = "🤔 thinking"
                    elif energy < 0.2:
                        state = "🎯 focused"
                    elif energy < 0.3:
                        state = "⚡ active"
                    else:
                        state = "🔥 intense"
                    
                    # Check for interesting patterns
                    if hasattr(brain, 'cycle_count'):
                        cycles = brain.cycle_count
                        if cycles > 0 and cycles % 1000 == 0:
                            state += f" | milestone: {cycles:,} cycles!"
                    
                    return f"{state} | energy={energy:.3f} | activity={active_percent:.1f}%{motor_str}"
        except Exception as e:
            import traceback
            print(f"Debug: Error in brain insights: {e}")
            traceback.print_exc()
        return "🔄 processing..."