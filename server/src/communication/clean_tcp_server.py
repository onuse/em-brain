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
    # Try backward-compatible protocol first
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
                        response_data = self.connection_handler.handle_sensory_input(client_id, vector_data)
                        response = protocol.encode_action_output(response_data)
                        
                    else:
                        # Unknown message type
                        response = protocol.encode_error(1.0)  # Unknown message type error
                    
                    # Send response
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