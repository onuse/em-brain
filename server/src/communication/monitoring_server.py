"""
Brain Monitoring Server

Separate socket server for brain statistics, validation data, and third-party UI access.
This runs parallel to the robot communication socket and provides read-only monitoring.
"""

import socket
import threading
import time
import json
import datetime
from typing import Dict, Any, Optional, Callable

class BrainMonitoringServer:
    """
    TCP server for monitoring brain statistics and validation data.
    
    Provides read-only access to brain internals for validation experiments,
    debugging tools, and third-party UIs without interfering with robot communication.
    """
    
    def __init__(self, brain_factory, host: str = '0.0.0.0', port: int = 9998):
        """
        Initialize monitoring server.
        
        Args:
            brain_factory: Brain factory to monitor
            host: Server host address
            port: Monitoring port (different from robot port)
        """
        self.brain_factory = brain_factory
        self.host = host
        self.port = port
        
        # Networking
        self.server_socket = None
        self.running = False
        
        # Client management
        self.clients = {}  # client_id -> client_info
        self.client_counter = 0
        
        print(f"📊 BrainMonitoringServer initialized")
        print(f"   Host: {host}:{port}")
        print(f"   Purpose: Statistics, validation, third-party UI access")
    
    def start(self):
        """Start the monitoring server."""
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"\n[{timestamp}] 📊 Starting monitoring server on {self.host}:{self.port}")
        
        try:
            # Create server socket
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(3)  # Allow monitoring clients
            
            self.running = True
            
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print(f"[{timestamp}] ✅ Monitoring server listening on {self.host}:{self.port}")
            print("   Accepting connections from validation experiments and UIs...")
            
            # Main server loop
            while self.running:
                try:
                    client_socket, client_address = self.server_socket.accept()
                    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    print(f"[{timestamp}] 📊 Monitoring client connected from {client_address}")
                    
                    # Handle client in separate thread
                    client_thread = threading.Thread(
                        target=self._handle_monitoring_client,
                        args=(client_socket, client_address),
                        daemon=True
                    )
                    client_thread.start()
                    
                except socket.error as e:
                    if self.running:  # Only log if not shutting down
                        print(f"⚠️  Monitoring server socket error: {e}")
                        break
        
        except Exception as e:
            print(f"❌ Failed to start monitoring server: {e}")
            self.running = False
        
        finally:
            self._cleanup()
    
    def stop(self):
        """Stop the monitoring server."""
        print("🛑 Stopping monitoring server...")
        self.running = False
        
        if self.server_socket:
            try:
                self.server_socket.close()
            except:
                pass
    
    def _handle_monitoring_client(self, client_socket: socket.socket, client_address: tuple):
        """Handle individual monitoring client connection."""
        
        # Generate unique client ID
        client_id = f"monitor_{self.client_counter}"
        self.client_counter += 1
        
        # Track client info
        client_info = {
            'id': client_id,
            'address': client_address,
            'connected_at': time.time(),
            'requests_served': 0,
            'last_activity': time.time()
        }
        self.clients[client_id] = client_info
        
        print(f"📋 Monitoring client {client_id} registered from {client_address}")
        
        try:
            client_socket.settimeout(60.0)  # 1 minute timeout
            
            # Main client communication loop
            while self.running:
                try:
                    # Receive request from monitoring client
                    data = client_socket.recv(1024).decode('utf-8').strip()
                    if not data:
                        break
                    
                    client_info['last_activity'] = time.time()
                    
                    # Process monitoring request
                    response = self._handle_monitoring_request(client_info, data)
                    
                    # Send JSON response
                    response_json = json.dumps(response, indent=2) + "\n"
                    client_socket.send(response_json.encode('utf-8'))
                    
                    client_info['requests_served'] += 1
                
                except socket.timeout:
                    print(f"⏰ Monitoring client {client_id} timed out")
                    break
                except ConnectionError:
                    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    print(f"[{timestamp}] 💔 Monitoring client {client_id} disconnected")
                    break
                except Exception as e:
                    print(f"❌ Error handling monitoring client {client_id}: {e}")
                    break
        
        finally:
            # Cleanup client connection
            try:
                client_socket.close()
            except:
                pass
            
            if client_id in self.clients:
                del self.clients[client_id]
            
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print(f"[{timestamp}] 👋 Monitoring client {client_id} disconnected")
    
    def _handle_monitoring_request(self, client_info: dict, request: str) -> Dict[str, Any]:
        """Handle monitoring data request."""
        
        try:
            # Parse request
            request_lower = request.lower().strip()
            
            if request_lower == "brain_stats":
                # Get comprehensive brain statistics
                brain_stats = self.brain_factory.get_brain_stats()
                return {
                    'status': 'success',
                    'request': request,
                    'timestamp': time.time(),
                    'data': brain_stats
                }
            
            elif request_lower == "brain_state":
                # Get current brain state (for validation)
                try:
                    # Get latest brain state from factory
                    brain_state = {}
                    if hasattr(self.brain_factory, '_current_brain_state'):
                        brain_state = self.brain_factory._current_brain_state.copy()
                    
                    return {
                        'status': 'success',
                        'request': request,
                        'timestamp': time.time(),
                        'data': brain_state
                    }
                except Exception as e:
                    return {
                        'status': 'error',
                        'request': request,
                        'error': f"Brain state access failed: {e}",
                        'timestamp': time.time()
                    }
            
            elif request_lower == "prediction_metrics":
                # Get prediction-driven learning metrics
                brain_stats = self.brain_factory.get_brain_stats()
                brain_summary = brain_stats.get('brain_summary', {})
                
                return {
                    'status': 'success',
                    'request': request,
                    'timestamp': time.time(),
                    'data': {
                        'prediction_efficiency': brain_summary.get('prediction_efficiency', 0.0),
                        'learning_detected': brain_summary.get('learning_detected', False),
                        'prediction_accuracy': brain_summary.get('prediction_accuracy', 0.0),
                        'field_evolution_cycles': brain_summary.get('field_evolution_cycles', 0),
                        'prediction_confidence': brain_summary.get('prediction_confidence', 0.0),
                        'exploration_score': brain_summary.get('exploration_score', 0.0),
                        'biological_realism': brain_summary.get('biological_realism', 0.0),
                        'strategy_emergence': brain_summary.get('strategy_emergence', 0.0)
                    }
                }
            
            elif request_lower == "server_status":
                # Get monitoring server status
                return {
                    'status': 'success',
                    'request': request,
                    'timestamp': time.time(),
                    'data': {
                        'monitoring_server': {
                            'host': self.host,
                            'port': self.port,
                            'running': self.running,
                            'active_clients': len(self.clients)
                        },
                        'brain_factory': str(self.brain_factory)
                    }
                }
            
            elif request_lower == "ping":
                # Simple ping response
                return {
                    'status': 'success',
                    'request': request,
                    'timestamp': time.time(),
                    'data': {'pong': True}
                }
            
            else:
                return {
                    'status': 'error',
                    'request': request,
                    'error': f"Unknown request: {request}",
                    'timestamp': time.time(),
                    'available_requests': [
                        'brain_stats', 'brain_state', 'prediction_metrics', 
                        'server_status', 'ping'
                    ]
                }
        
        except Exception as e:
            return {
                'status': 'error',
                'request': request,
                'error': f"Request processing failed: {e}",
                'timestamp': time.time()
            }
    
    def _cleanup(self):
        """Cleanup server resources."""
        if self.server_socket:
            try:
                self.server_socket.close()
            except:
                pass
        
        self.clients.clear()
        print("🧹 Monitoring server cleanup complete")
    
    def __str__(self) -> str:
        if self.running:
            return f"BrainMonitoringServer(running on {self.host}:{self.port}, {len(self.clients)} clients)"
        else:
            return f"BrainMonitoringServer(stopped)"
    
    def __repr__(self) -> str:
        return self.__str__()