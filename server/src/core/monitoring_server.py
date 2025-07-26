"""
Dynamic Brain Monitoring Server

Provides read-only monitoring access to the dynamic brain architecture.
Supports statistics, validation data, and third-party UI access.
"""

import socket
import threading
import time
import json
import datetime
from typing import Dict, Any, Optional


class DynamicMonitoringServer:
    """
    TCP server for monitoring the dynamic brain architecture.
    
    Provides JSON-based read-only access to:
    - Brain statistics
    - Session information
    - Performance metrics
    - Field evolution data
    """
    
    def __init__(self, brain_service, connection_handler, host: str = '0.0.0.0', port: int = 9998):
        """
        Initialize monitoring server.
        
        Args:
            brain_service: The BrainService instance to monitor
            connection_handler: The ConnectionHandler for connection stats
            host: Server host address
            port: Monitoring port (different from robot port)
        """
        self.brain_service = brain_service
        self.connection_handler = connection_handler
        self.host = host
        self.port = port
        
        # Networking
        self.server_socket = None
        self.running = False
        self.server_thread = None
        
        # Client management
        self.clients = {}
        self.client_counter = 0
        
        print(f"📊 Dynamic Monitoring Server initialized")
        print(f"   Host: {host}:{port}")
        print(f"   Purpose: Statistics, validation, third-party UI access")
    
    def start(self):
        """Start the monitoring server in a background thread."""
        if self.running:
            print("⚠️  Monitoring server already running")
            return
        
        self.running = True
        self.server_thread = threading.Thread(target=self._run_server, daemon=True)
        self.server_thread.start()
        
        print(f"✅ Monitoring server started on {self.host}:{self.port}")
    
    def stop(self):
        """Stop the monitoring server."""
        self.running = False
        
        if self.server_socket:
            try:
                self.server_socket.close()
            except:
                pass
        
        if self.server_thread:
            self.server_thread.join(timeout=2.0)
        
        print("📊 Monitoring server stopped")
    
    def _run_server(self):
        """Main server loop."""
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(5)
            self.server_socket.settimeout(1.0)  # Allow periodic checks
            
            while self.running:
                try:
                    client_socket, client_address = self.server_socket.accept()
                    
                    # Handle client in a new thread
                    client_thread = threading.Thread(
                        target=self._handle_monitoring_client,
                        args=(client_socket, client_address),
                        daemon=True
                    )
                    client_thread.start()
                    
                except socket.timeout:
                    continue
                except Exception as e:
                    if self.running:
                        print(f"❌ Monitoring server error: {e}")
                        break
        
        finally:
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
        
        print(f"📋 Monitoring client {client_id} connected from {client_address}")
        
        try:
            client_socket.settimeout(1800.0)  # 30 minute timeout
            
            # Send welcome message
            welcome = {
                'status': 'connected',
                'server': 'Dynamic Brain Monitoring Server',
                'version': '2.0',
                'commands': [
                    'brain_stats',
                    'session_info',
                    'connection_stats',
                    'active_brains',
                    'performance_metrics'
                ]
            }
            client_socket.send((json.dumps(welcome) + "\n").encode('utf-8'))
            
            # Main client communication loop
            while self.running:
                try:
                    # Receive request
                    data = client_socket.recv(1024).decode('utf-8').strip()
                    if not data:
                        break
                    
                    client_info['last_activity'] = time.time()
                    
                    # Process request
                    response = self._handle_monitoring_request(client_info, data)
                    
                    # Send JSON response
                    response_json = json.dumps(response, indent=2) + "\n"
                    client_socket.send(response_json.encode('utf-8'))
                    
                    client_info['requests_served'] += 1
                
                except socket.timeout:
                    print(f"⏰ Monitoring client {client_id} timed out")
                    break
                except ConnectionError:
                    break
                except Exception as e:
                    print(f"❌ Error handling monitoring request: {e}")
                    break
        
        finally:
            try:
                client_socket.close()
            except:
                pass
            
            if client_id in self.clients:
                del self.clients[client_id]
            
            print(f"👋 Monitoring client {client_id} disconnected")
    
    def _handle_monitoring_request(self, client_info: dict, request: str) -> Dict[str, Any]:
        """Handle monitoring data request."""
        
        try:
            request_lower = request.lower().strip()
            
            if request_lower == "brain_stats":
                # Get all session statistics
                sessions = self.brain_service.get_all_sessions()
                return {
                    'status': 'success',
                    'request': request,
                    'timestamp': time.time(),
                    'data': {
                        'sessions': sessions,
                        'total_sessions': len(sessions)
                    }
                }
            
            elif request_lower == "session_info":
                # Get detailed session information
                sessions = self.brain_service.get_all_sessions()
                session_info = []
                
                for session_id, stats in sessions.items():
                    info = {
                        'session_id': session_id,
                        'robot_type': stats.get('robot_type', 'unknown'),
                        'brain_dimensions': stats.get('brain_dimensions', 0),
                        'cycles': stats.get('cycles_processed', 0),
                        'experiences': stats.get('total_experiences', 0),
                        'avg_cycle_ms': stats.get('average_cycle_time_ms', 0),
                        'uptime': stats.get('uptime_seconds', 0)
                    }
                    session_info.append(info)
                
                return {
                    'status': 'success',
                    'request': request,
                    'timestamp': time.time(),
                    'data': session_info
                }
            
            elif request_lower == "connection_stats":
                # Get connection statistics
                stats = self.connection_handler.get_stats()
                return {
                    'status': 'success',
                    'request': request,
                    'timestamp': time.time(),
                    'data': stats
                }
            
            elif request_lower == "active_brains":
                # Get active brain information
                active_brains = self.brain_service.brain_pool.get_active_brains()
                brain_info = []
                
                for profile_key, brain in active_brains.items():
                    config = self.brain_service.brain_pool.get_brain_config(profile_key)
                    info = {
                        'profile': profile_key,
                        'field_dimensions': brain.get_field_dimensions(),
                        'sensory_dim': config.get('sensory_dim', 0),
                        'motor_dim': config.get('motor_dim', 0),
                        'spatial_resolution': config.get('spatial_resolution', 0)
                    }
                    brain_info.append(info)
                
                return {
                    'status': 'success',
                    'request': request,
                    'timestamp': time.time(),
                    'data': brain_info
                }
            
            elif request_lower == "performance_metrics":
                # Get performance metrics across all sessions
                sessions = self.brain_service.get_all_sessions()
                
                total_cycles = sum(s.get('cycles_processed', 0) for s in sessions.values())
                total_experiences = sum(s.get('total_experiences', 0) for s in sessions.values())
                avg_cycle_times = [s.get('average_cycle_time_ms', 0) for s in sessions.values() if s.get('average_cycle_time_ms', 0) > 0]
                
                metrics = {
                    'total_cycles': total_cycles,
                    'total_experiences': total_experiences,
                    'active_sessions': len(sessions),
                    'average_cycle_time_ms': sum(avg_cycle_times) / len(avg_cycle_times) if avg_cycle_times else 0,
                    'monitoring_clients': len(self.clients)
                }
                
                return {
                    'status': 'success',
                    'request': request,
                    'timestamp': time.time(),
                    'data': metrics
                }
            
            else:
                # Unknown request
                return {
                    'status': 'error',
                    'request': request,
                    'error': f"Unknown command: {request}",
                    'timestamp': time.time(),
                    'available_commands': [
                        'brain_stats',
                        'session_info',
                        'connection_stats',
                        'active_brains',
                        'performance_metrics'
                    ]
                }
        
        except Exception as e:
            return {
                'status': 'error',
                'request': request,
                'error': str(e),
                'timestamp': time.time()
            }