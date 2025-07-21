"""
Brain Monitoring Client

Client for accessing brain monitoring data from validation experiments and third-party UIs.
Connects to the separate monitoring socket (not the robot communication socket).
"""

import socket
import json
import time
from typing import Dict, Any, Optional

class BrainMonitoringClient:
    """
    Client for accessing brain monitoring data.
    
    Connects to the monitoring server (separate from robot communication)
    to fetch brain statistics, validation metrics, and internal state.
    """
    
    def __init__(self, server_host: str = 'localhost', server_port: int = 9998):
        """
        Initialize monitoring client.
        
        Args:
            server_host: Monitoring server hostname/IP
            server_port: Monitoring server port (9998, not robot port 9999)
        """
        self.server_host = server_host
        self.server_port = server_port
        
        # Connection state
        self.socket = None
        self.connected = False
        
        print(f"📊 BrainMonitoringClient initialized for {server_host}:{server_port}")
    
    def connect(self) -> bool:
        """
        Connect to the monitoring server.
        
        Returns:
            True if connected successfully
        """
        try:
            # Create socket connection
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(5.0)  # 5 second connection timeout
            
            print(f"📊 Connecting to monitoring server at {self.server_host}:{self.server_port}...")
            self.socket.connect((self.server_host, self.server_port))
            
            self.connected = True
            print(f"✅ Connected to monitoring server!")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to connect to monitoring server: {e}")
            self._cleanup_connection()
            return False
    
    def get_brain_stats(self) -> Optional[Dict[str, Any]]:
        """Get comprehensive brain statistics."""
        return self._send_request("brain_stats")
    
    def get_brain_state(self) -> Optional[Dict[str, Any]]:
        """Get current brain state (for validation)."""
        return self._send_request("brain_state")
    
    def get_prediction_metrics(self) -> Optional[Dict[str, Any]]:
        """Get prediction-driven learning metrics."""
        return self._send_request("prediction_metrics")
    
    def ping(self) -> bool:
        """
        Send keepalive ping to server.
        
        Returns:
            True if server responded, False if disconnected
        """
        response = self._send_request("ping")
        if response is None:
            return False
        return response.get('status') == 'pong'
    
    def get_server_status(self) -> Optional[Dict[str, Any]]:
        """Get monitoring server status."""
        return self._send_request("server_status")
    
    
    def _send_request(self, request: str) -> Optional[Dict[str, Any]]:
        """
        Make a monitoring request to the server.
        
        Args:
            request: Request string
            
        Returns:
            Response data or None if failed
        """
        if not self.connected:
            print("⚠️  Not connected to monitoring server")
            return None
        
        try:
            # Send request
            self.socket.send((request + "\n").encode('utf-8'))
            
            # Receive JSON response with timeout
            response_data = b""
            timeout_start = time.time()
            timeout_seconds = 10.0  # 10 second timeout for monitoring requests
            
            while True:
                # Check for timeout
                if time.time() - timeout_start > timeout_seconds:
                    print(f"⚠️  Monitoring request timeout ({timeout_seconds}s)")
                    self._cleanup_connection()
                    return None
                
                try:
                    # Set short timeout for recv to allow periodic timeout checks
                    self.socket.settimeout(1.0)
                    chunk = self.socket.recv(4096)
                    
                    if not chunk:
                        print("⚠️  Monitoring server disconnected (empty response)")
                        self._cleanup_connection()
                        return None
                        
                    response_data += chunk
                    
                    # Check if we have a complete JSON response
                    try:
                        response_str = response_data.decode('utf-8')
                        # Try to parse JSON - if it fails, we need more data
                        response = json.loads(response_str)
                        
                        if response.get('status') == 'success':
                            return response.get('data')
                        else:
                            print(f"⚠️  Monitoring server error: {response.get('error', 'Unknown error')}")
                            return None
                            
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        # Need more data - continue receiving
                        continue
                        
                except socket.timeout:
                    # Normal timeout - continue and check overall timeout
                    continue
                
        except Exception as e:
            print(f"❌ Monitoring request failed: {e}")
            self._cleanup_connection()
            return None
    
    def disconnect(self):
        """Disconnect from monitoring server."""
        print("👋 Disconnecting from monitoring server...")
        self._cleanup_connection()
    
    def _cleanup_connection(self):
        """Clean up connection resources."""
        self.connected = False
        
        if self.socket:
            try:
                self.socket.close()
            except:
                pass
            self.socket = None
    
    def is_connected(self) -> bool:
        """Check if connected to monitoring server."""
        return self.connected
    
    def __str__(self) -> str:
        if self.connected:
            return f"BrainMonitoringClient(connected to {self.server_host}:{self.server_port})"
        else:
            return f"BrainMonitoringClient(disconnected)"
    
    def __repr__(self) -> str:
        return self.__str__()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()


def create_monitoring_client(server_host: str = 'localhost', server_port: int = 9998) -> BrainMonitoringClient:
    """
    Create and connect to brain monitoring server.
    
    Args:
        server_host: Monitoring server hostname/IP
        server_port: Monitoring server port
        
    Returns:
        Connected monitoring client or None if failed
    """
    client = BrainMonitoringClient(server_host, server_port)
    if client.connect():
        return client
    else:
        return None