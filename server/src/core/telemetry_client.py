"""
Telemetry Client for Brain Monitoring

Provides easy access to brain telemetry from tests and external tools.
"""

import socket
import json
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass


@dataclass
class TelemetrySnapshot:
    """Simplified telemetry data for easy access"""
    session_id: str
    cycles: int
    energy: float
    confidence: float
    mode: str
    phase: str
    memory_regions: int
    constraints: int
    prediction_error: Optional[float] = None
    prediction_history: List[float] = None
    improvement_rate: float = 0.0
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()
        if self.prediction_history is None:
            self.prediction_history = []


class TelemetryClient:
    """Client for accessing brain telemetry through monitoring server"""
    
    def __init__(self, host: str = 'localhost', port: int = 9998, timeout: float = 5.0):
        """
        Initialize telemetry client.
        
        Args:
            host: Monitoring server host
            port: Monitoring server port
            timeout: Socket timeout in seconds
        """
        self.host = host
        self.port = port
        self.timeout = timeout
        self.socket = None
        self.connected = False
    
    def connect(self) -> bool:
        """Connect to monitoring server"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(self.timeout)
            self.socket.connect((self.host, self.port))
            
            # Receive and discard welcome message
            self.socket.recv(1024)
            
            self.connected = True
            return True
            
        except Exception as e:
            print(f"❌ Telemetry connection failed: {e}")
            self.connected = False
            return False
    
    def disconnect(self):
        """Disconnect from monitoring server"""
        if self.socket:
            try:
                self.socket.close()
            except:
                pass
        self.connected = False
    
    def _send_request(self, command: str) -> Optional[Dict[str, Any]]:
        """Send request and receive JSON response"""
        if not self.connected:
            if not self.connect():
                return None
        
        try:
            self.socket.send((command + "\n").encode('utf-8'))
            response = self.socket.recv(8192).decode('utf-8').strip()
            return json.loads(response)
        except Exception as e:
            print(f"❌ Telemetry request failed: {e}")
            self.connected = False
            return None
    
    def get_all_sessions(self) -> List[str]:
        """Get list of active session IDs"""
        response = self._send_request("telemetry")
        if response and response.get('status') == 'success':
            return list(response.get('data', {}).keys())
        return []
    
    def get_telemetry_summary(self) -> Dict[str, Dict[str, Any]]:
        """Get telemetry summary for all sessions"""
        response = self._send_request("telemetry")
        if response and response.get('status') == 'success':
            return response.get('data', {})
        return {}
    
    def get_session_telemetry(self, session_id: str) -> Optional[TelemetrySnapshot]:
        """Get detailed telemetry for a specific session"""
        # First get summary
        summary_response = self._send_request("telemetry")
        if not summary_response or summary_response.get('status') != 'success':
            return None
        
        summary_data = summary_response.get('data', {}).get(session_id, {})
        
        # Then get detailed
        detailed_response = self._send_request(f"telemetry {session_id}")
        if not detailed_response or detailed_response.get('status') != 'success':
            return None
        
        detailed_data = detailed_response.get('data', {})
        
        # Combine into snapshot
        return TelemetrySnapshot(
            session_id=session_id,
            cycles=summary_data.get('cycles', 0),
            energy=summary_data.get('energy', 0.0),
            confidence=detailed_data.get('prediction_confidence', summary_data.get('confidence', 0.0)),
            mode=summary_data.get('mode', 'unknown'),
            phase=summary_data.get('phase', 'unknown'),
            memory_regions=summary_data.get('memory_regions', 0),
            constraints=summary_data.get('constraints', 0),
            prediction_error=detailed_data.get('prediction_error'),
            prediction_history=detailed_data.get('prediction_history', []),
            improvement_rate=detailed_data.get('improvement_rate', 0.0)
        )
    
    def wait_for_session(self, max_wait: float = 5.0) -> Optional[str]:
        """Wait for a session to become available"""
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            sessions = self.get_all_sessions()
            if sessions:
                return sessions[0]  # Return first available session
            time.sleep(0.1)
        
        return None
    
    def track_prediction_learning(self, session_id: str, duration: float = 10.0) -> Dict[str, Any]:
        """Track prediction learning progress over time"""
        start_time = time.time()
        snapshots = []
        
        while time.time() - start_time < duration:
            telemetry = self.get_session_telemetry(session_id)
            if telemetry:
                snapshots.append({
                    'time': time.time() - start_time,
                    'confidence': telemetry.confidence,
                    'error': telemetry.prediction_error,
                    'cycles': telemetry.cycles
                })
            time.sleep(0.1)
        
        if not snapshots:
            return {'success': False, 'error': 'No telemetry data collected'}
        
        # Calculate improvement
        early_confidence = sum(s['confidence'] for s in snapshots[:5]) / min(5, len(snapshots))
        late_confidence = sum(s['confidence'] for s in snapshots[-5:]) / min(5, len(snapshots))
        improvement = late_confidence - early_confidence
        
        return {
            'success': True,
            'snapshots': snapshots,
            'early_confidence': early_confidence,
            'late_confidence': late_confidence,
            'improvement': improvement,
            'final_confidence': snapshots[-1]['confidence'] if snapshots else 0
        }
    
    def __enter__(self):
        """Context manager support"""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager cleanup"""
        self.disconnect()