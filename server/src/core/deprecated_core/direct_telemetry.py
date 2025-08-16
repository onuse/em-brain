"""
Direct Telemetry Access

Provides telemetry access directly from brain sessions without requiring
a monitoring server or socket connection. This is ideal for tests and
scenarios where the monitoring infrastructure isn't needed.
"""

from typing import Dict, Any, Optional, List
import time
from ..core.telemetry_client import TelemetrySnapshot


class DirectTelemetry:
    """Direct access to brain telemetry without sockets"""
    
    def __init__(self, brain_service=None, connection_handler=None):
        """
        Initialize direct telemetry.
        
        Args:
            brain_service: BrainService instance for session access
            connection_handler: ConnectionHandler for robot mapping
        """
        self.brain_service = brain_service
        self.connection_handler = connection_handler
        self.connected = True  # Always "connected" for compatibility
    
    def connect(self) -> bool:
        """Compatibility method - always succeeds"""
        return True
    
    def disconnect(self):
        """Compatibility method - no-op"""
        pass
    
    def get_all_sessions(self) -> List[str]:
        """Get list of active session IDs"""
        if self.brain_service:
            return list(self.brain_service.sessions.keys())
        return []
    
    def wait_for_session(self, max_wait: float = 5.0, client_id: Optional[str] = None) -> Optional[str]:
        """
        Wait for a session to become available.
        
        Args:
            max_wait: Maximum time to wait
            client_id: Specific client to look for
            
        Returns:
            Session ID or None
        """
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            if client_id and self.connection_handler:
                # Look for specific client's session  
                if hasattr(self.connection_handler, 'client_sessions'):
                    session = self.connection_handler.client_sessions.get(client_id)
                    if session:
                        return session.session_id
            else:
                # Return first available session
                sessions = self.get_all_sessions()
                if sessions:
                    return sessions[0]
            
            time.sleep(0.1)
        
        return None
    
    def get_session_telemetry(self, session_id: str) -> Optional[TelemetrySnapshot]:
        """Get telemetry for a specific session"""
        if not self.brain_service:
            return None
        
        # Get session
        session = self.brain_service.sessions.get(session_id)
        if not session or not hasattr(session, 'telemetry_adapter'):
            return None
        
        # Get telemetry from adapter
        telemetry = session.telemetry_adapter.get_telemetry()
        
        # Convert to snapshot
        return TelemetrySnapshot(
            session_id=session_id,
            cycles=telemetry.brain_cycles,
            energy=telemetry.field_energy,
            confidence=telemetry.prediction_confidence,
            mode=telemetry.cognitive_mode,
            phase=telemetry.phase_state,
            memory_regions=telemetry.memory_regions,
            constraints=telemetry.active_constraints,
            prediction_error=telemetry.prediction_error,
            prediction_history=telemetry.prediction_history,
            improvement_rate=telemetry.improvement_rate,
            timestamp=telemetry.timestamp
        )
    
    def get_telemetry_summary(self) -> Dict[str, Dict[str, Any]]:
        """Get telemetry summary for all sessions"""
        if not self.brain_service:
            return {}
        
        return self.brain_service.get_all_telemetry()
    
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
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager cleanup"""
        pass


def create_telemetry_client(brain_service=None, connection_handler=None, use_socket=False):
    """
    Factory function to create appropriate telemetry client.
    
    Args:
        brain_service: BrainService instance
        connection_handler: ConnectionHandler instance
        use_socket: If True, use socket-based TelemetryClient
        
    Returns:
        DirectTelemetry or TelemetryClient instance
    """
    if use_socket:
        from .telemetry_client import TelemetryClient
        client = TelemetryClient()
        client.connect()
        return client
    else:
        return DirectTelemetry(brain_service, connection_handler)