"""
Connection Handler implementation.

Orchestrates client connections, routing messages to appropriate services
while maintaining clean separation from network transport concerns.
"""

import time
from typing import Dict, List, Optional
from threading import RLock

from .interfaces import (
    IConnectionHandler, IRobotRegistry, IBrainService, 
    IBrainSession
)
from .error_codes import (
    BrainError, ErrorHandler, ErrorCode,
    handshake_error, session_not_found, brain_processing_error
)


class ConnectionHandler(IConnectionHandler):
    """
    Manages client connections and routes to appropriate services.
    
    This is the main orchestrator that coordinates between the robot
    registry, brain service, and active sessions. It maintains the
    mapping between network clients and their brain sessions.
    """
    
    def __init__(self, robot_registry: IRobotRegistry, brain_service: IBrainService):
        self.robot_registry = robot_registry
        self.brain_service = brain_service
        
        # Mappings
        self.client_sessions: Dict[str, IBrainSession] = {}  # client_id -> session
        self.client_robots: Dict[str, str] = {}  # client_id -> robot_id
        
        # Thread safety
        self.lock = RLock()
        
        # Statistics
        self.total_connections = 0
        self.total_messages = 0
        self.start_time = time.time()
        
        # Error handling
        self.error_handler = ErrorHandler()
    
    def handle_handshake(self, client_id: str, capabilities: List[float]) -> List[float]:
        """Handle handshake and return response capabilities."""
        
        with self.lock:
            # Check if client already has a session
            if client_id in self.client_sessions:
                error = BrainError(
                    ErrorCode.DUPLICATE_SESSION,
                    f"Client {client_id} already has an active session",
                    {'client_id': client_id}
                )
                self.error_handler.handle_error(error, client_id)
                # Return existing session's handshake response
                return self.client_sessions[client_id].get_handshake_response()
            
            try:
                # Register robot based on capabilities
                robot = self.robot_registry.register_robot(capabilities)
                
                # Create brain session for this robot
                session = self.brain_service.create_session(robot)
                
                # Store mappings
                self.client_sessions[client_id] = session
                self.client_robots[client_id] = robot.robot_id
                
                # Update statistics
                self.total_connections += 1
                
                # Get handshake response from session
                response = session.get_handshake_response()
                
                print(f"✅ Handshake complete for client {client_id}")
                print(f"   Session: {session.get_session_id()}")
                
                return response
                
            except BrainError as be:
                # Already a BrainError, just handle it
                self.error_handler.handle_error(be, client_id)
                print(f"❌ {be}")
                return be.to_response()
                
            except Exception as e:
                # Wrap in BrainError
                error = handshake_error(str(e), capabilities)
                self.error_handler.handle_error(error, client_id)
                print(f"❌ Handshake failed for client {client_id}: {e}")
                import traceback
                traceback.print_exc()
                return error.to_response()
    
    def handle_sensory_input(self, client_id: str, sensory_data: List[float]) -> List[float]:
        """Process sensory input and return motor commands."""
        
        with self.lock:
            # Get client's session
            session = self.client_sessions.get(client_id)
            
            if not session:
                error = session_not_found(client_id)
                self.error_handler.handle_error(error, client_id)
                print(f"⚠️  {error}")
                # Return empty array - client needs to handshake first
                return []
            
            try:
                # Process through brain session
                motor_commands = session.process_sensory_input(sensory_data)
                
                # Update statistics
                self.total_messages += 1
                
                return motor_commands
                
            except BrainError as be:
                # Already a BrainError, just handle it
                self.error_handler.handle_error(be, client_id)
                print(f"❌ {be}")
                return self._safe_motor_response(client_id)
                
            except Exception as e:
                # Wrap in BrainError
                error = brain_processing_error(
                    f"Error processing sensory input: {e}",
                    {'sensory_dimensions': len(sensory_data) if sensory_data else 0}
                )
                self.error_handler.handle_error(error, client_id)
                print(f"❌ Error processing sensory input for client {client_id}: {e}")
                return self._safe_motor_response(client_id)
    
    def handle_disconnect(self, client_id: str) -> None:
        """Clean up when client disconnects."""
        
        self.logger.info(f"DISCONNECT_START: {client_id} beginning cleanup")
        
        with self.lock:
            # Get session
            session = self.client_sessions.get(client_id)
            
            if session:
                session_id = session.get_session_id()
                robot_id = self.client_robots.get(client_id)
                
                self.logger.debug(f"DISCONNECT_SESSION: {client_id} closing session {session_id}")
                # Close brain session
                self.brain_service.close_session(session_id)
                
                # Remove mappings
                del self.client_sessions[client_id]
                
                if client_id in self.client_robots:
                    del self.client_robots[client_id]
                
                self.logger.info(f"DISCONNECT_COMPLETE: {client_id} cleaned up (session={session_id}, robot={robot_id})")
                print(f"🧹 Cleaned up connection for client {client_id}")
            else:
                self.logger.warning(f"DISCONNECT_NO_SESSION: {client_id} had no active session")
                print(f"⚠️  No session to clean up for client {client_id}")
    
    def get_stats(self) -> Dict:
        """Get connection handler statistics."""
        
        uptime = time.time() - self.start_time
        
        with self.lock:
            active_connections = len(self.client_sessions)
            
            # Collect session stats
            session_stats = []
            for client_id, session in self.client_sessions.items():
                if hasattr(session, 'get_stats'):
                    session_stats.append(session.get_stats())
        
        return {
            'uptime_seconds': uptime,
            'total_connections': self.total_connections,
            'active_connections': active_connections,
            'total_messages': self.total_messages,
            'messages_per_second': self.total_messages / uptime if uptime > 0 else 0,
            'active_sessions': session_stats,
            'error_stats': self.error_handler.get_error_stats()
        }
    
    def get_session_id_for_client(self, client_id: str) -> Optional[str]:
        """Get the brain session ID for a given client ID."""
        with self.lock:
            session = self.client_sessions.get(client_id)
            if session:
                return session.session_id
            return None
    
    def get_active_clients(self) -> List[str]:
        """Get list of active client IDs."""
        with self.lock:
            return list(self.client_sessions.keys())
    
    def _safe_motor_response(self, client_id: str) -> List[float]:
        """Generate safe motor response for error conditions."""
        # Get motor dimensions from robot
        robot_id = self.client_robots.get(client_id)
        if robot_id:
            robot = self.robot_registry.get_robot(robot_id)
            if robot:
                return [0.0] * len(robot.motor_channels)
        # Default to empty if we can't determine dimensions
        return []