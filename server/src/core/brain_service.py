"""
Brain Service with Integrated Persistence

Manages brain lifecycle and sessions with automatic persistence.
This consolidates the old brain_service.py and brain_service_with_persistence.py
into a single implementation that always includes persistence.
"""

import time
import uuid
from typing import Dict, List, Optional, Any

from .interfaces import (
    IBrainService, IBrainSession, IBrainPool, IAdapterFactory,
    Robot, BrainSessionInfo
)
from ..persistence.integrated_persistence import (
    IntegratedPersistence, initialize_persistence, get_persistence
)
from .brain_telemetry import BrainTelemetryAdapter, SessionTelemetryWrapper


class BrainSession(IBrainSession):
    """
    Handles one robot's interaction with a brain.
    
    This session encapsulates the brain instance, adapters, and persistence.
    """
    
    def __init__(self, session_id: str, robot: Robot, brain, sensory_adapter, motor_adapter, 
                 persistence: Optional[IntegratedPersistence] = None, logger=None):
        self.session_id = session_id
        self.robot = robot
        self.brain = brain
        self.sensory_adapter = sensory_adapter
        self.motor_adapter = motor_adapter
        self.persistence = persistence
        self.logger = logger
        self.created_at = time.time()
        
        # Session statistics
        self.cycles_processed = 0
        self.total_processing_time = 0.0
        
        # Experience tracking
        self.last_sensory_input = None
        self.last_motor_output = None
        self.total_experiences = 0
        
        # Load any existing state on session start
        if self.persistence:
            self.persistence.recover_brain_state(self.brain)
        
        # Create telemetry adapter
        self.telemetry_adapter = BrainTelemetryAdapter(self.brain)
    
    def process_sensory_input(self, raw_sensory: List[float]) -> List[float]:
        """Process sensory input and return motor commands."""
        
        start_time = time.time()
        
        try:
            # Store experience if we have previous data
            if self.last_sensory_input is not None and self.last_motor_output is not None:
                # The current sensory input is the outcome of the previous action
                self._store_experience(
                    sensory_input=self.last_sensory_input,
                    action_taken=self.last_motor_output,
                    outcome=raw_sensory
                )
                self.total_experiences += 1
            
            # Adapt sensory to field space
            field_input = self.sensory_adapter.to_field_space(raw_sensory)
            
            # Process through brain
            field_output = self.brain.process_field_dynamics(field_input)
            
            # Adapt field to motor space
            motor_commands = self.motor_adapter.from_field_space(field_output)
            
            # Store current data for next experience
            self.last_sensory_input = raw_sensory.copy() if isinstance(raw_sensory, list) else raw_sensory
            self.last_motor_output = motor_commands.copy() if isinstance(motor_commands, list) else motor_commands
            
            # Update statistics
            self.cycles_processed += 1
            cycle_time = time.time() - start_time
            self.total_processing_time += cycle_time
            
            # Log brain cycle if logger available
            if self.logger and hasattr(self.logger, 'log_brain_cycle'):
                brain_state = {
                    'field_dimensions': self.brain.get_field_dimensions(),
                    'prediction_confidence': 0.5,  # Could be extracted from brain
                    'active_patterns': 0  # Could be extracted from brain
                }
                self.logger.log_brain_cycle(
                    session_id=self.session_id,
                    brain_state=brain_state,
                    cycle_time_ms=cycle_time * 1000,
                    cycle_num=self.cycles_processed
                )
            
            # Check for auto-save
            if self.persistence:
                self.persistence.check_auto_save(self.brain)
            
            return motor_commands
            
        except Exception as e:
            print(f"❌ Session {self.session_id} processing error: {e}")
            # Return safe motor commands on error
            return [0.0] * len(self.robot.motor_channels)
    
    def _store_experience(self, sensory_input: List[float], action_taken: List[float], 
                         outcome: List[float]) -> str:
        """
        Store an experience in the brain's memory system.
        
        This captures:
        - What the robot sensed
        - What action it took
        - What happened next (outcome)
        """
        # For now, just track the experience occurred
        # The brain itself handles memory formation through field dynamics
        return f"exp_{self.session_id}_{self.total_experiences}"
    
    def get_session_info(self) -> BrainSessionInfo:
        """Get current session information."""
        uptime = time.time() - self.created_at
        avg_cycle_time = (self.total_processing_time / self.cycles_processed 
                         if self.cycles_processed > 0 else 0.0)
        
        return BrainSessionInfo(
            session_id=self.session_id,
            robot_id=self.robot.robot_id,
            brain_dimensions=self.brain.get_field_dimensions(),
            created_at=self.created_at
        )
    
    def get_handshake_response(self) -> List[float]:
        """Get handshake response for this session."""
        # Return brain's field dimensions and version info
        return [
            1.0,  # Protocol version
            float(self.brain.get_field_dimensions()),  # Field dimensions
            float(len(self.robot.sensory_channels)),  # Expected sensory dims
            float(len(self.robot.motor_channels)),     # Expected motor dims
            1.0   # Session active
        ]
    
    def get_session_id(self) -> str:
        """Get unique session identifier."""
        return self.session_id
    
    def close(self):
        """Close session with final save."""
        # Perform shutdown save
        if self.persistence:
            self.persistence.shutdown_save(self.brain)


class BrainService(IBrainService):
    """
    Brain service with integrated persistence.
    
    This service manages brain instances and sessions with automatic:
    - State recovery on startup
    - Periodic auto-saves during operation
    - Shutdown saves
    - Cross-session learning continuity
    """
    
    def __init__(self, brain_pool: IBrainPool, adapter_factory: IAdapterFactory,
                 memory_path: str = "./brain_memory",
                 save_interval_cycles: int = 1000,
                 auto_save: bool = True,
                 enable_logging: bool = True, 
                 log_dir: str = "logs",
                 quiet: bool = False):
        """
        Initialize brain service with persistence.
        
        Args:
            brain_pool: Pool of brain instances
            adapter_factory: Factory for robot-brain adapters
            memory_path: Directory for brain state files
            save_interval_cycles: Save every N brain cycles
            auto_save: Enable automatic periodic saves
            enable_logging: Enable logging system
            log_dir: Directory for log files
        """
        self.brain_pool = brain_pool
        self.adapter_factory = adapter_factory
        
        # Active sessions
        self.sessions: Dict[str, IBrainSession] = {}
        
        # Statistics
        self.total_sessions_created = 0
        self.start_time = time.time()
        
        # Initialize integrated persistence
        self.integrated_persistence = None
        try:
            self.integrated_persistence = initialize_persistence(
                memory_path=memory_path,
                save_interval_cycles=save_interval_cycles,
                auto_save=auto_save,
                use_binary=True  # Use fast binary format
            )
            if not quiet:
                print(f"✅ Integrated persistence initialized")
                print(f"   Memory path: {memory_path}")
                print(f"   Auto-save: every {save_interval_cycles} cycles")
        except Exception as e:
            if not quiet:
                print(f"❌ Failed to initialize integrated persistence: {e}")
                print(f"   Brain will run without persistence")
        
        # Logging setup (if needed)
        self.logging_service = None
        if enable_logging:
            # Could initialize a logging service here if needed
            pass
    
    def create_session(self, robot: Robot) -> IBrainSession:
        """Create a new brain session with persistence support."""
        
        # Generate session ID
        session_id = f"session_{uuid.uuid4().hex[:8]}"
        
        # Get or create brain for this robot's profile
        profile_key = robot.get_profile_key()
        brain = self.brain_pool.get_brain_for_profile(profile_key)
        
        # Get brain dimensions
        brain_dims = brain.get_field_dimensions()
        
        # Create adapters for this specific robot
        sensory_adapter = self.adapter_factory.create_sensory_adapter(robot, brain_dims)
        motor_adapter = self.adapter_factory.create_motor_adapter(robot, brain_dims)
        
        # Create logger for this session if logging is enabled
        session_logger = None
        if self.logging_service:
            session_logger = self.logging_service.create_session_logger(session_id, robot.robot_type)
        
        # Create session with integrated persistence
        session = BrainSession(
            session_id=session_id,
            robot=robot,
            brain=brain,
            sensory_adapter=sensory_adapter,
            motor_adapter=motor_adapter,
            persistence=self.integrated_persistence,
            logger=session_logger
        )
        
        # Track session
        self.sessions[session_id] = session
        self.total_sessions_created += 1
        
        return session
    
    def get_session(self, session_id: str) -> Optional[IBrainSession]:
        """Get existing session by ID."""
        return self.sessions.get(session_id)
    
    def get_session_info(self, session_id: str) -> Optional[BrainSessionInfo]:
        """Get information about a session."""
        session = self.sessions.get(session_id)
        if session:
            return session.get_session_info()
        return None
    
    def list_sessions(self) -> List[BrainSessionInfo]:
        """List all active sessions."""
        return [session.get_session_info() for session in self.sessions.values()]
    
    def close_session(self, session_id: str) -> bool:
        """Close and cleanup a session."""
        session = self.sessions.get(session_id)
        if not session:
            return False
        
        # Close the session (triggers save)
        session.close()
        
        # Remove from active sessions
        del self.sessions[session_id]
        
        return True
    
    def get_brain_for_session(self, session_id: str) -> Optional[Any]:
        """Get the brain instance for a specific session."""
        session = self.sessions.get(session_id)
        if session and hasattr(session, 'brain'):
            return session.brain
        return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get service statistics."""
        uptime = time.time() - self.start_time
        
        # Get persistence stats if available
        persistence_stats = {}
        if self.integrated_persistence:
            persistence_stats = self.integrated_persistence.get_persistence_stats()
        
        # Get active brains info
        active_brains = self.brain_pool.get_active_brains()
        brain_pool_stats = {
            'active_brains': len(active_brains),
            'brain_profiles': list(active_brains.keys())
        }
        
        return {
            'active_sessions': len(self.sessions),
            'total_sessions_created': self.total_sessions_created,
            'uptime_seconds': uptime,
            'brain_pool_stats': brain_pool_stats,
            'persistence_stats': persistence_stats
        }
    
    def get_all_sessions(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all active sessions."""
        sessions_stats = {}
        
        for session_id, session in self.sessions.items():
            avg_cycle_time = (session.total_processing_time / session.cycles_processed 
                             if session.cycles_processed > 0 else 0.0)
            uptime = time.time() - session.created_at
            
            sessions_stats[session_id] = {
                'robot_type': session.robot.robot_type,
                'robot_id': session.robot.robot_id,
                'brain_dimensions': session.brain.get_field_dimensions(),
                'cycles_processed': session.cycles_processed,
                'total_experiences': session.total_experiences,
                'average_cycle_time_ms': avg_cycle_time * 1000,
                'uptime_seconds': uptime,
                'created_at': session.created_at
            }
        
        return sessions_stats
    
    def get_session_telemetry(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get telemetry for a specific session"""
        session = self.sessions.get(session_id)
        if session and hasattr(session, 'telemetry_adapter'):
            return session.telemetry_adapter.get_telemetry_summary()
        return None
    
    def get_all_telemetry(self) -> Dict[str, Dict[str, Any]]:
        """Get telemetry for all active sessions"""
        telemetry = {}
        for session_id, session in self.sessions.items():
            # Try to get brain state directly for SimplifiedUnifiedBrain
            if hasattr(session, 'brain') and session.brain:
                brain = session.brain
                if hasattr(brain, '_create_brain_state'):
                    # SimplifiedUnifiedBrain
                    telemetry[session_id] = brain._create_brain_state()
                elif hasattr(brain, 'get_brain_state'):
                    # Legacy brain
                    telemetry[session_id] = brain.get_brain_state()
            elif hasattr(session, 'telemetry_adapter'):
                # Old telemetry adapter
                telemetry[session_id] = session.telemetry_adapter.get_telemetry_summary()
        return telemetry
    
    def get_detailed_telemetry(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed telemetry data for a session"""
        session = self.sessions.get(session_id)
        if not session:
            return None
            
        # Try to get brain state directly for SimplifiedUnifiedBrain
        if hasattr(session, 'brain') and session.brain:
            brain = session.brain
            if hasattr(brain, '_create_brain_state'):
                # SimplifiedUnifiedBrain
                brain_state = brain._create_brain_state()
                # Add evolution state if available
                if hasattr(brain, 'get_evolution_state'):
                    brain_state['evolution_state'] = brain.get_evolution_state()
                return brain_state
            elif hasattr(brain, 'get_brain_state'):
                # Legacy brain
                return brain.get_brain_state()
        elif session and hasattr(session, 'telemetry_adapter'):
            # Old telemetry adapter
            telemetry = session.telemetry_adapter.get_telemetry()
            return {
                'brain_cycles': telemetry.brain_cycles,
                'field_energy': telemetry.field_energy,
                'prediction_confidence': telemetry.prediction_confidence,
                'cognitive_mode': telemetry.cognitive_mode,
                'memory_regions': telemetry.memory_regions,
                'experiences': telemetry.experiences_stored,
                'constraints': telemetry.active_constraints,
                'phase': telemetry.phase_state,
                'blend_ratio': telemetry.blended_reality_ratio,
                'prediction_error': telemetry.prediction_error,
                'prediction_history': telemetry.prediction_history,
                'improvement_rate': telemetry.improvement_rate,
                'timestamp': telemetry.timestamp
            }
        return None
    
    def shutdown(self):
        """Shutdown the brain service and save all brain states."""
        print("   Shutting down brain service...")
        
        # Save all brain states
        for session_id, session in self.sessions.items():
            if session.brain:
                print(f"   💾 Saving state for session {session_id}")
                self.persistence.save_brain_state(session.brain, force=True)
        
        # Stop persistence if it has a stop method
        if hasattr(self.persistence, 'stop'):
            self.persistence.stop()
        
        print("   ✅ Brain service shutdown complete")