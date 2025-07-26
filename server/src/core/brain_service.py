"""
Brain Service implementation.

Manages brain lifecycle and sessions, coordinating between brains,
adapters, and robot configurations.
"""

import time
import uuid
from typing import Dict, List, Optional

from .interfaces import (
    IBrainService, IBrainSession, IBrainPool, IAdapterFactory,
    Robot, BrainSessionInfo
)


class BrainSession(IBrainSession):
    """
    Handles one robot's interaction with a brain.
    
    This session encapsulates the brain instance and the adapters
    needed to translate between robot and brain spaces.
    """
    
    def __init__(self, session_id: str, robot: Robot, brain, sensory_adapter, motor_adapter, 
                 persistence_manager=None, logger=None):
        self.session_id = session_id
        self.robot = robot
        self.brain = brain
        self.sensory_adapter = sensory_adapter
        self.motor_adapter = motor_adapter
        self.persistence_manager = persistence_manager
        self.logger = logger
        self.created_at = time.time()
        
        # Session statistics
        self.cycles_processed = 0
        self.total_processing_time = 0.0
        self.last_persistence_save = 0
        
        # Experience tracking
        self.last_sensory_input = None
        self.last_motor_output = None
        self.total_experiences = 0
    
    def process_sensory_input(self, raw_sensory: List[float]) -> List[float]:
        """Process sensory input and return motor commands."""
        
        start_time = time.time()
        
        try:
            # Store experience if we have previous data
            if self.last_sensory_input is not None and self.last_motor_output is not None:
                # The current sensory input is the outcome of the previous action
                experience_id = self._store_experience(
                    sensory_input=self.last_sensory_input,
                    action_taken=self.last_motor_output,
                    outcome=raw_sensory
                )
                self.total_experiences += 1
            
            # Adapt sensory to field space
            field_input = self.sensory_adapter.to_field_space(raw_sensory)
            
            # Process through brain (brain knows nothing about robots!)
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
            
            # Save brain state periodically if persistence is enabled
            if self.persistence_manager and self.cycles_processed % 100 == 0:
                try:
                    # Create a brain wrapper that matches old BrainFactory interface
                    # This allows the existing persistence system to work
                    class BrainFactoryAdapter:
                        def __init__(self, brain, session):
                            self.brain = brain
                            self.total_cycles = session.cycles_processed
                            self.brain_cycles = session.cycles_processed
                            
                        def get_brain_state_for_persistence(self):
                            return {
                                'brain_type': 'unified_field',
                                'field_dimensions': getattr(self.brain, 'field_dimensions', 36),
                                'brain_cycles': self.brain_cycles,
                                'total_factory_cycles': self.total_cycles,
                                'field_parameters': {}
                            }
                    
                    brain_adapter = BrainFactoryAdapter(self.brain, self)
                    # Use incremental save with the adapter
                    self.persistence_manager.save_brain_state_incremental(brain_adapter)
                    self.last_persistence_save = self.cycles_processed
                except Exception as e:
                    print(f"⚠️ Persistence save failed for session {self.session_id}: {e}")
            
            return motor_commands
            
        except Exception as e:
            print(f"❌ Error in brain session {self.session_id}: {e}")
            print(f"   Error type: {type(e).__name__}")
            print(f"   Sensory input type: {type(raw_sensory)}, length: {len(raw_sensory) if hasattr(raw_sensory, '__len__') else 'N/A'}")
            import traceback
            traceback.print_exc()
            # Return safe motor commands (zeros)
            return [0.0] * len(self.robot.motor_channels)
    
    def get_handshake_response(self) -> List[float]:
        """Get handshake response for this session."""
        
        # Response format:
        # [brain_version, accepted_sensory_dim, accepted_action_dim, gpu_available, brain_capabilities]
        
        brain_dims = self.brain.get_field_dimensions()
        
        response = [
            4.0,  # Brain version
            float(len(self.robot.sensory_channels)),  # Accepted sensory dimensions
            float(len(self.robot.motor_channels)),   # Accepted motor dimensions
            1.0,  # GPU available (could check actual availability)
            15.0  # Brain capabilities mask (all capabilities)
        ]
        
        return response
    
    def get_session_id(self) -> str:
        """Get unique session identifier."""
        return self.session_id
    
    def get_stats(self) -> Dict:
        """Get session statistics."""
        avg_cycle_time = (self.total_processing_time / self.cycles_processed 
                         if self.cycles_processed > 0 else 0.0)
        
        return {
            'session_id': self.session_id,
            'robot_id': self.robot.robot_id,
            'robot_type': self.robot.robot_type,
            'brain_dimensions': self.brain.get_field_dimensions(),
            'cycles_processed': self.cycles_processed,
            'average_cycle_time_ms': avg_cycle_time * 1000,
            'uptime_seconds': time.time() - self.created_at,
            'total_experiences': self.total_experiences
        }
    
    def _store_experience(self, sensory_input: List[float], action_taken: List[float], 
                         outcome: List[float]) -> str:
        """Store an experience sequence for learning."""
        # Generate experience ID
        experience_id = f"exp_{self.session_id}_{self.total_experiences}_{int(time.time() * 1000) % 10000}"
        
        # In the field brain paradigm, experiences are encoded in the field dynamics
        # The brain learns through field evolution, not explicit experience replay
        # This method exists for compatibility and tracking
        
        # Could optionally log experiences for analysis
        if self.cycles_processed < 5:  # Log first few
            print(f"💾 {self.robot.robot_id}: Experience {experience_id} stored (total: {self.total_experiences + 1})")
        
        # Log experience if logger available
        if self.logger and hasattr(self.logger, 'log_experience'):
            self.logger.log_experience(
                session_id=self.session_id,
                experience_id=experience_id,
                sensory_dim=len(sensory_input),
                motor_dim=len(action_taken)
            )
        
        return experience_id


class BrainService(IBrainService):
    """
    Manages brain lifecycle and sessions.
    
    This service coordinates the creation of brain sessions, ensuring
    each robot gets the appropriate brain and adapters.
    """
    
    def __init__(self, brain_pool: IBrainPool, adapter_factory: IAdapterFactory, 
                 enable_persistence: bool = True, memory_path: str = "./server/robot_memory",
                 enable_logging: bool = True, log_dir: str = "logs"):
        self.brain_pool = brain_pool
        self.adapter_factory = adapter_factory
        self.active_sessions: Dict[str, BrainSession] = {}
        
        # Initialize persistence if enabled
        self.persistence_manager = None
        if enable_persistence:
            try:
                from ..persistence.persistence_manager import PersistenceManager
                self.persistence_manager = PersistenceManager(memory_path=memory_path)
                print(f"💾 Persistence subsystem initialized at {memory_path}")
            except Exception as e:
                print(f"⚠️ Failed to initialize persistence: {e}")
                self.persistence_manager = None
        
        # Initialize logging if enabled
        self.logging_service = None
        if enable_logging:
            try:
                from .logging_service import LoggingService
                self.logging_service = LoggingService(log_dir=log_dir, quiet_mode=True)
            except Exception as e:
                print(f"⚠️ Failed to initialize logging: {e}")
                self.logging_service = None
    
    def create_session(self, robot: Robot) -> IBrainSession:
        """Create a new brain session for a robot."""
        
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
        
        # Create session with persistence and logging support
        session = BrainSession(
            session_id=session_id,
            robot=robot,
            brain=brain,
            sensory_adapter=sensory_adapter,
            motor_adapter=motor_adapter,
            persistence_manager=self.persistence_manager,
            logger=self.logging_service
        )
        
        # Track session
        self.active_sessions[session_id] = session
        
        print(f"🤝 Created brain session {session_id} for {robot.robot_type} robot")
        print(f"   Robot: {len(robot.sensory_channels)} sensors → {len(robot.motor_channels)} motors")
        print(f"   Brain: {brain_dims}D unified field")
        
        return session
    
    def get_session_info(self, session_id: str) -> Optional[BrainSessionInfo]:
        """Get information about a session."""
        
        session = self.active_sessions.get(session_id)
        if not session:
            return None
        
        return BrainSessionInfo(
            session_id=session.session_id,
            robot_id=session.robot.robot_id,
            brain_dimensions=session.brain.get_field_dimensions(),
            created_at=session.created_at
        )
    
    def close_session(self, session_id: str) -> None:
        """Close a brain session."""
        
        if session_id in self.active_sessions:
            session = self.active_sessions[session_id]
            stats = session.get_stats()
            
            print(f"👋 Closing brain session {session_id}")
            print(f"   Processed {stats['cycles_processed']} cycles")
            print(f"   Average cycle time: {stats['average_cycle_time_ms']:.1f}ms")
            
            # Close session logger
            if self.logging_service:
                self.logging_service.close_session(session_id)
            
            del self.active_sessions[session_id]
    
    def get_all_sessions(self) -> Dict[str, Dict]:
        """Get information about all active sessions."""
        
        sessions_info = {}
        for session_id, session in self.active_sessions.items():
            sessions_info[session_id] = session.get_stats()
        
        return sessions_info
    
    def shutdown(self):
        """Shutdown the brain service and persistence."""
        # Save final state for all sessions
        if self.persistence_manager:
            for session_id, session in self.active_sessions.items():
                try:
                    # Create adapter for persistence
                    class BrainFactoryAdapter:
                        def __init__(self, brain, session):
                            self.brain = brain
                            self.total_cycles = session.cycles_processed
                            self.brain_cycles = session.cycles_processed
                            
                        def get_brain_state_for_persistence(self):
                            return {
                                'brain_type': 'unified_field',
                                'field_dimensions': getattr(self.brain, 'field_dimensions', 36),
                                'brain_cycles': self.brain_cycles,
                                'total_factory_cycles': self.total_cycles,
                                'field_parameters': {}
                            }
                    
                    brain_adapter = BrainFactoryAdapter(session.brain, session)
                    self.persistence_manager.save_brain_state_blocking(brain_adapter)
                except Exception as e:
                    print(f"⚠️ Final save failed for session {session_id}: {e}")
            
            # Shutdown persistence
            self.persistence_manager.shutdown()
            print("💾 Persistence subsystem shutdown complete")
        
        # Shutdown logging
        if self.logging_service:
            self.logging_service.shutdown()