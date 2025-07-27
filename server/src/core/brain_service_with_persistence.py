"""
Enhanced Brain Service with Integrated Persistence

This extends the brain service to include the new integrated persistence system
that works with the dynamic brain architecture.
"""

import time
import uuid
from typing import Dict, List, Optional

from .brain_service import BrainService, BrainSession
from .interfaces import (
    IBrainService, IBrainSession, IBrainPool, IAdapterFactory,
    Robot, BrainSessionInfo
)
from ..persistence.integrated_persistence import (
    IntegratedPersistence, initialize_persistence, get_persistence
)


class BrainSessionWithPersistence(BrainSession):
    """Enhanced brain session that includes automatic persistence."""
    
    def __init__(self, session_id: str, robot: Robot, brain, sensory_adapter, motor_adapter, 
                 persistence: Optional[IntegratedPersistence] = None, logger=None):
        # Initialize base session
        super().__init__(
            session_id=session_id,
            robot=robot,
            brain=brain,
            sensory_adapter=sensory_adapter,
            motor_adapter=motor_adapter,
            persistence_manager=None,  # We handle persistence differently
            logger=logger
        )
        
        # Store integrated persistence
        self.persistence = persistence
        
        # Try to recover state on session creation
        if self.persistence:
            try:
                self.persistence.recover_brain_state(brain)
            except Exception as e:
                print(f"⚠️ Could not recover brain state: {e}")
    
    def process_sensory_input(self, raw_sensory: List[float]) -> List[float]:
        """Process sensory input with automatic persistence checks."""
        
        # Process normally
        motor_commands = super().process_sensory_input(raw_sensory)
        
        # Check for auto-save
        if self.persistence:
            self.persistence.check_auto_save(self.brain)
        
        return motor_commands
    
    def close(self):
        """Close session with final save."""
        # Perform shutdown save
        if self.persistence:
            self.persistence.shutdown_save(self.brain)
        
        # Call parent close if it exists
        if hasattr(super(), 'close'):
            super().close()


class BrainServiceWithPersistence(BrainService):
    """
    Enhanced brain service with integrated persistence.
    
    This service adds:
    - Automatic state recovery on startup
    - Periodic auto-saves during operation
    - Shutdown saves
    - Cross-session learning continuity
    """
    
    def __init__(self, brain_pool: IBrainPool, adapter_factory: IAdapterFactory,
                 enable_persistence: bool = True, 
                 memory_path: str = "./brain_memory",
                 save_interval_cycles: int = 1000,
                 auto_save: bool = True,
                 enable_logging: bool = True, 
                 log_dir: str = "logs"):
        """
        Initialize enhanced brain service.
        
        Args:
            brain_pool: Pool of brain instances
            adapter_factory: Factory for robot-brain adapters
            enable_persistence: Enable the integrated persistence system
            memory_path: Directory for brain state files
            save_interval_cycles: Save every N brain cycles
            auto_save: Enable automatic periodic saves
            enable_logging: Enable logging system
            log_dir: Directory for log files
        """
        # Initialize base service (without old persistence)
        super().__init__(
            brain_pool=brain_pool,
            adapter_factory=adapter_factory,
            enable_persistence=False,  # Disable old persistence
            enable_logging=enable_logging,
            log_dir=log_dir
        )
        
        # Initialize new integrated persistence
        self.integrated_persistence = None
        if enable_persistence:
            try:
                self.integrated_persistence = initialize_persistence(
                    memory_path=memory_path,
                    save_interval_cycles=save_interval_cycles,
                    auto_save=auto_save
                )
                print(f"✅ Integrated persistence initialized")
                print(f"   Memory path: {memory_path}")
                print(f"   Auto-save: every {save_interval_cycles} cycles")
            except Exception as e:
                print(f"❌ Failed to initialize integrated persistence: {e}")
    
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
        
        # Create enhanced session with integrated persistence
        session = BrainSessionWithPersistence(
            session_id=session_id,
            robot=robot,
            brain=brain,
            sensory_adapter=sensory_adapter,
            motor_adapter=motor_adapter,
            persistence=self.integrated_persistence,
            logger=self.logging_service
        )
        
        # Track session
        self.active_sessions[session_id] = session
        
        print(f"🤝 Created persistent brain session {session_id} for {robot.robot_type} robot")
        print(f"   Robot: {len(robot.sensory_channels)} sensors → {len(robot.motor_channels)} motors")
        print(f"   Brain: {brain_dims}D unified field")
        if self.integrated_persistence:
            stats = self.integrated_persistence.get_persistence_stats()
            print(f"   Persistence: Session #{stats.get('session_id', 'unknown')}")
        
        return session
    
    def close_session(self, session_id: str) -> None:
        """Close a brain session with persistence save."""
        
        if session_id in self.active_sessions:
            session = self.active_sessions[session_id]
            
            # Perform shutdown save if it's our enhanced session
            if isinstance(session, BrainSessionWithPersistence):
                session.close()
            
            # Continue with normal close
            super().close_session(session_id)
    
    def shutdown(self):
        """Shutdown the brain service, saving all active sessions."""
        print("🔚 Shutting down brain service...")
        
        # Close all sessions (which triggers saves)
        session_ids = list(self.active_sessions.keys())
        for session_id in session_ids:
            self.close_session(session_id)
        
        print("✅ Brain service shutdown complete")
    
    def get_persistence_stats(self) -> Optional[Dict]:
        """Get persistence statistics."""
        if self.integrated_persistence:
            return self.integrated_persistence.get_persistence_stats()
        return None