"""
Simplified Brain Factory - Unified Field Intelligence Only

This factory creates and manages the UnifiedFieldBrain - the only brain implementation.
All legacy brain types have been removed for simplicity and maintainability.
"""

import os
import time
from typing import List, Dict, Tuple, Optional, Any

# Direct import of the unified brain implementation
try:
    from .brains.field.core_brain import UnifiedFieldBrain
    from .utils.cognitive_autopilot import CognitiveAutopilot
    from .utils.brain_logger import BrainLogger
    # from .persistence import PersistenceManager, PersistenceConfig  # Temporarily disabled
except ImportError:
    from brains.field.core_brain import UnifiedFieldBrain
    from utils.cognitive_autopilot import CognitiveAutopilot
    from utils.brain_logger import BrainLogger
    # from persistence import PersistenceManager, PersistenceConfig  # Temporarily disabled


class BrainFactory:
    """
    Simplified Factory for UnifiedFieldBrain only.
    
    All legacy brain types have been removed. This factory now only creates
    and manages the 37D UnifiedFieldBrain implementation.
    """
    
    def __init__(self, config=None, enable_logging=True, log_session_name=None, quiet_mode=False):
        """Initialize the brain factory with UnifiedFieldBrain."""
        
        # Store config
        self.config = config or {}
        self.quiet_mode = quiet_mode
        
        # Configure GPU memory management
        try:
            from .config.gpu_memory_manager import configure_gpu_memory
        except ImportError:
            from config.gpu_memory_manager import configure_gpu_memory
        configure_gpu_memory(self.config)
        
        # Get brain configuration
        brain_config = self.config.get('brain', {})
        
        # Create the UnifiedFieldBrain - the only brain implementation
        if not quiet_mode:
            print(f"🧠 Creating UnifiedFieldBrain (37D physics-organized field intelligence)")
        
        self.brain = UnifiedFieldBrain(
            spatial_resolution=brain_config.get('field_spatial_resolution', None),  # Let hardware adaptation decide
            temporal_window=brain_config.get('field_temporal_window', 10.0),
            field_evolution_rate=brain_config.get('field_evolution_rate', 0.1),
            constraint_discovery_rate=brain_config.get('constraint_discovery_rate', 0.15),
            quiet_mode=quiet_mode
        )
        
        # Initialize cognitive autopilot for adaptive processing
        self.cognitive_autopilot = CognitiveAutopilot()
        
        # Initialize logging if enabled
        if enable_logging:
            self.logger = BrainLogger(
                log_session_name or f"brain_session_{int(time.time())}", 
                config=self.config
            )
        else:
            self.logger = None
        
        # Initialize persistence
        if config.get('memory', {}).get('enable_persistence', True):
            try:
                from .persistence.persistence_manager import PersistenceManager
            except ImportError:
                from persistence.persistence_manager import PersistenceManager
            memory_path = config.get('memory', {}).get('persistent_memory_path', './server/robot_memory')
            self.persistence_manager = PersistenceManager(memory_path=memory_path)
        else:
            self.persistence_manager = None
        
        # Performance tracking
        self.total_cycles = 0
        self.total_experiences = 0
        self.total_predictions = 0
        self.brain_start_time = time.time()
        
        if not quiet_mode:
            print(f"✅ BrainFactory initialized with 37D UnifiedFieldBrain")
            print(f"   Field dimensions: {self.brain.total_dimensions}D")
            print(f"   Persistence: {'enabled' if self.persistence_manager else 'disabled'}")
    
    def process_sensory_input(self, sensory_input: List[float]) -> Tuple[List[float], Dict[str, Any]]:
        """Process sensory input through the unified field brain."""
        
        # Process through UnifiedFieldBrain
        action, brain_state = self.brain.process_robot_cycle(sensory_input)
        
        # Track performance
        self.total_cycles += 1
        self.total_experiences += 1
        
        # Log if enabled (using simplified approach)
        if self.logger:
            self.logger.log_brain_state(self, self.total_experiences)
        
        # Save state if persistence enabled
        if self.persistence_manager and self.total_cycles % 100 == 0:
            try:
                self.persistence_manager.save_brain_state_incremental(self)
            except Exception as e:
                if not self.quiet_mode:
                    print(f"⚠️ Persistence save failed: {e}")
        
        # Combine states and add missing fields for TCP server compatibility
        combined_state = {
            **brain_state,
            'total_cycles': self.total_cycles,
            'total_experiences': self.total_experiences,
            'uptime_seconds': time.time() - self.brain_start_time,
            'prediction_method': brain_state.get('brain_type', 'unified_field'),
            'prediction_confidence': brain_state.get('last_action_confidence', 0.5)
        }
        
        return action, combined_state
    
    def store_experience(self, sensory_input: List[float], action_taken: List[float], 
                        outcome: List[float], predicted_action: List[float]) -> str:
        """Store experience for learning - simplified implementation for UnifiedFieldBrain."""
        
        # The UnifiedFieldBrain handles experience storage internally through field dynamics
        # This method provides compatibility with the TCP server's experience tracking
        
        # Generate a simple experience ID for tracking
        experience_id = f"exp_{self.total_experiences}_{int(time.time() * 1000) % 10000}"
        
        # The field brain naturally incorporates experiences through field evolution
        # No explicit storage needed - field topology encodes all experiences
        
        return experience_id
    
    def get_brain_stats(self) -> Dict[str, Any]:
        """Get comprehensive brain statistics."""
        brain_stats = {}
        
        # Get UnifiedFieldBrain stats
        if hasattr(self.brain, 'get_brain_stats'):
            brain_stats = self.brain.get_brain_stats()
        
        # Add factory-level stats with TCP server compatibility
        factory_stats = {
            'factory': {
                'total_cycles': self.total_cycles,
                'total_experiences': self.total_experiences,
                'total_predictions': self.total_predictions,
                'uptime_seconds': time.time() - self.brain_start_time,
                'cycles_per_minute': self.total_cycles / max(1, (time.time() - self.brain_start_time) / 60),
                'experiences_per_minute': self.total_experiences / max(1, (time.time() - self.brain_start_time) / 60),
                'architecture': 'unified_field_37D',
            },
            # Add brain_summary for TCP server compatibility
            'brain_summary': {
                'total_cycles': self.total_cycles,
                'total_experiences': self.total_experiences,
                'brain_type': 'unified_field',
                'field_dimensions': self.brain.total_dimensions,
                'uptime_seconds': time.time() - self.brain_start_time
            }
        }
        
        return {**factory_stats, **brain_stats}
    
    def get_brain_state_for_persistence(self) -> Dict[str, Any]:
        """Get brain state for persistence - simplified for UnifiedFieldBrain only."""
        return {
            'brain_type': 'unified_field',
            'field_dimensions': self.brain.total_dimensions,
            'brain_cycles': self.brain.brain_cycles,
            'total_factory_cycles': self.total_cycles,
            'field_parameters': {
                'spatial_resolution': self.brain.spatial_resolution,
                'temporal_window': self.brain.temporal_window,
                'field_evolution_rate': self.brain.field_evolution_rate,
                'constraint_discovery_rate': self.brain.constraint_discovery_rate,
            },
            # Note: We don't serialize the full field tensor - it's too large
            # The brain will reconstruct its field state from parameters
        }
    
    def restore_brain_state(self, state: Dict[str, Any]) -> bool:
        """Restore brain state - simplified for UnifiedFieldBrain only."""
        try:
            if state.get('brain_type') != 'unified_field':
                print(f"⚠️ Cannot restore non-unified brain state: {state.get('brain_type')}")
                return False
            
            # Restore basic counters
            self.total_cycles = state.get('total_factory_cycles', 0)
            
            # The UnifiedFieldBrain will rebuild its field state naturally
            # through normal operation - no complex restoration needed
            
            if not self.quiet_mode:
                print(f"✅ Brain state restored: {state.get('field_dimensions')}D field")
            
            return True
            
        except Exception as e:
            print(f"⚠️ Brain state restoration failed: {e}")
            return False
    
    def shutdown(self):
        """Clean shutdown of brain factory."""
        try:
            # Save final state if persistence enabled
            if self.persistence_manager:
                self.persistence_manager.save_brain_state_blocking(self)
                
            # Close logger
            if self.logger:
                self.logger.close_session()
                
            # Shutdown brain
            self.brain.shutdown()
            
            # Shutdown persistence
            if self.persistence_manager:
                self.persistence_manager.shutdown()
                
            if not self.quiet_mode:
                print(f"🔌 BrainFactory shutdown complete")
                
        except Exception as e:
            print(f"⚠️ Shutdown error: {e}")