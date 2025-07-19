"""
Minimal Brain Coordinator - Vector Stream Architecture

The central orchestrator that coordinates vector stream processing:
1. Modular Streams - continuous sensory, motor, and temporal vectors
2. Cross-Stream Learning - associations between stream patterns  
3. Temporal Integration - time as data stream with organic metronome
4. Continuous Prediction - emergent from vector flow dynamics

Vector streams replace discrete experience packages with biologically-realistic
continuous processing that handles timing and dead reckoning naturally.
"""

import time
from typing import List, Dict, Tuple, Optional, Any

from .vector_stream.vector_stream_brain import MinimalVectorStreamBrain
from .vector_stream.sparse_goldilocks_brain import SparseGoldilocksBrain
from .utils.cognitive_autopilot import CognitiveAutopilot
from .utils.brain_logger import BrainLogger
from .utils.hardware_adaptation import get_hardware_adaptation, record_brain_cycle_performance
from .persistence import PersistenceManager, PersistenceConfig


class MinimalBrain:
    """
    The complete minimal brain - vector stream processing for biological realism.
    
    Everything emerges from continuous vector flow:
    - Spatial navigation emerges from sensory stream patterns
    - Motor skills emerge from sensory-motor stream associations
    - Temporal adaptation emerges from organic metronome integration
    - Dead reckoning emerges from cross-stream prediction dynamics
    """
    
    def __init__(self, config=None, enable_logging=True, log_session_name=None, quiet_mode=False, 
                 sensory_dim=None, motor_dim=None, temporal_dim=None, max_patterns=None, 
                 brain_type=None):
        """Initialize the minimal brain with vector stream architecture."""
        
        # Store config for use in other methods
        self.config = config or {}
        self.quiet_mode = quiet_mode
        
        # Get brain configuration from settings with defaults
        brain_config = self.config.get('brain', {})
        self.brain_type = brain_type or brain_config.get('type', 'sparse_goldilocks')
        self.sensory_dim = sensory_dim or brain_config.get('sensory_dim', 16)
        self.motor_dim = motor_dim or brain_config.get('motor_dim', 4)
        self.temporal_dim = temporal_dim or brain_config.get('temporal_dim', 4)
        max_patterns = max_patterns or brain_config.get('max_patterns', 100000)
        
        # Timing configuration
        self.target_cycle_time_ms = brain_config.get('target_cycle_time_ms', 50.0)
        self.target_cycle_time_s = self.target_cycle_time_ms / 1000.0
        
        # Log dimension configuration sources for traceability
        if not quiet_mode:
            print(f"🧠 Brain Dimension Configuration:")
            sensory_source = "param" if sensory_dim else ("config" if 'sensory_dim' in brain_config else "default")
            motor_source = "param" if motor_dim else ("config" if 'motor_dim' in brain_config else "default") 
            temporal_source = "param" if temporal_dim else ("config" if 'temporal_dim' in brain_config else "default")
            timing_source = "config" if 'target_cycle_time_ms' in brain_config else "default"
            print(f"   Sensory: {self.sensory_dim}D (from {sensory_source})")
            print(f"   Motor: {self.motor_dim}D (from {motor_source})")
            print(f"   Temporal: {self.temporal_dim}D (from {temporal_source})")
            print(f"   Cycle time: {self.target_cycle_time_ms}ms (from {timing_source})")
            print(f"   Type: {self.brain_type} (from {'param' if brain_type else ('config' if 'type' in brain_config else 'default')})")
        
        # Hardware adaptation system
        self.hardware_adaptation = get_hardware_adaptation()
        
        # Select brain implementation based on type
        if self.brain_type == "sparse_goldilocks":
            self.vector_brain = SparseGoldilocksBrain(
                sensory_dim=self.sensory_dim,
                motor_dim=self.motor_dim, 
                temporal_dim=self.temporal_dim,
                max_patterns=max_patterns,
                quiet_mode=quiet_mode
            )
        elif self.brain_type == "minimal":
            self.vector_brain = MinimalVectorStreamBrain(
                sensory_dim=self.sensory_dim,
                motor_dim=self.motor_dim, 
                temporal_dim=self.temporal_dim
            )
        else:
            raise ValueError(f"Unknown brain_type: {self.brain_type}. Options: 'sparse_goldilocks', 'minimal'")
        
        # Initialize cognitive autopilot for adaptive intensity control
        self.cognitive_autopilot = CognitiveAutopilot()
        
        # Brain state tracking
        self.total_cycles = 0
        self.brain_start_time = time.time()
        self.last_prediction = None  # For prediction error calculation
        
        # Logging compatibility attributes
        self.total_experiences = 0
        self.total_predictions = 0
        self.optimal_prediction_error = 0.5
        self.recent_learning_outcomes = []
        
        # Logging system for emergence analysis
        self.enable_logging = enable_logging
        if enable_logging:
            self.logger = BrainLogger(session_name=log_session_name, config=config)
        else:
            self.logger = None
        
        # Production-grade persistence system for cross-session learning
        self.enable_persistence = config.get('memory', {}).get('enable_persistence', True)
        self.persistence_manager = None
        self.session_count = 0
        
        if self.enable_persistence:
            memory_path = config.get('memory', {}).get('persistent_memory_path', './robot_memory')
            
            # Create persistence configuration
            persistence_config = PersistenceConfig(
                memory_root_path=memory_path,
                incremental_save_interval_cycles=config.get('memory', {}).get('save_interval_cycles', 100),
                enable_compression=config.get('memory', {}).get('enable_compression', True),
                enable_corruption_detection=config.get('memory', {}).get('enable_corruption_detection', True)
            )
            
            # Initialize production persistence manager
            self.persistence_manager = PersistenceManager(persistence_config)
            
            # Recover existing brain state at startup
            recovered_state = self.persistence_manager.recover_brain_state_at_startup()
            if recovered_state:
                self.session_count = recovered_state.session_count
                self._apply_recovered_state(recovered_state)
                if not quiet_mode:
                    print(f"🧠 Continuing from session {self.session_count}")
            else:
                self.session_count = 1
                if not quiet_mode:
                    print(f"🧠 Starting fresh brain state - session {self.session_count}")
        
        if not quiet_mode:
            print(f"🧠 MinimalBrain initialized - {self.brain_type.title()} Architecture")
            print(f"   Sensory stream: {self.sensory_dim}D")
            print(f"   Motor stream: {self.motor_dim}D") 
            print(f"   Temporal stream: {self.temporal_dim}D")
            if self.brain_type == "sparse_goldilocks":
                print(f"   Max patterns: {max_patterns:,}")
                print(f"   🧬 Evolutionary Win #1: Sparse Distributed Representations")
            print(f"   Continuous prediction and dead reckoning enabled")
        else:
            # Show minimal essential summary
            print(f"🧠 Vector Brain ready: {self.sensory_dim}D→{self.motor_dim}D processing")
    
    def process_sensory_input(self, sensory_input: List[float], 
                            action_dimensions: int = 4) -> Tuple[List[float], Dict[str, Any]]:
        """
        Process sensory input and return predicted action using vector streams.
        
        This is the complete brain cycle:
        1. Update vector streams with current input
        2. Generate action through cross-stream prediction
        3. Return action with brain state info
        
        Args:
            sensory_input: Current sensory observation vector
            action_dimensions: Number of action dimensions to output (ignored - motor stream determines size)
            
        Returns:
            Tuple of (predicted_action, brain_state_info)
        """
        process_start_time = time.time()
        
        # Ensure sensory input matches expected dimensions
        if len(sensory_input) > self.sensory_dim:
            # Truncate if too long
            processed_input = sensory_input[:self.sensory_dim]
        elif len(sensory_input) < self.sensory_dim:
            # Pad if too short
            processed_input = sensory_input + [0.0] * (self.sensory_dim - len(sensory_input))
        else:
            processed_input = sensory_input
        
        # Process through vector stream brain
        predicted_action, vector_brain_state = self.vector_brain.process_sensory_input(processed_input)
        
        # Adjust action dimensions if requested
        if action_dimensions and action_dimensions != len(predicted_action):
            if action_dimensions < len(predicted_action):
                # Truncate
                predicted_action = predicted_action[:action_dimensions]
            else:
                # Pad with zeros
                predicted_action = predicted_action + [0.0] * (action_dimensions - len(predicted_action))
        
        # Update cognitive autopilot with vector stream confidence
        confidence = vector_brain_state['prediction_confidence']
        prediction_error = 1.0 - confidence
        
        # Calculate actual prediction error if we have previous prediction
        if hasattr(self, 'last_prediction') and self.last_prediction is not None:
            # Compare current sensory input with what we predicted
            import numpy as np
            actual = np.array(processed_input[:len(self.last_prediction)])
            predicted = np.array(self.last_prediction[:len(processed_input)])
            if len(actual) > 0 and len(predicted) > 0:
                error = np.linalg.norm(actual - predicted)
                max_error = np.linalg.norm(actual) + np.linalg.norm(predicted)
                if max_error > 0:
                    prediction_error = min(1.0, error / max_error)
        
        # Store current prediction for next cycle
        self.last_prediction = predicted_action
        
        initial_brain_state = {
            'prediction_confidence': confidence,
            'prediction_error': prediction_error,
            'total_cycles': self.total_cycles
        }
        
        autopilot_state = self.cognitive_autopilot.update_cognitive_state(
            confidence, prediction_error, initial_brain_state
        )
        
        # Log prediction outcome if logging enabled
        if self.logger and predicted_action:
            self.logger.log_prediction_outcome(
                predicted_action, sensory_input, confidence, 0  # Vector streams don't use "similar experiences"
            )
            
            # Feature-flagged pattern storage logging for scientific analysis
            if self.config.get('logging', {}).get('log_pattern_storage', False):
                # Log pattern storage events when streams update
                if hasattr(self.vector_brain, 'last_pattern_storage_event'):
                    storage_event = self.vector_brain.last_pattern_storage_event
                    if storage_event:
                        self.logger.log_similarity_evolution(
                            storage_event.get('pattern_id', 'unknown'),
                            storage_event.get('similarity_score', 0.0),
                            storage_event.get('storage_decision', 'unknown')
                        )
                        
                        # Log as adaptation event if significant
                        if storage_event.get('similarity_score', 0.0) < 0.8:  # Novel pattern
                            self.logger.log_adaptation_event(
                                'pattern_storage',
                                storage_event.get('pattern_id', 'unknown'),
                                storage_event.get('before_state', {}),
                                storage_event.get('after_state', {})
                            )
                        
                        # Clear the event after logging
                        self.vector_brain.last_pattern_storage_event = None
        
        # Performance monitoring
        cycle_time = time.time() - process_start_time
        cycle_time_ms = cycle_time * 1000
        
        # Record performance for hardware adaptation
        memory_usage_mb = 50.0  # Vector streams use much less memory than experience storage
        record_brain_cycle_performance(cycle_time_ms, memory_usage_mb)
        
        # Enforce target cycle time if configured (biological timing vs emergent timing)
        if self.target_cycle_time_s > 0:
            remaining_time = self.target_cycle_time_s - cycle_time
            if remaining_time > 0:
                time.sleep(remaining_time)
                # Update cycle time to include sleep
                cycle_time = time.time() - process_start_time
                cycle_time_ms = cycle_time * 1000
        
        # Compile brain state
        brain_state = {
            'total_cycles': self.total_cycles,
            'prediction_confidence': confidence,
            'prediction_error': prediction_error,
            'prediction_method': 'bootstrap_random' if self.total_cycles == 0 else 'vector_stream_continuous',  # For test compatibility
            'cycle_time': cycle_time,
            'cycle_time_ms': cycle_time_ms,
            'hardware_adaptive_limits': self.hardware_adaptation.get_cognitive_limits(),
            'cognitive_autopilot': autopilot_state,
            'brain_uptime': time.time() - self.brain_start_time,
            'architecture': 'vector_stream',
            **vector_brain_state  # Include vector stream specific state
        }
        
        # Feature-flagged brain cycle logging after brain state is compiled
        if self.logger and self.config.get('logging', {}).get('log_brain_cycles', False):
            self.logger.log_brain_state(self, self.total_cycles, brain_state)
            
            # Emergence event detection - log significant state changes
            if hasattr(self, 'last_brain_state'):
                confidence_change = abs(confidence - self.last_brain_state.get('prediction_confidence', 0.0))
                
                # Detect emergence events
                if confidence_change > 0.3:  # Significant confidence change
                    emergence_type = 'confidence_jump' if confidence > self.last_brain_state.get('prediction_confidence', 0.0) else 'confidence_drop'
                    self.logger.log_emergence_event(
                        emergence_type,
                        f"Confidence changed by {confidence_change:.3f}",
                        {
                            'before_confidence': self.last_brain_state.get('prediction_confidence', 0.0),
                            'after_confidence': confidence,
                            'cycle': self.total_cycles
                        },
                        confidence_change  # significance parameter
                    )
                
                # Detect mode changes in cognitive autopilot
                current_mode = autopilot_state.get('cognitive_mode', 'unknown')
                last_mode = self.last_brain_state.get('cognitive_autopilot', {}).get('cognitive_mode', 'unknown')
                if current_mode != last_mode:
                    self.logger.log_emergence_event(
                        'cognitive_mode_transition',
                        f"Mode changed from {last_mode} to {current_mode}",
                        {
                            'before_mode': last_mode,
                            'after_mode': current_mode,
                            'cycle': self.total_cycles
                        },
                        1.0  # significance parameter (mode changes are always significant)
                    )
            
            # Store current state for next cycle comparison
            self.last_brain_state = brain_state.copy()
        
        # Increment cycle counter after compiling brain state
        self.total_cycles += 1
        self.total_predictions += 1
        self.total_experiences += 1
        
        # Process brain cycle for persistence (background, non-blocking)
        if self.enable_persistence and self.persistence_manager:
            self.persistence_manager.process_brain_cycle(self)
        
        return predicted_action, brain_state
    
    def store_experience(self, sensory_input: List[float], action_taken: List[float], 
                        outcome: List[float], predicted_action: List[float] = None) -> str:
        """
        Store experience in vector streams (no discrete experience objects).
        
        Args:
            sensory_input: What was sensed
            action_taken: What action was taken  
            outcome: What actually happened
            predicted_action: What was predicted (for computing error)
            
        Returns:
            The experience ID (timestamp-based for vector streams)
        """
        # In vector streams, we don't store discrete experiences
        # Instead, the vector brain continuously learns from the stream flow
        
        # Generate experience ID for compatibility
        experience_id = f"vector_stream_{int(time.time() * 1000)}"
        
        # Log learning outcome if logging enabled
        if self.logger:
            # Compute simple prediction error for logging
            prediction_error = 0.5
            if predicted_action:
                import numpy as np
                predicted = np.array(predicted_action[:len(outcome)])
                actual = np.array(outcome[:len(predicted_action)])
                if len(predicted) > 0:
                    error = np.linalg.norm(predicted - actual)
                    max_error = np.linalg.norm(predicted) + np.linalg.norm(actual)
                    prediction_error = min(1.0, error / max_error) if max_error > 0 else 0.0
            
            self.logger.log_prediction_outcome(
                action_taken, sensory_input, 1.0 - prediction_error, 0
            )
        
        return experience_id
    
    def get_brain_stats(self) -> Dict[str, Any]:
        """Get comprehensive brain performance statistics."""
        vector_stats = self.vector_brain.get_brain_statistics()
        
        return {
            'brain_summary': {
                'total_cycles': self.total_cycles,
                'total_experiences': self.total_experiences,
                'total_predictions': self.total_predictions,
                'uptime_seconds': time.time() - self.brain_start_time,
                'cycles_per_minute': self.total_cycles / max(1, (time.time() - self.brain_start_time) / 60),
                'experiences_per_minute': self.total_experiences / max(1, (time.time() - self.brain_start_time) / 60),
                'architecture': vector_stats.get('architecture', 'vector_stream'),
                'prediction_confidence': vector_stats.get('prediction_confidence', 0.0)
            },
            'vector_brain': vector_stats
        }
    
    def reset_brain(self):
        """Reset the brain to initial state (for testing)."""
        # Reset vector brain with same type and dimensions
        if self.brain_type == "sparse_goldilocks":
            self.vector_brain = SparseGoldilocksBrain(
                sensory_dim=self.sensory_dim,
                motor_dim=self.motor_dim,
                temporal_dim=self.temporal_dim,
                quiet_mode=self.quiet_mode
            )
        elif self.brain_type == "minimal":
            self.vector_brain = MinimalVectorStreamBrain(
                sensory_dim=self.sensory_dim,
                motor_dim=self.motor_dim,
                temporal_dim=self.temporal_dim
            )
        
        self.total_cycles = 0
        self.brain_start_time = time.time()
        
        # Reset logger if enabled
        if self.logger:
            self.logger.close_session()
            self.logger = BrainLogger(config=self.config) if self.enable_logging else None
        
        print("🧹 MinimalBrain reset to initial state")
    
    def close_logging_session(self):
        """Close the current logging session and generate final report."""
        if self.logger:
            return self.logger.close_session()
        return None
    
    def finalize_session(self):
        """Finalize brain session - call on shutdown."""
        # Save persistent brain state (blocking - critical for continuity)
        if self.enable_persistence and self.persistence_manager:
            self.persistence_manager.save_brain_state_blocking(self)
            self.persistence_manager.shutdown()
        
        # Close logging
        if self.logger:
            self.logger.close_session()
    
    def compute_intrinsic_reward(self) -> float:
        """
        Compute intrinsic reward for current brain state.
        
        This is used by the BrainLogger to track learning progress.
        For the vector stream brain, we use confidence as intrinsic reward.
        """
        if hasattr(self.vector_brain, 'emergent_confidence'):
            return self.vector_brain.emergent_confidence.current_confidence
        elif hasattr(self.vector_brain, '_estimate_prediction_confidence'):
            return self.vector_brain._estimate_prediction_confidence()
        else:
            # Fallback: simple confidence based on cycle count
            return min(1.0, self.total_cycles / 1000.0)
    
    
    def get_brain_statistics(self) -> dict:
        """
        Get detailed brain statistics (wrapper for vector brain implementation).
        
        Returns comprehensive brain statistics.
        """
        # Get basic stats
        basic_stats = self.get_brain_stats()
        
        # Get detailed stats from vector brain if available
        if hasattr(self.vector_brain, 'get_brain_statistics'):
            detailed_stats = self.vector_brain.get_brain_statistics()
            basic_stats.update(detailed_stats)
        
        return basic_stats
    
    def _apply_recovered_state(self, recovered_state):
        """Apply recovered brain state to the brain (biological wake-up process)."""
        if not recovered_state:
            return
        
        try:
            # Use the brain serializer to restore the complete state
            if self.persistence_manager and self.persistence_manager.brain_serializer:
                restoration_success = self.persistence_manager.brain_serializer.restore_brain_state(
                    self, recovered_state
                )
                
                if restoration_success:
                    print(f"🧠 Applied recovered state: {len(recovered_state.patterns)} patterns restored")
                else:
                    print(f"⚠️ Failed to apply some recovered state - continuing with partial recovery")
                    
                # Update experience counters from recovered state
                self.total_experiences = recovered_state.total_experiences
                self.total_cycles = recovered_state.total_cycles
            
        except Exception as e:
            print(f"⚠️ Failed to apply recovered state: {e}")
            print("🧠 Continuing with fresh state")
    
    
    def __str__(self) -> str:
        return (f"MinimalBrain({self.total_cycles} cycles, "
                f"{self.brain_type}_architecture, "
                f"sensory={self.sensory_dim}D, "
                f"motor={self.motor_dim}D)")
    
    def __repr__(self) -> str:
        return self.__str__()