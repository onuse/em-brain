"""
Brain Factory - Multi-Implementation Brain Coordinator

Factory that creates and manages different brain implementations:
1. Field Brain - unified field dynamics for spatial intelligence
2. Sparse Goldilocks - sparse pattern learning with biological timing
3. Minimal Brain - lightweight vector stream processing

Provides a unified interface while delegating to specialized brain implementations
based on configuration and requirements.
"""

import time
from typing import List, Dict, Tuple, Optional, Any

from .brains.minimal.core_brain import MinimalVectorStreamBrain
from .brains.sparse_goldilocks.core_brain import SparseGoldilocksBrain
from .brains.field.tcp_adapter import FieldBrainTCPAdapter, FieldBrainConfig
from .brains.brain_maintenance_interface import BrainMaintenanceInterface, MaintenanceScheduler
from .utils.cognitive_autopilot import CognitiveAutopilot
from .utils.brain_logger import BrainLogger
from .utils.hardware_adaptation import get_hardware_adaptation, record_brain_cycle_performance
from .persistence import PersistenceManager, PersistenceConfig


class BrainFactory:
    """
    Factory for creating and managing different brain implementations.
    
    Supports multiple brain architectures:
    - field: Unified field dynamics with spatial intelligence
    - sparse_goldilocks: Sparse pattern learning with biological timing
    - minimal: Lightweight vector stream processing
    
    Provides unified interface while delegating to specialized implementations.
    """
    
    def __init__(self, config=None, enable_logging=True, log_session_name=None, quiet_mode=False, 
                 sensory_dim=None, motor_dim=None, temporal_dim=None, max_patterns=None, 
                 brain_type=None):
        """Initialize the brain factory with specified implementation type."""
        
        # Store config for use in other methods
        self.config = config or {}
        self.quiet_mode = quiet_mode
        
        # Get brain implementation configuration from settings
        brain_config = self.config.get('brain', {})
        self.brain_type = brain_type or brain_config.get('type', 'sparse_goldilocks')
        self.sensory_dim = sensory_dim or brain_config.get('sensory_dim', 16)
        self.motor_dim = motor_dim or brain_config.get('motor_dim', 4)
        self.temporal_dim = temporal_dim or brain_config.get('temporal_dim', 4)
        max_patterns = max_patterns or brain_config.get('max_patterns', 100000)
        
        # Biological timing configuration (for sparse_goldilocks brain)
        self.target_cycle_time_ms = brain_config.get('target_cycle_time_ms', 25.0)  # Gamma frequency: 40Hz = 25ms
        self.target_cycle_time_s = self.target_cycle_time_ms / 1000.0
        
        # Initialize biological oscillator for sparse_goldilocks brain
        self.biological_oscillator = None
        if brain_config.get('enable_biological_timing', True):
            from .brains.sparse_goldilocks.systems.biological_oscillator import create_biological_oscillator
            oscillator_config = {
                'gamma_freq': 1000.0 / self.target_cycle_time_ms,  # Convert ms to Hz
                'theta_freq': brain_config.get('theta_freq', 6.0)
            }
            self.biological_oscillator = create_biological_oscillator(oscillator_config, quiet_mode)
        
        # Initialize parallel coordinator for sparse_goldilocks brain
        self.parallel_coordinator = None
        if brain_config.get('enable_parallel_processing', False) and self.biological_oscillator:
            from .brains.sparse_goldilocks.systems.parallel_brain_coordinator import ParallelBrainCoordinator
            self.parallel_coordinator = ParallelBrainCoordinator(
                None,  # Will be set after vector brain initialization
                self.biological_oscillator,
                quiet_mode
            )
        
        # Log brain factory configuration for traceability
        if not quiet_mode:
            print(f"🏭 Brain Factory Configuration:")
            sensory_source = "param" if sensory_dim else ("config" if 'sensory_dim' in brain_config else "default")
            motor_source = "param" if motor_dim else ("config" if 'motor_dim' in brain_config else "default") 
            temporal_source = "param" if temporal_dim else ("config" if 'temporal_dim' in brain_config else "default")
            timing_source = "config" if 'target_cycle_time_ms' in brain_config else "default"
            print(f"   Sensory: {self.sensory_dim}D (from {sensory_source})")
            print(f"   Motor: {self.motor_dim}D (from {motor_source})")
            print(f"   Temporal: {self.temporal_dim}D (from {temporal_source})")
            timing_mode = "biological_gamma" if self.biological_oscillator else "traditional"
            print(f"   Cycle time: {self.target_cycle_time_ms}ms (from {timing_source}, mode: {timing_mode})")
            print(f"   Type: {self.brain_type} (from {'param' if brain_type else ('config' if 'type' in brain_config else 'default')})")
            parallel_status = "enabled" if self.parallel_coordinator else "disabled"
            print(f"   Parallel processing: {parallel_status}")
        
        # Hardware adaptation system (field brains have their own optimizations)
        if self.brain_type == "field":
            self.hardware_adaptation = None
        else:
            self.hardware_adaptation = get_hardware_adaptation(self.config)
        
        # Create brain implementation based on configured type
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
        elif self.brain_type == "field":
            # Create field brain configuration
            field_config = FieldBrainConfig(
                sensory_dimensions=self.sensory_dim,
                motor_dimensions=self.motor_dim,
                spatial_resolution=brain_config.get('field_spatial_resolution', 20),
                temporal_window=brain_config.get('field_temporal_window', 10.0),
                field_evolution_rate=brain_config.get('field_evolution_rate', 0.1),
                constraint_discovery_rate=brain_config.get('constraint_discovery_rate', 0.15),
                performance_mode=brain_config.get('performance_mode', 'balanced'),
                enable_enhanced_dynamics=brain_config.get('enable_enhanced_dynamics', True),
                enable_attention_guidance=brain_config.get('enable_attention_guidance', True),
                enable_hierarchical_processing=brain_config.get('enable_hierarchical_processing', True),
                enable_attention_super_resolution=brain_config.get('enable_attention_super_resolution', False),
                attention_base_resolution=brain_config.get('attention_base_resolution', 50),
                attention_focus_resolution=brain_config.get('attention_focus_resolution', 100),
                hierarchical_max_time_ms=brain_config.get('hierarchical_max_time_ms', 40.0),
                target_cycle_time_ms=brain_config.get('target_cycle_time_ms', 150.0),
                quiet_mode=quiet_mode
            )
            
            self.vector_brain = FieldBrainTCPAdapter(field_config)
        else:
            raise ValueError(f"Unknown brain_type: {self.brain_type}. Options: 'sparse_goldilocks', 'minimal', 'field'")
        
        # Connect parallel coordinator to brain implementation after initialization
        if self.parallel_coordinator:
            self.parallel_coordinator.set_vector_brain(self.vector_brain)
        
        # Initialize cognitive autopilot for adaptive intensity control
        self.cognitive_autopilot = CognitiveAutopilot()
        
        # Initialize maintenance scheduler if brain supports maintenance interface
        self.maintenance_scheduler = None
        if isinstance(self.vector_brain, BrainMaintenanceInterface):
            self.maintenance_scheduler = MaintenanceScheduler(self.vector_brain)
            if not quiet_mode:
                print(f"🔧 Maintenance interface enabled for {self.brain_type} brain")
        
        # Factory state tracking
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
            self.logger = BrainLogger(session_name=log_session_name, config=self.config)
        else:
            self.logger = None
        
        # Persistence system for cross-session learning
        self.enable_persistence = self.config.get('memory', {}).get('enable_persistence', True)
        self.persistence_manager = None
        self.session_count = 0
        
        if self.enable_persistence:
            memory_path = self.config.get('memory', {}).get('persistent_memory_path', './robot_memory')
            
            # Create persistence configuration
            persistence_config = PersistenceConfig(
                memory_root_path=memory_path,
                incremental_save_interval_cycles=self.config.get('memory', {}).get('save_interval_cycles', 100),
                enable_compression=self.config.get('memory', {}).get('enable_compression', True),
                enable_corruption_detection=self.config.get('memory', {}).get('enable_corruption_detection', True)
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
            print(f"🏭 BrainFactory initialized - {self.brain_type.title()} Implementation")
            print(f"   Sensory stream: {self.sensory_dim}D")
            print(f"   Motor stream: {self.motor_dim}D") 
            print(f"   Temporal stream: {self.temporal_dim}D")
            if self.brain_type == "sparse_goldilocks":
                print(f"   Max patterns: {max_patterns:,}")
                print(f"   🧬 Sparse Distributed Representations enabled")
            print(f"   Continuous prediction and dead reckoning enabled")
    
    def process_sensory_input(self, sensory_input: List[float], 
                            action_dimensions: int = 4) -> Tuple[List[float], Dict[str, Any]]:
        """
        Process sensory input through the configured brain implementation.
        
        Delegates to the configured brain implementation:
        1. Field brain: unified field dynamics processing
        2. Sparse goldilocks: sparse pattern learning  
        3. Minimal brain: lightweight vector processing
        
        Args:
            sensory_input: Current sensory observation vector
            action_dimensions: Number of action dimensions to output (ignored - motor stream determines size)
            
        Returns:
            Tuple of (predicted_action, brain_state_info)
        """
        process_start_time = time.time()
        
        # Calculate initial cognitive load estimate (will refine after processing)
        input_novelty = min(1.0, len(sensory_input) / max(1, self.sensory_dim)) * 0.5  # Basic novelty estimate
        
        # Mark brain as active for maintenance scheduling with initial estimate
        if self.maintenance_scheduler:
            self.maintenance_scheduler.mark_activity(input_novelty)
        
        # Ensure sensory input matches expected dimensions
        if len(sensory_input) > self.sensory_dim:
            # Truncate if too long
            processed_input = sensory_input[:self.sensory_dim]
        elif len(sensory_input) < self.sensory_dim:
            # Pad if too short
            processed_input = sensory_input + [0.0] * (self.sensory_dim - len(sensory_input))
        else:
            processed_input = sensory_input
        
        # Process through vector stream brain (with optional parallel coordination)
        if self.parallel_coordinator and self.parallel_coordinator.parallel_mode_enabled:
            # Use parallel processing with biological coordination
            predicted_action, vector_brain_state = self.parallel_coordinator.process_with_parallel_coordination(processed_input)
        else:
            # Use traditional sequential processing
            predicted_action, vector_brain_state = self.vector_brain.process_sensory_input(processed_input)
        
        # Adjust action dimensions if requested
        if action_dimensions and action_dimensions != len(predicted_action):
            if action_dimensions < len(predicted_action):
                # Truncate
                predicted_action = predicted_action[:action_dimensions]
            else:
                # Pad with zeros
                predicted_action = predicted_action + [0.0] * (action_dimensions - len(predicted_action))
        
        # Use cognitive autopilot state for maintenance scheduling
        confidence = vector_brain_state['prediction_confidence']
        
        # Update cognitive autopilot with vector stream confidence
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
        
        # Use cognitive autopilot mode for maintenance activity marking
        if self.maintenance_scheduler:
            cognitive_mode = autopilot_state.get('cognitive_mode', 'deep_think')
            
            # Convert autopilot modes to cognitive load for maintenance scheduling:
            # - autopilot: low cognitive load (0.1) - brain coasting, maintenance OK
            # - focused: medium cognitive load (0.5) - moderate thinking, some maintenance OK  
            # - deep_think: high cognitive load (0.9) - intensive thinking, avoid maintenance
            cognitive_load_map = {
                'autopilot': 0.1,   # Coasting - maintenance encouraged
                'focused': 0.5,     # Moderate - light maintenance OK
                'deep_think': 0.9   # Working hard - avoid maintenance
            }
            
            cognitive_load = cognitive_load_map.get(cognitive_mode, 0.9)
            self.maintenance_scheduler.mark_activity(cognitive_load)
        
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
        
        # Record performance for hardware adaptation (skip for field brains)
        if self.hardware_adaptation is not None:
            memory_usage_mb = 50.0  # Vector streams use much less memory than experience storage
            record_brain_cycle_performance(cycle_time_ms, memory_usage_mb)
        
        # Enforce biological timing constraints (Phase 7: gamma-frequency cycles)
        if self.biological_oscillator:
            # Use biological oscillator for natural gamma-frequency timing
            timing = self.biological_oscillator.get_current_timing()
            remaining_budget = self.biological_oscillator.estimate_processing_budget()
            
            # If we have remaining time in the gamma cycle, sleep to maintain biological rhythm
            if remaining_budget > 0.001:  # At least 1ms remaining
                time.sleep(remaining_budget)
                # Update cycle time to include biological sleep
                cycle_time = time.time() - process_start_time
                cycle_time_ms = cycle_time * 1000
        elif self.target_cycle_time_s > 0:
            # Fallback to traditional timing if biological oscillator disabled
            remaining_time = self.target_cycle_time_s - cycle_time
            if remaining_time > 0:
                time.sleep(remaining_time)
                # Update cycle time to include sleep
                cycle_time = time.time() - process_start_time
                cycle_time_ms = cycle_time * 1000
        
        # Compile brain state with biological timing information
        brain_state = {
            'total_cycles': self.total_cycles,
            'prediction_confidence': confidence,
            'prediction_error': prediction_error,
            'prediction_method': 'bootstrap_random' if self.total_cycles == 0 else 'vector_stream_continuous',  # For test compatibility
            'cycle_time': cycle_time,
            'cycle_time_ms': cycle_time_ms,
            'hardware_adaptive_limits': self.hardware_adaptation.get_cognitive_limits() if self.hardware_adaptation else {},
            'cognitive_autopilot': autopilot_state,
            'brain_uptime': time.time() - self.brain_start_time,
            'architecture': 'vector_stream_biological' if self.biological_oscillator else 'vector_stream',
            **vector_brain_state  # Include vector stream specific state
        }
        
        # Ensure parallel_processing flag is always present for compatibility
        if 'parallel_processing' not in brain_state:
            brain_state['parallel_processing'] = bool(
                self.parallel_coordinator and 
                self.parallel_coordinator.parallel_mode_enabled and
                vector_brain_state.get('parallel_processing', False)
            )
        
        # Add biological timing information if available
        if self.biological_oscillator:
            coordination_signals = self.biological_oscillator.get_coordination_signal()
            oscillator_stats = self.biological_oscillator.get_oscillator_stats()
            brain_state.update({
                'biological_timing': {
                    'gamma_phase': coordination_signals['gamma_phase'],
                    'theta_phase': coordination_signals['theta_phase'],
                    'current_phase': oscillator_stats['current_phase'],
                    'binding_window_active': coordination_signals['binding_window'],
                    'consolidation_active': coordination_signals['consolidation_window'],
                    'gamma_cycles_completed': coordination_signals['cycle_count'],
                    'theta_cycles_completed': coordination_signals['theta_cycle_count'],
                    'avg_gamma_frequency': oscillator_stats['avg_gamma_frequency']
                }
            })
        
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
                
                # Track meaningful field brain learning metrics (for field brains only)
                if self.brain_type == "field" and hasattr(self.vector_brain, 'field_brain'):
                    field_brain = self.vector_brain.field_brain
                    
                    # Log significant field evolution events
                    if hasattr(field_brain, 'field_evolution_cycles'):
                        current_evolution_cycles = field_brain.field_evolution_cycles
                        last_evolution_cycles = getattr(self, '_last_evolution_cycles', 0)
                        
                        # Report every 10 field evolution cycles (meaningful learning activity)
                        if current_evolution_cycles > 0 and current_evolution_cycles % 10 == 0 and current_evolution_cycles != last_evolution_cycles:
                            try:
                                field_energy = field_brain.unified_field.sum().item()
                            except:
                                field_energy = 0.0
                                
                            print(f"🧠 Field Learning: {current_evolution_cycles} evolution cycles, field energy: {field_energy:.1f}")
                            self._last_evolution_cycles = current_evolution_cycles
            
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
        # Cleanup parallel coordinator
        if self.parallel_coordinator:
            self.parallel_coordinator.cleanup()
        
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
        
        # Add parallel processing stats if available
        if self.parallel_coordinator:
            parallel_stats = self.parallel_coordinator.get_performance_comparison()
            basic_stats['parallel_processing'] = parallel_stats
        
        return basic_stats
    
    def enable_parallel_processing(self, enabled: bool = True):
        """Enable or disable parallel processing mode."""
        if self.parallel_coordinator:
            self.parallel_coordinator.set_parallel_mode(enabled)
            return True
        else:
            if not self.quiet_mode:
                print("⚠️ Parallel processing not available (enable_parallel_processing=True in config required)")
            return False
    
    def get_parallel_performance_stats(self) -> Dict[str, Any]:
        """Get detailed parallel processing performance statistics."""
        if self.parallel_coordinator:
            return self.parallel_coordinator.get_performance_comparison()
        else:
            return {'parallel_processing_available': False}
    
    def light_maintenance(self) -> None:
        """Trigger light maintenance operations on the brain."""
        if isinstance(self.vector_brain, BrainMaintenanceInterface):
            self.vector_brain.safe_light_maintenance()
    
    def heavy_maintenance(self) -> None:
        """Trigger heavy maintenance operations on the brain."""
        if isinstance(self.vector_brain, BrainMaintenanceInterface):
            self.vector_brain.safe_heavy_maintenance()
    
    def deep_consolidation(self) -> None:
        """Trigger deep consolidation operations on the brain."""
        if isinstance(self.vector_brain, BrainMaintenanceInterface):
            self.vector_brain.safe_deep_consolidation()
    
    def run_recommended_maintenance(self) -> Dict[str, bool]:
        """Run maintenance operations recommended for current idle time."""
        if self.maintenance_scheduler:
            return self.maintenance_scheduler.run_recommended_maintenance()
        else:
            return {
                'light_maintenance': False,
                'heavy_maintenance': False,
                'deep_consolidation': False
            }
    
    def get_maintenance_status(self) -> Dict[str, Any]:
        """Get maintenance metrics and timing information."""
        if isinstance(self.vector_brain, BrainMaintenanceInterface):
            return self.vector_brain.get_maintenance_status()
        else:
            return {'maintenance_available': False}
    
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
        return (f"BrainFactory({self.total_cycles} cycles, "
                f"{self.brain_type}_implementation, "
                f"sensory={self.sensory_dim}D, "
                f"motor={self.motor_dim}D)")
    
    def __repr__(self) -> str:
        return self.__str__()


# Backwards compatibility alias
MinimalBrain = BrainFactory