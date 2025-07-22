"""
Brain Factory - Multi-Implementation Brain Coordinator

Factory that creates and manages different brain implementations:
1. Field Brain - unified field dynamics for spatial intelligence
2. Sparse Goldilocks - sparse pattern learning with biological timing
3. Minimal Brain - lightweight vector stream processing

Provides a unified interface while delegating to specialized brain implementations
based on configuration and requirements.
"""

import os
import time
from typing import List, Dict, Tuple, Optional, Any

# UNIFIED: Direct import of the only brain implementation
from .brains.field.core_brain import UnifiedFieldBrain
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
        
        # Configure GPU memory management
        from .config.gpu_memory_manager import configure_gpu_memory
        configure_gpu_memory(self.config)
        
        # Get brain implementation configuration from settings
        brain_config = self.config.get('brain', {})
        self.brain_type = brain_type or brain_config.get('type', 'field')
        self.sensory_dim = sensory_dim or brain_config.get('sensory_dim', 16)
        self.motor_dim = motor_dim or brain_config.get('motor_dim', 4)
        self.temporal_dim = temporal_dim or brain_config.get('temporal_dim', 4)
        max_patterns = max_patterns or brain_config.get('max_patterns', 100000)
        
        # Biological timing configuration (for sparse_goldilocks brain)
        self.target_cycle_time_ms = brain_config.get('target_cycle_time_ms', 25.0)  # Gamma frequency: 40Hz = 25ms
        self.target_cycle_time_s = self.target_cycle_time_ms / 1000.0
        
        # Field brain has its own timing system - no need for biological oscillator
        self.biological_oscillator = None
        self.parallel_coordinator = None
        
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
        
        # UNIFIED: Create the one true brain implementation - UnifiedFieldBrain
        # All brain_type configurations now map to the same advanced field brain
        if not quiet_mode:
            print(f"🧠 Using UnifiedFieldBrain (36D physics-organized field intelligence)")
        
        self.unified_brain = UnifiedFieldBrain(
            spatial_resolution=brain_config.get('field_spatial_resolution', 20),
            temporal_window=brain_config.get('field_temporal_window', 10.0),
            field_evolution_rate=brain_config.get('field_evolution_rate', 0.2),
            constraint_discovery_rate=brain_config.get('constraint_discovery_rate', 0.15),
            quiet_mode=quiet_mode
        )
        
        # Initialize cognitive autopilot for adaptive intensity control
        self.cognitive_autopilot = CognitiveAutopilot()
        
        # UNIFIED: No maintenance scheduler needed - UnifiedFieldBrain handles its own maintenance
        self.maintenance_scheduler = None
        
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
            # Read memory path from config, defaulting to server/robot_memory
            server_dir = os.path.dirname(os.path.dirname(__file__))  # This is the server/ directory
            config_memory_path = self.config.get('memory', {}).get('persistent_memory_path', './robot_memory')
            
            # If it's a relative path, make it relative to server directory
            if config_memory_path.startswith('./'):
                memory_path = os.path.join(server_dir, config_memory_path[2:])  # Remove './'
            else:
                memory_path = config_memory_path
            
            if not self.quiet_mode:
                print(f"🗂️  Memory path: {memory_path}")
            
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
        
        # UNIFIED: Process through UnifiedFieldBrain directly
        predicted_action, field_brain_state = self.unified_brain.process_robot_cycle(processed_input)
        
        # Adjust action dimensions if requested
        if action_dimensions and action_dimensions != len(predicted_action):
            if action_dimensions < len(predicted_action):
                # Truncate
                predicted_action = predicted_action[:action_dimensions]
            else:
                # Pad with zeros
                predicted_action = predicted_action + [0.0] * (action_dimensions - len(predicted_action))
        
        # Use cognitive autopilot state for maintenance scheduling
        confidence = field_brain_state.get('last_action_confidence', 0.0)
        
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
            
            # UNIFIED: Field topology discovery logging for scientific analysis
            if self.config.get('logging', {}).get('log_pattern_storage', False):
                # Log topology region discoveries (field-native equivalent of pattern storage)
                if self.unified_brain.topology_discoveries > 0:
                    for region_key, region_info in self.unified_brain.topology_regions.items():
                        self.logger.log_similarity_evolution(
                            region_key,
                            region_info.get('stability', 0.0),
                            'topology_discovery'
                        )
                        
                        # Log as adaptation event for significant topology discoveries
                        if region_info.get('activation', 0.0) > 0.1:
                            self.logger.log_adaptation_event(
                                'topology_discovery',
                                region_key,
                                {'field_energy': 0.0},
                                {'field_energy': region_info.get('activation', 0.0)}
                            )
        
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
            'architecture': 'field_dynamics_biological' if self.biological_oscillator else 'field_dynamics',
            **field_brain_state  # Include field brain adapter specific state
        }
        
        # Ensure parallel_processing flag is always present for compatibility
        if 'parallel_processing' not in brain_state:
            brain_state['parallel_processing'] = bool(
                self.parallel_coordinator and 
                self.parallel_coordinator.parallel_mode_enabled and
                field_brain_state.get('parallel_processing', False)
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
                
                # UNIFIED: Track field brain learning metrics directly
                current_evolution_cycles = self.unified_brain.field_evolution_cycles
                last_evolution_cycles = getattr(self, '_last_evolution_cycles', 0)
                
                # Report every 100 field evolution cycles (reduce spam on fast CUDA machines)
                if current_evolution_cycles > 0 and current_evolution_cycles % 100 == 0 and current_evolution_cycles != last_evolution_cycles:
                    try:
                        field_energy = self.unified_brain.unified_field.sum().item()
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
        
        # Store current brain state for monitoring server access
        self._current_brain_state = brain_state.copy()
        
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
        # UNIFIED: Get brain statistics directly from UnifiedFieldBrain
        unified_brain_stats = self.unified_brain._get_field_brain_state(0)
        
        return {
            'brain_summary': {
                'total_cycles': self.total_cycles,
                'total_experiences': self.total_experiences,
                'total_predictions': self.total_predictions,
                'uptime_seconds': time.time() - self.brain_start_time,
                'cycles_per_minute': self.total_cycles / max(1, (time.time() - self.brain_start_time) / 60),
                'experiences_per_minute': self.total_experiences / max(1, (time.time() - self.brain_start_time) / 60),
                'architecture': 'unified_field_36D',
                'prediction_confidence': unified_brain_stats.get('last_action_confidence', 0.0)
            },
            'unified_field_brain': unified_brain_stats
        }
    
    def reset_brain(self):
        """Reset the brain to initial state (for testing)."""
        # UNIFIED: Reset UnifiedFieldBrain to initial state
        self.unified_brain = UnifiedFieldBrain(
            spatial_resolution=20,
            temporal_window=10.0,
            field_evolution_rate=0.2,
            constraint_discovery_rate=0.15,
            quiet_mode=self.quiet_mode
        )
        
        self.total_cycles = 0
        self.brain_start_time = time.time()
        
        # Reset logger if enabled
        if self.logger:
            self.logger.close_session()
            self.logger = BrainLogger(config=self.config) if self.enable_logging else None
        
        print("🧹 Field brain reset to initial state")
    
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
        # UNIFIED: Get prediction confidence from UnifiedFieldBrain
        if self.unified_brain.field_actions:
            return self.unified_brain.field_actions[-1].action_confidence
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
        
        # UNIFIED: Get detailed stats from UnifiedFieldBrain
        field_memory_stats = self.unified_brain.get_field_memory_stats()
        basic_stats.update(field_memory_stats)
        
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
        """UNIFIED: UnifiedFieldBrain handles its own maintenance internally."""
        pass
    
    def heavy_maintenance(self) -> None:
        """UNIFIED: UnifiedFieldBrain handles its own maintenance internally."""
        pass
    
    def deep_consolidation(self) -> None:
        """UNIFIED: Field consolidation available via consolidate_field() method."""
        self.unified_brain.consolidate_field(consolidation_strength=0.1)
    
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
        """UNIFIED: UnifiedFieldBrain maintains itself internally."""
        return {
            'maintenance_available': True,
            'maintenance_type': 'field_native',
            'field_consolidation_available': True,
            'brain_cycles': self.unified_brain.brain_cycles,
            'field_evolution_cycles': self.unified_brain.field_evolution_cycles
        }
    
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