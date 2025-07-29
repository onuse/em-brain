"""
Optimized Brain Cycle

Performance-optimized version of the brain processing cycle.
Key optimizations:
1. Extract patterns once and reuse
2. Disable predictive actions when not needed
3. Use torch.no_grad() for inference
4. Pre-allocate tensors where possible
"""

import torch
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass


@dataclass
class CycleCache:
    """Cache for reusing computations within a cycle."""
    patterns: List = None
    pattern_features: Dict = None
    attention_state: Dict = None


class OptimizedBrainMixin:
    """
    Mixin for performance-optimized brain operations.
    
    This can be mixed into SimplifiedUnifiedBrain to override
    performance-critical methods.
    """
    
    def __init__(self):
        self._cycle_cache = CycleCache()
        self._use_predictive_actions = False  # Disable by default for speed
        self._pattern_extraction_limit = 5  # Reduce from 10
        
    @torch.no_grad()  # Disable gradient computation
    def process_robot_cycle_optimized(self, sensory_input: List[float]) -> Tuple[List[float], Dict[str, Any]]:
        """Optimized version of process_robot_cycle."""
        import time
        cycle_start = time.perf_counter()
        
        # Clear cycle cache
        self._cycle_cache = CycleCache()
        
        try:
            # 1. Create field experience (minimal overhead)
            experience = self._create_field_experience(sensory_input)
            
            # 2. Skip prediction tracking if not needed
            if self._predicted_field is not None:
                # Use faster operations
                with torch.no_grad():
                    prediction_error = torch.mean(torch.abs(
                        self.unified_field - self._predicted_field
                    )).item()
                self._last_prediction_error = prediction_error
                self._current_prediction_confidence = 1.0 - min(1.0, prediction_error * 2.0)
            
            # 3. Imprint experience
            self._imprint_experience(experience)
            
            # 4. Extract patterns ONCE for entire cycle
            self._extract_cycle_patterns()
            
            # 5. Process attention using cached patterns
            self._process_attention_optimized(sensory_input)
            
            # 6. Update field dynamics
            reward = sensory_input[-1] if len(sensory_input) > 24 else 0.0
            field_state = self.field_dynamics.compute_field_state(self.unified_field)
            self.field_dynamics.update_confidence(self._last_prediction_error)
            
            has_input = len(sensory_input) > 0 and any(abs(v) > 0.01 for v in sensory_input[:-1])
            self.modulation = self.field_dynamics.compute_field_modulation(
                field_state, has_sensory_input=has_input
            )
            
            # 7. Process reward topology (only if significant)
            if abs(reward) > 0.1:
                self.topology_shaper.process_reward(
                    current_field=self.unified_field,
                    reward=reward,
                    threshold=0.1
                )
            
            # 8. Evolve field
            self._evolve_field()
            
            # 9. Generate motor action (optimized)
            motor_output = self._generate_motor_action_optimized()
            
            # Update state
            self.brain_cycles += 1
            self._last_cycle_time = time.perf_counter() - cycle_start
            
            # Create minimal brain state
            brain_state = self._create_brain_state_minimal()
            return motor_output, brain_state
            
        except Exception as e:
            # Minimal error handling
            self.brain_cycles += 1
            return [0.0] * (self.motor_cortex.motor_dim - 1), {'error': str(e)}
    
    def _extract_cycle_patterns(self):
        """Extract patterns once for the entire cycle."""
        # Use pattern cache pool if available
        if hasattr(self, 'pattern_cache_pool'):
            patterns = self.pattern_cache_pool.extract_patterns_fast(
                self.unified_field,
                n_patterns=self._pattern_extraction_limit
            )
            # Convert dict patterns to list format expected by cache
            self._cycle_cache.patterns = patterns
            self._cycle_cache.pattern_features = patterns[0] if patterns else {}
        else:
            # Fallback to regular extraction
            patterns = self.pattern_system.extract_patterns(
                self.unified_field, 
                n_patterns=self._pattern_extraction_limit
            )
            self._cycle_cache.patterns = patterns
            
            # Convert to features dict
            if patterns:
                self._cycle_cache.pattern_features = patterns[0].to_dict()
            else:
                self._cycle_cache.pattern_features = {}
    
    def _process_attention_optimized(self, sensory_input: List[float]):
        """Process attention using cached patterns."""
        # Create minimal sensory patterns
        sensory_patterns = {
            'primary': torch.tensor(sensory_input[:-1], dtype=torch.float32, device=self.device)
        }
        
        # Use cached patterns instead of re-extracting
        if hasattr(self.pattern_attention, 'process_with_cached_patterns'):
            attention_state = self.pattern_attention.process_with_cached_patterns(
                cached_patterns=self._cycle_cache.patterns,
                sensory_patterns=sensory_patterns
            )
        else:
            # Fallback to normal processing
            attention_state = self.pattern_attention.process_field_patterns(
                field=self.unified_field,
                sensory_patterns=sensory_patterns
            )
        
        self._cycle_cache.attention_state = attention_state
        self._last_attention_state = attention_state
    
    @torch.no_grad()
    def _generate_motor_action_optimized(self) -> List[float]:
        """Generate motor action using cached computations."""
        exploration_params = {
            'exploration_drive': self.modulation.get('exploration_drive', 0.5),
            'motor_noise': 0.2
        }
        
        # Generate motor commands directly
        motor_commands = self.pattern_motor.generate_motor_action(
            field=self.unified_field,
            spontaneous_activity=None,
            attention_state=self._cycle_cache.attention_state,
            exploration_params=exploration_params
        )
        
        # Skip predictive actions if disabled
        if not self._use_predictive_actions:
            # Direct conversion
            final_commands = torch.clamp(motor_commands, -1.0, 1.0)
            return final_commands.tolist()
        
        # Predictive actions (only if enabled)
        candidates = self.predictive_actions.generate_action_candidates(
            current_field=self.unified_field,
            current_patterns=self._cycle_cache.pattern_features,
            n_candidates=2  # Even fewer candidates
        )
        
        # Skip preview for speed
        selected_action = candidates[0] if candidates else None
        
        if selected_action and len(motor_commands) == len(selected_action.motor_pattern):
            final_commands = motor_commands * 0.7 + selected_action.motor_pattern * 0.3
        else:
            final_commands = motor_commands
        
        final_commands = torch.clamp(final_commands, -1.0, 1.0)
        return final_commands.tolist()
    
    def _create_brain_state_minimal(self) -> Dict[str, Any]:
        """Create minimal brain state for performance."""
        # Only compute what's needed
        return {
            'cycle': self.brain_cycles,
            'cycle_time_ms': self._last_cycle_time * 1000,
            'field_energy': float(torch.mean(torch.abs(self.unified_field))),
            'exploration_drive': self.modulation.get('exploration_drive', 0.5)
        }
    
    def enable_predictive_actions(self, enabled: bool = True):
        """Enable or disable predictive actions for performance."""
        self._use_predictive_actions = enabled
    
    def set_pattern_extraction_limit(self, limit: int):
        """Set the maximum number of patterns to extract per cycle."""
        self._pattern_extraction_limit = max(1, min(limit, 20))