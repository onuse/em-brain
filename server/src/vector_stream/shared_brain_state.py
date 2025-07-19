#!/usr/bin/env python3
"""
Shared Brain State Infrastructure

Provides atomic, lock-free shared state for parallel stream processing.
Enables coordination between concurrent streams without explicit synchronization.

Key principles:
- Lock-free reads and writes for high-performance parallel access
- Atomic updates for consistency without blocking
- Constraint-based coordination through shared resource competition
- Biological realism through natural information flow
"""

import time
import threading
from typing import Dict, Any, Optional, List, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import numpy as np

from .stream_types import StreamType, ConstraintType, StreamState


@dataclass
class CrossStreamBinding:
    """Information binding between streams during gamma windows."""
    source_stream: StreamType
    target_stream: StreamType
    binding_strength: float
    pattern_associations: List[Tuple[int, int]]  # (source_pattern, target_pattern)
    binding_time: float
    gamma_cycle: int


class SharedBrainState:
    """
    Atomic shared brain state for parallel stream coordination.
    
    Provides lock-free access to shared brain information while maintaining
    consistency through atomic operations and constraint-based coordination.
    """
    
    def __init__(self, biological_oscillator=None, quiet_mode: bool = False):
        """
        Initialize shared brain state.
        
        Args:
            biological_oscillator: Biological oscillator for timing coordination
            quiet_mode: Suppress debug output
        """
        self.biological_oscillator = biological_oscillator
        self.quiet_mode = quiet_mode
        
        # Stream states (atomic updates)
        self._stream_states = {}
        self._stream_lock = threading.RLock()  # Minimal locking for state updates
        
        # Cross-stream bindings (gamma-frequency binding windows)
        self._bindings = []
        self._bindings_lock = threading.RLock()
        
        # Shared resources for constraint-based coordination
        self._shared_resources = {
            'attention_budget': 1.0,      # Total attention available
            'processing_energy': 1.0,     # Total processing energy
            'working_memory_slots': 8,    # Active pattern slots
            'binding_capacity': 10        # Cross-stream binding slots
        }
        self._resource_lock = threading.RLock()
        
        # Coordination signals (lock-free reads)
        self._coordination_signals = {}
        
        # Phase 7c: Enhanced constraint propagation system
        from .constraint_propagation_system import create_constraint_propagation_system
        self.constraint_propagation = create_constraint_propagation_system(
            biological_oscillator, quiet_mode
        )
        
        # Phase 7c: Emergent attention allocation system
        from .emergent_attention_allocation import create_attention_allocator, AttentionCompetitionMode
        self.attention_allocator = create_attention_allocator(
            budget=1.0,
            mode=AttentionCompetitionMode.BIOLOGICAL_INHIBITION,
            quiet_mode=quiet_mode
        )
        
        # Phase 7c: Constraint-based pattern inhibition system
        from .constraint_pattern_inhibition import create_pattern_inhibitor, PatternSelectionMode
        self.pattern_inhibitor = create_pattern_inhibitor(
            max_patterns=20,  # Allow more patterns across all streams
            mode=PatternSelectionMode.COMPETITIVE_INHIBITION,
            quiet_mode=quiet_mode
        )
        
        # Statistics for monitoring
        self._stats = {
            'total_updates': 0,
            'binding_events': 0,
            'resource_competitions': 0,
            'coordination_events': 0,
            'constraint_propagations': 0
        }
        
        # Initialize stream states
        for stream_type in StreamType:
            self._stream_states[stream_type] = StreamState(
                stream_type=stream_type,
                last_update_time=time.time(),
                processing_phase="inactive"
            )
        
        if not quiet_mode:
            print(f"🧠 SharedBrainState initialized")
            print(f"   Streams: {[s.value for s in StreamType]}")
            print(f"   Resources: {self._shared_resources}")
    
    def update_stream_state(self, stream_type: StreamType, **updates) -> bool:
        """
        Atomically update stream state.
        
        Args:
            stream_type: Type of stream to update
            **updates: State updates to apply
            
        Returns:
            True if update successful, False if resource constraints prevent it
        """
        with self._stream_lock:
            if stream_type not in self._stream_states:
                return False
            
            stream_state = self._stream_states[stream_type]
            
            # Apply updates
            for key, value in updates.items():
                if hasattr(stream_state, key):
                    setattr(stream_state, key, value)
            
            # Update timestamp
            stream_state.last_update_time = time.time()
            
            # Update statistics
            self._stats['total_updates'] += 1
            
            return True
    
    def get_stream_state(self, stream_type: StreamType) -> Optional[StreamState]:
        """
        Get current state of a stream (lock-free read).
        
        Args:
            stream_type: Type of stream to query
            
        Returns:
            Current stream state or None if not found
        """
        return self._stream_states.get(stream_type)
    
    def get_all_stream_states(self) -> Dict[StreamType, StreamState]:
        """Get all current stream states (lock-free read)."""
        return self._stream_states.copy()
    
    def request_resource(self, stream_type: StreamType, resource_type: str, amount: float) -> float:
        """
        Request shared resource allocation (constraint-based coordination).
        
        Args:
            stream_type: Stream requesting the resource
            resource_type: Type of resource ('attention_budget', 'processing_energy', etc.)
            amount: Amount of resource requested
            
        Returns:
            Amount of resource actually allocated (may be less than requested)
        """
        with self._resource_lock:
            if resource_type not in self._shared_resources:
                return 0.0
            
            available = self._shared_resources[resource_type]
            allocated = min(amount, available)
            
            if allocated > 0:
                self._shared_resources[resource_type] -= allocated
                self._stats['resource_competitions'] += 1
                
                # Log significant resource competition
                if allocated < amount * 0.5:  # Got less than half requested
                    if not self.quiet_mode:
                        competition_level = allocated / amount
                        print(f"⚡ Resource competition: {stream_type.value} got {competition_level:.1%} of requested {resource_type}")
            
            return allocated
    
    def release_resource(self, resource_type: str, amount: float):
        """
        Release shared resource back to the pool.
        
        Args:
            resource_type: Type of resource to release
            amount: Amount to release
        """
        with self._resource_lock:
            if resource_type in self._shared_resources:
                # Cap at 1.0 for normalized resources
                if resource_type in ['attention_budget', 'processing_energy']:
                    self._shared_resources[resource_type] = min(1.0, self._shared_resources[resource_type] + amount)
                else:
                    self._shared_resources[resource_type] += amount
    
    def create_cross_stream_binding(self, source_stream: StreamType, target_stream: StreamType,
                                  source_patterns: List[int], target_patterns: List[int],
                                  binding_strength: float = 1.0) -> bool:
        """
        Create cross-stream binding during gamma window.
        
        Args:
            source_stream: Source stream type
            target_stream: Target stream type
            source_patterns: Active patterns in source stream
            target_patterns: Active patterns in target stream
            binding_strength: Strength of the binding
            
        Returns:
            True if binding created successfully
        """
        # Check if we're in a binding window
        if self.biological_oscillator:
            coordination = self.biological_oscillator.get_coordination_signal()
            if not coordination['binding_window']:
                return False  # Not in gamma binding window
        
        with self._bindings_lock:
            # Check binding capacity constraint
            if len(self._bindings) >= self._shared_resources.get('binding_capacity', 10):
                # Remove oldest binding to make room (FIFO constraint)
                self._bindings.pop(0)
            
            # Create pattern associations
            pattern_associations = []
            for src_pattern in source_patterns[:5]:  # Limit to top 5 patterns
                for tgt_pattern in target_patterns[:5]:
                    pattern_associations.append((src_pattern, tgt_pattern))
            
            # Create binding
            binding = CrossStreamBinding(
                source_stream=source_stream,
                target_stream=target_stream,
                binding_strength=binding_strength,
                pattern_associations=pattern_associations,
                binding_time=time.time(),
                gamma_cycle=self.biological_oscillator.cycle_count if self.biological_oscillator else 0
            )
            
            self._bindings.append(binding)
            self._stats['binding_events'] += 1
            
            return True
    
    def get_cross_stream_bindings(self, stream_type: StreamType = None, 
                                max_age_seconds: float = 1.0) -> List[CrossStreamBinding]:
        """
        Get recent cross-stream bindings for coordination.
        
        Args:
            stream_type: Filter by stream type (None for all)
            max_age_seconds: Maximum age of bindings to return
            
        Returns:
            List of recent cross-stream bindings
        """
        current_time = time.time()
        cutoff_time = current_time - max_age_seconds
        
        with self._bindings_lock:
            recent_bindings = [
                binding for binding in self._bindings
                if binding.binding_time >= cutoff_time
            ]
            
            if stream_type:
                recent_bindings = [
                    binding for binding in recent_bindings
                    if binding.source_stream == stream_type or binding.target_stream == stream_type
                ]
            
            return recent_bindings
    
    def set_coordination_signal(self, signal_name: str, signal_value: Any):
        """
        Set coordination signal for stream synchronization (lock-free write).
        
        Args:
            signal_name: Name of the coordination signal
            signal_value: Value of the signal
        """
        self._coordination_signals[signal_name] = {
            'value': signal_value,
            'timestamp': time.time()
        }
        self._stats['coordination_events'] += 1
    
    def get_coordination_signal(self, signal_name: str, max_age_seconds: float = 0.1) -> Any:
        """
        Get coordination signal value (lock-free read).
        
        Args:
            signal_name: Name of the coordination signal
            max_age_seconds: Maximum age of signal to accept
            
        Returns:
            Signal value or None if not found or too old
        """
        signal = self._coordination_signals.get(signal_name)
        if not signal:
            return None
        
        current_time = time.time()
        if current_time - signal['timestamp'] > max_age_seconds:
            return None  # Signal too old
        
        return signal['value']
    
    def get_coordination_signals(self) -> Dict[str, Any]:
        """Get all current coordination signals (lock-free read)."""
        return {
            name: signal['value'] 
            for name, signal in self._coordination_signals.items()
        }
    
    def cleanup_old_data(self, max_age_seconds: float = 10.0):
        """
        Clean up old bindings and signals to prevent memory growth.
        
        Args:
            max_age_seconds: Maximum age of data to keep
        """
        current_time = time.time()
        cutoff_time = current_time - max_age_seconds
        
        # Clean old bindings
        with self._bindings_lock:
            self._bindings = [
                binding for binding in self._bindings
                if binding.binding_time >= cutoff_time
            ]
        
        # Clean old coordination signals
        self._coordination_signals = {
            name: signal for name, signal in self._coordination_signals.items()
            if signal['timestamp'] >= cutoff_time
        }
    
    def get_shared_state_stats(self) -> Dict[str, Any]:
        """Get shared state statistics for monitoring."""
        current_time = time.time()
        
        # Calculate stream activity
        stream_activity = {}
        for stream_type, state in self._stream_states.items():
            time_since_update = current_time - state.last_update_time
            stream_activity[stream_type.value] = {
                'last_update_seconds_ago': time_since_update,
                'active': time_since_update < 1.0,
                'processing_phase': state.processing_phase,
                'active_patterns': len(state.active_patterns),
                'activation_strength': state.activation_strength
            }
        
        # Phase 7c: Calculate constraint pressures for each stream
        constraint_pressures = {}
        for stream_type in StreamType:
            stream_constraints = self.get_constraint_pressures(stream_type)
            total_pressure = self.get_total_constraint_pressure(stream_type)
            constraint_pressures[stream_type.value] = {
                'total_pressure': total_pressure,
                'constraint_breakdown': {ct.value: intensity for ct, intensity in stream_constraints.items()}
            }
        
        return {
            'stream_activity': stream_activity,
            'active_bindings': len(self._bindings),
            'coordination_signals': len(self._coordination_signals),
            'shared_resources': self._shared_resources.copy(),
            'statistics': self._stats.copy(),
            'biological_timing': (
                self.biological_oscillator.get_coordination_signal() 
                if self.biological_oscillator else None
            ),
            'constraint_propagation': {
                'constraint_pressures': constraint_pressures,
                'propagation_stats': self.constraint_propagation.get_propagation_stats()
            },
            'attention_allocation': {
                'current_allocations': self.attention_allocator.get_current_allocations(),
                'competition_stats': self.attention_allocator.get_competition_stats()
            },
            'pattern_inhibition': {
                'active_patterns_info': self.pattern_inhibitor.get_active_patterns_info(),
                'selection_stats': self.pattern_inhibitor.get_selection_stats()
            }
        }
    
    # Phase 7c: Enhanced Constraint Propagation Methods
    
    def propagate_constraint(self, source_stream: StreamType, constraint_type: ConstraintType,
                           intensity: float, metadata: Dict[str, Any] = None) -> bool:
        """
        Propagate a constraint from one stream to others.
        
        Args:
            source_stream: Stream originating the constraint
            constraint_type: Type of constraint to propagate
            intensity: Strength of constraint (0.0-1.0)
            metadata: Additional constraint information
            
        Returns:
            True if constraint was propagated successfully
        """
        success = self.constraint_propagation.propagate_constraint(
            source_stream, constraint_type, intensity, metadata
        )
        
        if success:
            self._stats['constraint_propagations'] += 1
        
        return success
    
    def get_constraint_pressures(self, stream: StreamType) -> Dict[ConstraintType, float]:
        """
        Get current constraint pressures affecting a specific stream.
        
        Args:
            stream: Stream to get constraints for
            
        Returns:
            Dictionary mapping constraint types to their current intensities
        """
        return self.constraint_propagation.get_stream_constraints(stream)
    
    def get_total_constraint_pressure(self, stream: StreamType) -> float:
        """
        Get total constraint pressure on a stream (sum of all constraint types).
        
        Args:
            stream: Stream to get total pressure for
            
        Returns:
            Total constraint pressure (0.0-1.0+)
        """
        return self.constraint_propagation.get_total_constraint_pressure(stream)
    
    def request_resource_with_constraints(self, stream_type: StreamType, resource_type: str, 
                                        amount: float) -> float:
        """
        Request shared resource allocation using emergent attention allocation and constraint pressures.
        
        This enhanced version uses emergent attention allocation for attention_budget and
        constraint-based allocation for other resources.
        
        Args:
            stream_type: Stream requesting the resource
            resource_type: Type of resource ('attention_budget', 'processing_energy', etc.)
            amount: Amount of resource requested
            
        Returns:
            Amount of resource actually allocated (may be less than requested)
        """
        with self._resource_lock:
            # Special handling for attention budget - use emergent allocation
            if resource_type == 'attention_budget':
                allocated_attention = self.get_allocated_attention(stream_type)
                # Return the amount from emergent allocation, scaled by request
                return min(amount, allocated_attention)
            
            # For other resources, use constraint-based allocation
            if resource_type not in self._shared_resources:
                return 0.0
            
            available = self._shared_resources[resource_type]
            
            # Get constraint pressures for this stream
            constraint_pressures = self.get_constraint_pressures(stream_type)
            
            # Calculate constraint-based priority modifier
            urgency_boost = constraint_pressures.get(ConstraintType.URGENCY_SIGNAL, 0.0)
            load_penalty = constraint_pressures.get(ConstraintType.PROCESSING_LOAD, 0.0)
            scarcity_pressure = constraint_pressures.get(ConstraintType.RESOURCE_SCARCITY, 0.0)
            
            # Priority = base + urgency boost - load penalty
            priority_modifier = 1.0 + (urgency_boost * 0.3) - (load_penalty * 0.2)
            priority_modifier = max(0.1, min(2.0, priority_modifier))  # Clamp to reasonable range
            
            # For attention-adjacent resources, consider emergent attention allocation
            if resource_type in ['processing_energy', 'working_memory_slots']:
                allocated_attention = self.get_allocated_attention(stream_type)
                attention_boost = allocated_attention * 0.5  # Attention winners get more processing resources
                priority_modifier *= (1.0 + attention_boost)
            
            # Adjust requested amount based on priority
            adjusted_amount = amount * priority_modifier
            
            # Allocate resources
            allocated = min(adjusted_amount, available)
            
            if allocated > 0:
                self._shared_resources[resource_type] -= allocated
                self._stats['resource_competitions'] += 1
                
                # If allocation was limited due to scarcity, propagate scarcity constraint
                if allocated < adjusted_amount * 0.8:  # Got less than 80% of adjusted request
                    scarcity_intensity = min(1.0, (adjusted_amount - allocated) / adjusted_amount)
                    self.propagate_constraint(
                        stream_type, 
                        ConstraintType.RESOURCE_SCARCITY,
                        scarcity_intensity,
                        {'resource_type': resource_type, 'shortage': adjusted_amount - allocated}
                    )
                
                # Log significant allocation changes
                if not self.quiet_mode and abs(priority_modifier - 1.0) > 0.1:
                    print(f"🔗 Enhanced allocation: {stream_type.value} priority={priority_modifier:.2f}")
                    if resource_type in ['processing_energy', 'working_memory_slots']:
                        allocated_attention = self.get_allocated_attention(stream_type)
                        print(f"   Attention: {allocated_attention:.3f}, Urgency: {urgency_boost:.2f}, Load: {load_penalty:.2f}")
            
            return allocated
    
    def update_constraint_dynamics(self):
        """Update constraint propagation dynamics and cleanup expired constraints."""
        self.constraint_propagation.update_constraint_dynamics()
        
        # Update constraint-based resource pressures
        self._update_resource_pressures_from_constraints()
    
    def _update_resource_pressures_from_constraints(self):
        """Update resource availability based on constraint pressures across streams."""
        # Calculate total system constraint pressure
        total_processing_load = 0.0
        total_urgency = 0.0
        
        for stream in StreamType:
            constraints = self.get_constraint_pressures(stream)
            total_processing_load += constraints.get(ConstraintType.PROCESSING_LOAD, 0.0)
            total_urgency += constraints.get(ConstraintType.URGENCY_SIGNAL, 0.0)
        
        # Adjust base resource availability based on system-wide pressures
        with self._resource_lock:
            # High system load reduces base resources (system protection)
            if total_processing_load > 2.0:  # More than 2 streams under load
                load_factor = max(0.7, 1.0 - (total_processing_load - 2.0) * 0.1)
                self._shared_resources['processing_energy'] *= load_factor
            
            # High urgency temporarily increases attention budget (emergency mode)
            if total_urgency > 1.5:  # Significant urgency in system
                urgency_factor = min(1.3, 1.0 + (total_urgency - 1.5) * 0.1)
                self._shared_resources['attention_budget'] = min(1.5, self._shared_resources['attention_budget'] * urgency_factor)
    
    def run_emergent_attention_allocation(self):
        """
        Run emergent attention allocation cycle based on current stream states and constraints.
        
        This method coordinates attention allocation through competitive bidding where
        streams compete for limited attention resources based on their current state.
        """
        # Collect current stream states
        stream_states = self.get_all_stream_states()
        
        # Collect constraint pressures for all streams
        constraint_pressures = {}
        for stream_type in StreamType:
            constraint_pressures[stream_type] = self.get_constraint_pressures(stream_type)
        
        # Compute attention bids from all streams
        bids = self.attention_allocator.compute_attention_bids(stream_states, constraint_pressures)
        
        # Allocate attention through competition
        allocations = self.attention_allocator.allocate_attention(bids)
        
        # Update attention budget based on allocations
        self._update_attention_budget_from_allocations(allocations)
        
        return allocations
    
    def _update_attention_budget_from_allocations(self, allocations):
        """Update the shared attention budget based on competitive allocations."""
        with self._resource_lock:
            # Set individual stream attention allocations
            total_allocated = 0.0
            for allocation in allocations:
                stream_attention_key = f'attention_{allocation.stream_type.value}'
                allocated_amount = allocation.allocated_attention
                
                # Store per-stream attention allocation
                self._shared_resources[stream_attention_key] = allocated_amount
                total_allocated += allocated_amount
            
            # Update global attention budget (remaining after allocations)
            self._shared_resources['attention_budget'] = max(0.0, 1.0 - total_allocated)
            
            if not self.quiet_mode and allocations:
                winner = max(allocations, key=lambda a: a.allocated_attention)
                print(f"🧠 Emergent attention: {winner.stream_type.value} allocated {winner.allocated_attention:.3f}")
    
    def get_allocated_attention(self, stream_type: StreamType) -> float:
        """Get the currently allocated attention for a specific stream."""
        stream_attention_key = f'attention_{stream_type.value}'
        return self._shared_resources.get(stream_attention_key, 0.0)
    
    def run_constraint_pattern_inhibition(self) -> Dict[StreamType, List[int]]:
        """
        Run constraint-based pattern inhibition and selection across all streams.
        
        This method coordinates pattern competition where patterns compete for
        limited activation resources based on constraint dynamics.
        """
        # Collect active patterns from all streams
        stream_patterns = {}
        stream_activations = {}
        
        for stream_type, stream_state in self._stream_states.items():
            if hasattr(stream_state, 'active_patterns') and stream_state.active_patterns:
                stream_patterns[stream_type] = stream_state.active_patterns
                
                # Get pattern activations (use activation_strength as proxy)
                activation = getattr(stream_state, 'activation_strength', 0.5)
                pattern_count = len(stream_state.active_patterns)
                
                # Create individual pattern activations
                # In a real implementation, each pattern would have its own activation
                if pattern_count > 0:
                    base_activation = activation / pattern_count
                    # Add some variation to pattern activations
                    activations = []
                    for i, pattern in enumerate(stream_state.active_patterns):
                        # Use pattern ID to create consistent but varied activations
                        pattern_variation = 0.1 * ((pattern % 10) - 5) / 5  # -0.1 to +0.1
                        pattern_activation = max(0.1, min(1.0, base_activation + pattern_variation))
                        activations.append(pattern_activation)
                    stream_activations[stream_type] = activations
        
        # Get constraint pressures for all streams
        constraint_pressures = {}
        for stream_type in StreamType:
            constraint_pressures[stream_type] = self.get_constraint_pressures(stream_type)
        
        # Run constraint-based pattern inhibition and selection
        selected_patterns = self.pattern_inhibitor.update_active_patterns(
            stream_patterns, stream_activations, constraint_pressures
        )
        
        # Update stream states with selected patterns
        self._update_stream_states_from_pattern_selection(selected_patterns)
        
        if not self.quiet_mode and selected_patterns:
            total_selected = sum(len(patterns) for patterns in selected_patterns.values())
            total_original = sum(len(patterns) for patterns in stream_patterns.values())
            print(f"🧠 Pattern inhibition: {total_selected}/{total_original} patterns selected")
        
        return selected_patterns
    
    def _update_stream_states_from_pattern_selection(self, selected_patterns: Dict[StreamType, List[int]]):
        """Update stream states based on pattern selection results."""
        with self._stream_lock:
            for stream_type, patterns in selected_patterns.items():
                if stream_type in self._stream_states:
                    stream_state = self._stream_states[stream_type]
                    
                    # Update active patterns with selected ones
                    if hasattr(stream_state, 'active_patterns'):
                        stream_state.active_patterns = patterns
                    
                    # Update pattern count in metadata
                    stream_state.processing_phase = "inhibition_applied"
    
    def reset_cycle_resources(self):
        """Reset shared resources at the start of each gamma cycle."""
        # Update constraint dynamics before resource reset
        self.update_constraint_dynamics()
        
        # Run emergent attention allocation
        self.run_emergent_attention_allocation()
        
        # Run constraint-based pattern inhibition and selection
        self.run_constraint_pattern_inhibition()
        
        # Automatic cleanup every 10 cycles to prevent memory leaks
        self._stats['total_updates'] += 1
        if self._stats['total_updates'] % 10 == 0:
            self.cleanup_old_data(max_age_seconds=30.0)  # More aggressive cleanup
            self.constraint_propagation.cleanup_expired_constraints(max_age_seconds=15.0)
        
        with self._resource_lock:
            # Reset renewable resources (but preserve attention allocations)
            # self._shared_resources['attention_budget'] = 1.0  # Now managed by attention allocator
            self._shared_resources['processing_energy'] = 1.0
            
            # Working memory and binding capacity persist across cycles


# Factory function for easy creation
def create_shared_brain_state(biological_oscillator=None, quiet_mode: bool = False) -> SharedBrainState:
    """
    Create shared brain state with biological oscillator integration.
    
    Args:
        biological_oscillator: Optional biological oscillator for timing
        quiet_mode: Suppress debug output
        
    Returns:
        Configured SharedBrainState instance
    """
    return SharedBrainState(biological_oscillator, quiet_mode)


# Example usage and testing
if __name__ == "__main__":
    print("🧠 Testing Shared Brain State Infrastructure")
    
    # Create shared state
    shared_state = create_shared_brain_state(quiet_mode=False)
    
    # Test stream state updates
    print(f"\n🔬 Testing stream state updates...")
    success = shared_state.update_stream_state(
        StreamType.SENSORY, 
        processing_phase="active",
        active_patterns=[1, 2, 3],
        activation_strength=0.8
    )
    print(f"Sensory stream update: {'✅' if success else '❌'}")
    
    # Test resource allocation
    print(f"\n🔬 Testing resource allocation...")
    attention_allocated = shared_state.request_resource(
        StreamType.SENSORY, 'attention_budget', 0.6
    )
    print(f"Attention allocated: {attention_allocated:.2f}")
    
    # Test cross-stream binding
    print(f"\n🔬 Testing cross-stream binding...")
    binding_success = shared_state.create_cross_stream_binding(
        StreamType.SENSORY, StreamType.MOTOR,
        [1, 2, 3], [4, 5, 6], 0.9
    )
    print(f"Cross-stream binding: {'✅' if binding_success else '❌'}")
    
    # Test coordination signals
    print(f"\n🔬 Testing coordination signals...")
    shared_state.set_coordination_signal("sensory_peak", True)
    peak_signal = shared_state.get_coordination_signal("sensory_peak")
    print(f"Coordination signal: {peak_signal}")
    
    # Display statistics
    stats = shared_state.get_shared_state_stats()
    print(f"\n📊 Shared State Statistics:")
    print(f"  Active streams: {sum(1 for s in stats['stream_activity'].values() if s['active'])}")
    print(f"  Active bindings: {stats['active_bindings']}")
    print(f"  Resource allocation: {stats['shared_resources']}")
    print(f"  Total updates: {stats['statistics']['total_updates']}")