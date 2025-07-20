#!/usr/bin/env python3
"""
Parallel Stream Processor

Implements async parallel processing for sensory, motor, and temporal streams
with biological coordination through gamma-frequency oscillations.

Key principles:
- All streams process simultaneously within gamma cycles (25ms)
- Coordination emerges from shared constraints, not explicit synchronization
- Biological oscillator provides natural timing coordination
- Cross-stream information sharing through binding windows
"""

import asyncio
import time
import threading
from typing import Dict, Any, List, Optional, Tuple, Callable
from dataclasses import dataclass
from enum import Enum
import numpy as np

from .biological_oscillator import BiologicalOscillator, OscillationPhase
from .shared_brain_state import SharedBrainState
from .stream_types import StreamType, ConstraintType, StreamState
from .adaptive_constraint_thresholds import AdaptiveConstraintThresholds


@dataclass
class StreamProcessingResult:
    """Result from processing a single stream."""
    stream_type: StreamType
    processing_time_ms: float
    output_patterns: List[int]
    activation_strength: float
    coordination_signals: Dict[str, Any]
    resource_usage: Dict[str, float]
    phase_alignment: bool  # True if processed in optimal phase


class AsyncStreamProcessor:
    """
    Base class for async stream processing.
    
    Each stream processes independently but coordinates through shared state
    and biological timing constraints.
    """
    
    def __init__(self, stream_type: StreamType, shared_state: SharedBrainState, 
                 biological_oscillator: BiologicalOscillator, quiet_mode: bool = False,
                 adaptive_thresholds: AdaptiveConstraintThresholds = None):
        """
        Initialize async stream processor.
        
        Args:
            stream_type: Type of stream this processor handles
            shared_state: Shared brain state for coordination
            biological_oscillator: Biological timing coordinator
            quiet_mode: Suppress debug output
            adaptive_thresholds: Adaptive constraint thresholds system
        """
        self.stream_type = stream_type
        self.shared_state = shared_state
        self.biological_oscillator = biological_oscillator
        self.quiet_mode = quiet_mode
        self.adaptive_thresholds = adaptive_thresholds
        
        # Processing state
        self.processing_active = False
        self.last_processing_time = 0.0
        self.total_processing_cycles = 0
        
        # Resource tracking
        self.resource_usage_history = []
        
        # Performance tracking
        self.processing_times = []
        self.phase_alignments = []
        
        if not quiet_mode:
            print(f"🧠 AsyncStreamProcessor initialized: {stream_type.value}")
    
    async def process_async(self, input_data: Any, processing_budget_ms: float = 20.0) -> StreamProcessingResult:
        """
        Process stream data asynchronously with biological timing constraints.
        
        Args:
            input_data: Input data for this stream
            processing_budget_ms: Maximum processing time available
            
        Returns:
            StreamProcessingResult with processing outcomes
        """
        start_time = time.time()
        
        # Check biological timing alignment
        timing = self.biological_oscillator.get_current_timing()
        optimal_phase = self._get_optimal_phase()
        phase_aligned = timing.oscillation_phase == optimal_phase
        
        # Phase 7c: Use constraint-aware resource allocation
        attention_allocated = self.shared_state.request_resource_with_constraints(
            self.stream_type, 'attention_budget', 0.3
        )
        energy_allocated = self.shared_state.request_resource_with_constraints(
            self.stream_type, 'processing_energy', 0.4
        )
        
        # Get current constraint pressures affecting this stream
        constraint_pressures = self.shared_state.get_constraint_pressures(self.stream_type)
        total_pressure = self.shared_state.get_total_constraint_pressure(self.stream_type)
        
        # Phase 7c: Adjust processing depth based on constraints, resources, and timing
        base_depth = min(
            attention_allocated * 2.0,  # More attention = deeper processing
            energy_allocated * 2.5,     # More energy = more computation
            1.0 if phase_aligned else 0.5  # Optimal phase = full processing
        )
        
        # Apply constraint-based modulation
        constraint_modifier = 1.0
        
        # High processing load reduces depth (resource conservation)
        load_pressure = constraint_pressures.get(ConstraintType.PROCESSING_LOAD, 0.0)
        constraint_modifier *= (1.0 - load_pressure * 0.3)
        
        # High urgency increases depth (priority processing)
        urgency_pressure = constraint_pressures.get(ConstraintType.URGENCY_SIGNAL, 0.0)
        constraint_modifier *= (1.0 + urgency_pressure * 0.2)
        
        # Resource scarcity reduces depth (conservation mode)
        scarcity_pressure = constraint_pressures.get(ConstraintType.RESOURCE_SCARCITY, 0.0)
        constraint_modifier *= (1.0 - scarcity_pressure * 0.4)
        
        # Apply constraint modulation
        processing_depth = base_depth * constraint_modifier
        processing_depth = max(0.1, min(2.0, processing_depth))  # Clamp to reasonable range
        
        try:
            # Mark as processing
            self.processing_active = True
            self.shared_state.update_stream_state(
                self.stream_type,
                processing_phase="active",
                processing_budget_remaining=processing_budget_ms
            )
            
            # Perform stream-specific processing
            output_patterns, activation_strength = await self._process_stream_data(
                input_data, processing_depth, processing_budget_ms
            )
            
            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000
            
            # Update stream state with results
            self.shared_state.update_stream_state(
                self.stream_type,
                processing_phase="completed",
                active_patterns=output_patterns,
                activation_strength=activation_strength,
                processing_budget_remaining=max(0, processing_budget_ms - processing_time)
            )
            
            # Generate coordination signals
            coordination_signals = self._generate_coordination_signals(
                output_patterns, activation_strength, phase_aligned
            )
            
            # Set coordination signals for other streams
            for signal_name, signal_value in coordination_signals.items():
                self.shared_state.set_coordination_signal(
                    f"{self.stream_type.value}_{signal_name}", signal_value
                )
            
            # Track performance
            self.processing_times.append(processing_time)
            self.phase_alignments.append(phase_aligned)
            self.total_processing_cycles += 1
            
            # Resource usage tracking
            resource_usage = {
                'attention_used': attention_allocated,
                'energy_used': energy_allocated,
                'processing_depth': processing_depth
            }
            self.resource_usage_history.append(resource_usage)
            
            # Phase 7c: Detect and propagate constraints based on processing conditions
            self._detect_and_propagate_constraints(
                processing_time, processing_budget_ms, activation_strength, 
                attention_allocated, energy_allocated, output_patterns
            )
            
            return StreamProcessingResult(
                stream_type=self.stream_type,
                processing_time_ms=processing_time,
                output_patterns=output_patterns,
                activation_strength=activation_strength,
                coordination_signals=coordination_signals,
                resource_usage=resource_usage,
                phase_alignment=phase_aligned
            )
            
        finally:
            # Always clean up
            self.processing_active = False
            
            # Release unused resources
            self.shared_state.release_resource('attention_budget', max(0, attention_allocated * 0.1))
            self.shared_state.release_resource('processing_energy', max(0, energy_allocated * 0.1))
    
    async def _process_stream_data(self, input_data: Any, processing_depth: float, 
                                 budget_ms: float) -> Tuple[List[int], float]:
        """
        Stream-specific processing implementation.
        
        Subclasses should override this method.
        
        Args:
            input_data: Input data for processing
            processing_depth: Depth of processing (0.0-1.0)
            budget_ms: Processing time budget
            
        Returns:
            Tuple of (output_patterns, activation_strength)
        """
        # Simulate processing delay based on depth
        processing_delay = processing_depth * 0.01  # Up to 10ms for full processing
        await asyncio.sleep(processing_delay)
        
        # Default implementation - return empty patterns
        return [], 0.5
    
    def _get_optimal_phase(self) -> OscillationPhase:
        """Get the optimal oscillation phase for this stream type."""
        phase_mapping = {
            StreamType.SENSORY: OscillationPhase.SENSORY_WINDOW,
            StreamType.MOTOR: OscillationPhase.MOTOR_WINDOW,
            StreamType.TEMPORAL: OscillationPhase.INTEGRATION_WINDOW,
            StreamType.CONFIDENCE: OscillationPhase.INTEGRATION_WINDOW,
            StreamType.ATTENTION: OscillationPhase.INTEGRATION_WINDOW
        }
        return phase_mapping.get(self.stream_type, OscillationPhase.INTEGRATION_WINDOW)
    
    def _generate_coordination_signals(self, output_patterns: List[int], 
                                     activation_strength: float, 
                                     phase_aligned: bool) -> Dict[str, Any]:
        """Generate coordination signals for other streams."""
        return {
            'active': len(output_patterns) > 0,
            'strength': activation_strength,
            'pattern_count': len(output_patterns),
            'phase_aligned': phase_aligned,
            'top_patterns': output_patterns[:3] if output_patterns else [],
            'processing_time': self.processing_times[-1] if self.processing_times else 0
        }
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for this stream."""
        if not self.processing_times:
            return {'status': 'no_data'}
        
        recent_times = self.processing_times[-10:]
        recent_alignments = self.phase_alignments[-10:]
        
        return {
            'stream_type': self.stream_type.value,
            'total_cycles': self.total_processing_cycles,
            'avg_processing_time_ms': np.mean(recent_times),
            'min_processing_time_ms': np.min(recent_times),
            'max_processing_time_ms': np.max(recent_times),
            'phase_alignment_rate': np.mean(recent_alignments),
            'resource_efficiency': np.mean([
                r['processing_depth'] for r in self.resource_usage_history[-10:]
            ]) if self.resource_usage_history else 0
        }
    
    def _detect_and_propagate_constraints(self, processing_time: float, budget_ms: float,
                                        activation_strength: float, attention_used: float,
                                        energy_used: float, output_patterns: List[int]):
        """
        Phase 7c: Detect processing conditions and propagate appropriate constraints.
        
        This implements biological constraint propagation where processing challenges
        naturally create adaptive pressure in other streams.
        """
        
        # Update adaptive thresholds with current load measurement
        if self.adaptive_thresholds:
            self.adaptive_thresholds.update_load_measurement(
                self.stream_type, processing_time, budget_ms, attention_used, energy_used
            )
        
        if not self.quiet_mode:
            print(f"🔍 {self.stream_type.value}: time={processing_time:.2f}ms, budget={budget_ms:.2f}ms, "
                  f"activation={activation_strength:.2f}, attention={attention_used:.2f}, "
                  f"energy={energy_used:.2f}, patterns={len(output_patterns)}")
        
        # Get adaptive thresholds or use defaults
        load_threshold = (self.adaptive_thresholds.get_threshold(ConstraintType.PROCESSING_LOAD) 
                         if self.adaptive_thresholds else 0.7)
        
        # Detect processing load constraint
        load_ratio = processing_time / budget_ms if budget_ms > 0 else 1.0
        if load_ratio > load_threshold:
            load_intensity = min(1.0, (load_ratio - load_threshold) / (1.0 - load_threshold))
            self.shared_state.propagate_constraint(
                self.stream_type,
                ConstraintType.PROCESSING_LOAD,
                load_intensity,
                {
                    'processing_time_ms': processing_time,
                    'budget_ms': budget_ms,
                    'load_ratio': load_ratio,
                    'threshold_used': load_threshold,
                    'cause': 'high_processing_demand'
                }
            )
            
            # Log constraint event for threshold adaptation
            if self.adaptive_thresholds:
                self.adaptive_thresholds.log_constraint_event(ConstraintType.PROCESSING_LOAD, load_intensity)
        
        # Get adaptive threshold for resource scarcity
        scarcity_threshold = (self.adaptive_thresholds.get_threshold(ConstraintType.RESOURCE_SCARCITY) 
                             if self.adaptive_thresholds else 0.3)
        
        # Detect resource scarcity constraint
        resource_shortage = max(0, 0.5 - attention_used) + max(0, 0.5 - energy_used)
        if resource_shortage > scarcity_threshold:
            scarcity_intensity = min(1.0, resource_shortage / scarcity_threshold)
            self.shared_state.propagate_constraint(
                self.stream_type,
                ConstraintType.RESOURCE_SCARCITY,
                scarcity_intensity,
                {
                    'attention_shortage': max(0, 0.5 - attention_used),
                    'energy_shortage': max(0, 0.5 - energy_used),
                    'total_shortage': resource_shortage,
                    'threshold_used': scarcity_threshold,
                    'cause': 'insufficient_resource_allocation'
                }
            )
            
            # Log constraint event for threshold adaptation
            if self.adaptive_thresholds:
                self.adaptive_thresholds.log_constraint_event(ConstraintType.RESOURCE_SCARCITY, scarcity_intensity)
        
        # Get adaptive threshold for urgency
        urgency_threshold = (self.adaptive_thresholds.get_threshold(ConstraintType.URGENCY_SIGNAL) 
                            if self.adaptive_thresholds else 0.6)
        
        # Detect urgency based on high activation with low resources  
        if activation_strength > urgency_threshold and (attention_used < 0.4 or energy_used < 0.4):
            urgency_intensity = activation_strength * (1.0 - min(attention_used, energy_used))
            self.shared_state.propagate_constraint(
                self.stream_type,
                ConstraintType.URGENCY_SIGNAL,
                urgency_intensity,
                {
                    'activation_strength': activation_strength,
                    'attention_available': attention_used,
                    'energy_available': energy_used,
                    'threshold_used': urgency_threshold,
                    'cause': 'high_activation_low_resources'
                }
            )
            
            # Log constraint event for threshold adaptation
            if self.adaptive_thresholds:
                self.adaptive_thresholds.log_constraint_event(ConstraintType.URGENCY_SIGNAL, urgency_intensity)
        
        # Get adaptive threshold for interference
        interference_threshold = (self.adaptive_thresholds.get_threshold(ConstraintType.INTERFERENCE) 
                                 if self.adaptive_thresholds else 0.6)
        
        # Detect interference based on high processing time with low output
        if processing_time > budget_ms * interference_threshold and len(output_patterns) < 5:
            interference_intensity = min(1.0, processing_time / budget_ms)
            self.shared_state.propagate_constraint(
                self.stream_type,
                ConstraintType.INTERFERENCE,
                interference_intensity,
                {
                    'processing_time_ms': processing_time,
                    'output_pattern_count': len(output_patterns),
                    'efficiency': len(output_patterns) / (processing_time / 10) if processing_time > 0 else 0,
                    'threshold_used': interference_threshold,
                    'cause': 'high_processing_low_output'
                }
            )
            
            # Log constraint event for threshold adaptation
            if self.adaptive_thresholds:
                self.adaptive_thresholds.log_constraint_event(ConstraintType.INTERFERENCE, interference_intensity)
        
        # Get adaptive threshold for coherence pressure
        coherence_threshold = (self.adaptive_thresholds.get_threshold(ConstraintType.COHERENCE_PRESSURE) 
                              if self.adaptive_thresholds else 0.2)
        
        # Detect coherence pressure based on activation strength variability
        if len(self.resource_usage_history) >= 3:
            recent_activations = [r.get('activation_strength', 0.5) for r in self.resource_usage_history[-3:]]
            activation_variance = np.var(recent_activations)
            
            if activation_variance > coherence_threshold:
                coherence_intensity = min(1.0, activation_variance / coherence_threshold)
                self.shared_state.propagate_constraint(
                    self.stream_type,
                    ConstraintType.COHERENCE_PRESSURE,
                    coherence_intensity,
                    {
                        'activation_variance': activation_variance,
                        'recent_activations': recent_activations,
                        'current_activation': activation_strength,
                        'threshold_used': coherence_threshold,
                        'cause': 'inconsistent_activation_patterns'
                    }
                )
                
                # Log constraint event for threshold adaptation
                if self.adaptive_thresholds:
                    self.adaptive_thresholds.log_constraint_event(ConstraintType.COHERENCE_PRESSURE, coherence_intensity)
        
        # Get adaptive threshold for energy depletion
        depletion_threshold = (self.adaptive_thresholds.get_threshold(ConstraintType.ENERGY_DEPLETION) 
                              if self.adaptive_thresholds else 2.0)
        
        # Detect energy depletion based on processing efficiency trends
        if len(self.processing_times) >= 5:
            recent_times = self.processing_times[-5:]
            time_trend = np.mean(recent_times[-3:]) - np.mean(recent_times[:2])
            
            if time_trend > depletion_threshold:
                depletion_intensity = min(1.0, time_trend / depletion_threshold)
                self.shared_state.propagate_constraint(
                    self.stream_type,
                    ConstraintType.ENERGY_DEPLETION,
                    depletion_intensity,
                    {
                        'time_trend_ms': time_trend,
                        'recent_avg_time': np.mean(recent_times[-3:]),
                        'early_avg_time': np.mean(recent_times[:2]),
                        'efficiency_decline': time_trend,
                        'threshold_used': depletion_threshold,
                        'cause': 'increasing_processing_times'
                    }
                )
                
                # Log constraint event for threshold adaptation
                if self.adaptive_thresholds:
                    self.adaptive_thresholds.log_constraint_event(ConstraintType.ENERGY_DEPLETION, depletion_intensity)


class SensoryStreamProcessor(AsyncStreamProcessor):
    """Async processor for sensory stream."""
    
    def __init__(self, sensory_stream, shared_state: SharedBrainState, 
                 biological_oscillator: BiologicalOscillator, quiet_mode: bool = False,
                 adaptive_thresholds = None):
        super().__init__(StreamType.SENSORY, shared_state, biological_oscillator, quiet_mode, adaptive_thresholds)
        self.sensory_stream = sensory_stream
    
    async def _process_stream_data(self, sensory_input: List[float], 
                                 processing_depth: float, budget_ms: float) -> Tuple[List[int], float]:
        """Process sensory input asynchronously."""
        # Simulate async processing
        await asyncio.sleep(0.002)  # 2ms base processing time
        
        # Use existing sensory stream logic
        if hasattr(self.sensory_stream, 'update'):
            # Convert list input to torch.Tensor for sparse stream compatibility
            if isinstance(sensory_input, list):
                import torch
                sensory_tensor = torch.tensor(sensory_input, dtype=torch.float32)
            else:
                sensory_tensor = sensory_input
                
            activation_info = self.sensory_stream.update(sensory_tensor, time.time())
            active_patterns = self.sensory_stream.get_active_pattern_indices(k=5)
            activation_strength = activation_info.get('activation_strength', 0.5)
            
            return active_patterns, activation_strength
        
        # Fallback for testing
        return [1, 2, 3], 0.7


class MotorStreamProcessor(AsyncStreamProcessor):
    """Async processor for motor stream."""
    
    def __init__(self, motor_stream, shared_state: SharedBrainState, 
                 biological_oscillator: BiologicalOscillator, quiet_mode: bool = False,
                 adaptive_thresholds = None):
        super().__init__(StreamType.MOTOR, shared_state, biological_oscillator, quiet_mode, adaptive_thresholds)
        self.motor_stream = motor_stream
    
    async def _process_stream_data(self, motor_input: List[float], 
                                 processing_depth: float, budget_ms: float) -> Tuple[List[int], float]:
        """Process motor output asynchronously."""
        # Simulate async processing
        await asyncio.sleep(0.003)  # 3ms base processing time
        
        # Use existing motor stream logic
        if hasattr(self.motor_stream, 'update'):
            # Convert list input to torch.Tensor for sparse stream compatibility
            if isinstance(motor_input, list):
                import torch
                motor_tensor = torch.tensor(motor_input, dtype=torch.float32)
            else:
                motor_tensor = motor_input
                
            activation_info = self.motor_stream.update(motor_tensor, time.time())
            active_patterns = self.motor_stream.get_active_pattern_indices(k=5)
            activation_strength = activation_info.get('activation_strength', 0.5)
            
            return active_patterns, activation_strength
        
        # Fallback for testing
        return [4, 5, 6], 0.6


class TemporalStreamProcessor(AsyncStreamProcessor):
    """Async processor for temporal stream."""
    
    def __init__(self, temporal_stream, shared_state: SharedBrainState, 
                 biological_oscillator: BiologicalOscillator, quiet_mode: bool = False,
                 adaptive_thresholds = None):
        super().__init__(StreamType.TEMPORAL, shared_state, biological_oscillator, quiet_mode, adaptive_thresholds)
        self.temporal_stream = temporal_stream
    
    async def _process_stream_data(self, temporal_input: List[float], 
                                 processing_depth: float, budget_ms: float) -> Tuple[List[int], float]:
        """Process temporal context asynchronously."""
        # Simulate async processing
        await asyncio.sleep(0.002)  # 2ms base processing time
        
        # Use existing temporal stream logic
        if hasattr(self.temporal_stream, 'update'):
            # Convert list input to torch.Tensor for sparse stream compatibility
            if isinstance(temporal_input, list):
                import torch
                temporal_tensor = torch.tensor(temporal_input, dtype=torch.float32)
            else:
                temporal_tensor = temporal_input
                
            activation_info = self.temporal_stream.update(temporal_tensor, time.time())
            active_patterns = self.temporal_stream.get_active_pattern_indices(k=5)
            activation_strength = activation_info.get('activation_strength', 0.5)
            
            return active_patterns, activation_strength
        
        # Fallback for testing
        return [7, 8, 9], 0.8


class ParallelStreamCoordinator:
    """
    Coordinates parallel stream processing with biological timing.
    
    Manages the async execution of all streams within gamma cycles
    while maintaining biological constraints and enabling emergent coordination.
    """
    
    def __init__(self, sensory_stream, motor_stream, temporal_stream,
                 shared_state: SharedBrainState, biological_oscillator: BiologicalOscillator,
                 quiet_mode: bool = False):
        """
        Initialize parallel stream coordinator.
        
        Args:
            sensory_stream: Sensory stream instance
            motor_stream: Motor stream instance  
            temporal_stream: Temporal stream instance
            shared_state: Shared brain state for coordination
            biological_oscillator: Biological timing coordinator
            quiet_mode: Suppress debug output
        """
        self.shared_state = shared_state
        self.biological_oscillator = biological_oscillator
        self.quiet_mode = quiet_mode
        
        # Create adaptive constraint thresholds system
        from .adaptive_constraint_thresholds import create_adaptive_thresholds
        self.adaptive_thresholds = create_adaptive_thresholds(quiet_mode)
        
        # Create async stream processors with adaptive thresholds
        self.processors = {
            StreamType.SENSORY: SensoryStreamProcessor(
                sensory_stream, shared_state, biological_oscillator, quiet_mode, self.adaptive_thresholds
            ),
            StreamType.MOTOR: MotorStreamProcessor(
                motor_stream, shared_state, biological_oscillator, quiet_mode, self.adaptive_thresholds
            ),
            StreamType.TEMPORAL: TemporalStreamProcessor(
                temporal_stream, shared_state, biological_oscillator, quiet_mode, self.adaptive_thresholds
            )
        }
        
        # Coordination statistics
        self.coordination_stats = {
            'total_parallel_cycles': 0,
            'successful_bindings': 0,
            'resource_conflicts': 0,
            'phase_misalignments': 0
        }
        
        if not quiet_mode:
            print(f"🧠 ParallelStreamCoordinator initialized")
            print(f"   Streams: {[s.value for s in self.processors.keys()]}")
    
    async def process_parallel_cycle(self, sensory_input: List[float], 
                                   motor_prediction: List[float],
                                   temporal_context: List[float],
                                   processing_budget_ms: float = 18.0) -> Dict[str, Any]:
        """
        Process one complete parallel cycle with all streams.
        
        Args:
            sensory_input: Sensory input data
            motor_prediction: Motor prediction data
            temporal_context: Temporal context data
            processing_budget_ms: Total processing budget for this cycle
            
        Returns:
            Combined results from all parallel streams
        """
        cycle_start_time = time.time()
        
        # Reset shared resources for new cycle
        self.shared_state.reset_cycle_resources()
        
        # Distribute processing budget across streams
        budget_per_stream = processing_budget_ms / 3
        
        # Create async tasks for all streams
        tasks = [
            self.processors[StreamType.SENSORY].process_async(
                sensory_input, budget_per_stream
            ),
            self.processors[StreamType.MOTOR].process_async(
                motor_prediction, budget_per_stream
            ),
            self.processors[StreamType.TEMPORAL].process_async(
                temporal_context, budget_per_stream
            )
        ]
        
        # Execute all streams in parallel
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            stream_results = {}
            successful_results = []
            
            for i, result in enumerate(results):
                stream_type = list(self.processors.keys())[i]
                if isinstance(result, StreamProcessingResult):
                    stream_results[stream_type] = result
                    successful_results.append(result)
                else:
                    # Handle exceptions
                    if not self.quiet_mode:
                        print(f"⚠️ Stream {stream_type.value} processing failed: {result}")
                    stream_results[stream_type] = None
            
            # Create cross-stream bindings during binding windows
            binding_results = await self._create_cross_stream_bindings(successful_results)
            
            # Calculate cycle statistics
            cycle_time = (time.time() - cycle_start_time) * 1000
            
            # Update coordination statistics
            self.coordination_stats['total_parallel_cycles'] += 1
            self.coordination_stats['successful_bindings'] += len(binding_results)
            
            # Compile parallel processing results
            parallel_results = {
                'cycle_time_ms': cycle_time,
                'stream_results': stream_results,
                'cross_stream_bindings': binding_results,
                'coordination_signals': self.shared_state.get_coordination_signals(),
                'resource_usage': self.shared_state.get_shared_state_stats()['shared_resources'],
                'biological_timing': self.biological_oscillator.get_coordination_signal(),
                'coordination_stats': self.coordination_stats.copy()
            }
            
            return parallel_results
            
        except Exception as e:
            if not self.quiet_mode:
                print(f"❌ Parallel cycle processing failed: {e}")
            return {'error': str(e), 'cycle_time_ms': (time.time() - cycle_start_time) * 1000}
    
    async def _create_cross_stream_bindings(self, results: List[StreamProcessingResult]) -> List[Dict[str, Any]]:
        """Create cross-stream bindings from parallel processing results."""
        bindings = []
        
        # Check if we're in a binding window
        timing = self.biological_oscillator.get_current_timing()
        if not timing.binding_window_active:
            return bindings
        
        # Create bindings between all stream pairs
        for i, result_a in enumerate(results):
            for j, result_b in enumerate(results[i+1:], i+1):
                if result_a.output_patterns and result_b.output_patterns:
                    # Calculate binding strength based on activation strengths
                    binding_strength = (result_a.activation_strength + result_b.activation_strength) / 2
                    
                    # Create binding if strong enough
                    if binding_strength > 0.5:
                        success = self.shared_state.create_cross_stream_binding(
                            result_a.stream_type, result_b.stream_type,
                            result_a.output_patterns, result_b.output_patterns,
                            binding_strength
                        )
                        
                        if success:
                            bindings.append({
                                'source': result_a.stream_type.value,
                                'target': result_b.stream_type.value,
                                'strength': binding_strength,
                                'source_patterns': result_a.output_patterns[:3],
                                'target_patterns': result_b.output_patterns[:3]
                            })
        
        return bindings
    
    def get_parallel_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive parallel processing performance statistics."""
        stream_stats = {}
        for stream_type, processor in self.processors.items():
            stream_stats[stream_type.value] = processor.get_performance_stats()
        
        shared_stats = self.shared_state.get_shared_state_stats()
        
        return {
            'coordination_stats': self.coordination_stats,
            'stream_performance': stream_stats,
            'shared_state_stats': shared_stats,
            'biological_timing': self.biological_oscillator.get_oscillator_stats()
        }


# Example usage and testing
if __name__ == "__main__":
    print("🧠 Testing Parallel Stream Processing")
    
    async def test_parallel_processing():
        from .biological_oscillator import create_biological_oscillator
        from .shared_brain_state import create_shared_brain_state
        
        # Create infrastructure
        oscillator = create_biological_oscillator(quiet_mode=True)
        shared_state = create_shared_brain_state(oscillator, quiet_mode=True)
        
        # Create mock streams (for testing)
        class MockStream:
            def update(self, data, time):
                return {'activation_strength': 0.7}
            def get_active_pattern_indices(self, k=5):
                return list(range(k))
        
        # Create coordinator
        coordinator = ParallelStreamCoordinator(
            MockStream(), MockStream(), MockStream(),
            shared_state, oscillator, quiet_mode=True
        )
        
        print("🔬 Testing parallel cycle processing...")
        
        # Test parallel processing
        results = await coordinator.process_parallel_cycle(
            [0.1, 0.2, 0.3], [0.4, 0.5], [0.6, 0.7, 0.8, 0.9]
        )
        
        print(f"Cycle time: {results['cycle_time_ms']:.2f}ms")
        print(f"Stream results: {len([r for r in results['stream_results'].values() if r])}")
        print(f"Cross-stream bindings: {len(results['cross_stream_bindings'])}")
        
        # Get performance stats
        stats = coordinator.get_parallel_performance_stats()
        print(f"Total parallel cycles: {stats['coordination_stats']['total_parallel_cycles']}")
        
        return True
    
    # Run async test
    asyncio.run(test_parallel_processing())
    print("✅ Parallel stream processing test completed!")