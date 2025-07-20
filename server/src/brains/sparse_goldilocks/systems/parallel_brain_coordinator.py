#!/usr/bin/env python3
"""
Parallel Brain Coordinator

Integrates parallel stream processing with the existing brain architecture.
Provides a bridge between sequential and parallel processing modes for testing.

Key features:
- Async parallel stream coordination
- Biological timing enforcement
- Cross-stream binding during gamma windows
- Performance comparison capabilities
- Backward compatibility with sequential processing
"""

import asyncio
import time
import threading
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

from .biological_oscillator import BiologicalOscillator
from .shared_brain_state import SharedBrainState, StreamType, create_shared_brain_state
from .parallel_stream_processor import ParallelStreamCoordinator


@dataclass
class ParallelProcessingResult:
    """Result from parallel brain processing cycle."""
    motor_output: List[float]
    brain_state: Dict[str, Any]
    parallel_stats: Dict[str, Any]
    processing_time_ms: float
    coordination_success: bool


class ParallelBrainCoordinator:
    """
    Coordinates parallel stream processing within the existing brain architecture.
    
    This class bridges the gap between the existing sequential brain and the new
    parallel architecture, allowing for A/B testing and gradual migration.
    """
    
    def __init__(self, vector_brain, biological_oscillator: BiologicalOscillator, 
                 quiet_mode: bool = False):
        """
        Initialize parallel brain coordinator.
        
        Args:
            vector_brain: Existing vector brain instance (SparseGoldilocksBrain)
            biological_oscillator: Biological timing coordinator
            quiet_mode: Suppress debug output
        """
        self.vector_brain = vector_brain
        self.biological_oscillator = biological_oscillator
        self.quiet_mode = quiet_mode
        
        # Create shared brain state
        self.shared_state = create_shared_brain_state(biological_oscillator, quiet_mode)
        
        # Stream coordinator will be created after vector brain is available
        self.stream_coordinator = None
        
        # Parallel processing mode
        self.parallel_mode_enabled = True
        self.enable_cross_stream_binding = True
        
        # Performance tracking
        self.parallel_cycles = 0
        self.sequential_cycles = 0
        self.performance_comparison = {
            'parallel_times': [],
            'sequential_times': [],
            'parallel_successes': 0,
            'sequential_successes': 0
        }
        
        # Simple circuit breaker for parallel processing failures
        self.parallel_failure_count = 0
        self.parallel_failure_threshold = 5
        self.last_failure_time = 0
        self.circuit_breaker_timeout = 30.0  # 30 seconds
        self.circuit_open = False
        
        # Threading for async operations
        self.async_loop = None
        self.async_thread = None
        self._setup_async_thread()
        
        if not quiet_mode:
            print(f"🧠 ParallelBrainCoordinator initialized")
            print(f"   Parallel mode: {'✅' if self.parallel_mode_enabled else '❌'}")
            print(f"   Cross-stream binding: {'✅' if self.enable_cross_stream_binding else '❌'}")
    
    def _ensure_stream_coordinator(self):
        """Ensure stream coordinator is initialized with vector brain streams."""
        if self.stream_coordinator is None and self.vector_brain is not None:
            if hasattr(self.vector_brain, 'sensory_stream'):
                self.stream_coordinator = ParallelStreamCoordinator(
                    self.vector_brain.sensory_stream,
                    self.vector_brain.motor_stream, 
                    self.vector_brain.temporal_stream,
                    self.shared_state,
                    self.biological_oscillator,
                    self.quiet_mode
                )
                if not self.quiet_mode:
                    print(f"🧠 ParallelStreamCoordinator created with constraint detection")
    
    def set_vector_brain(self, vector_brain):
        """Set the vector brain after initialization."""
        self.vector_brain = vector_brain
        self._ensure_stream_coordinator()
    
    def _setup_async_thread(self):
        """Setup dedicated thread for async operations."""
        def run_async_loop():
            self.async_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.async_loop)
            self.async_loop.run_forever()
        
        self.async_thread = threading.Thread(target=run_async_loop, daemon=True)
        self.async_thread.start()
        
        # Wait for loop to be ready
        while self.async_loop is None:
            time.sleep(0.001)
    
    def process_with_parallel_coordination(self, sensory_input: List[float]) -> Tuple[List[float], Dict[str, Any]]:
        """
        Process sensory input using parallel stream coordination.
        
        Args:
            sensory_input: Sensory input vector
            
        Returns:
            Tuple of (motor_output, brain_state)
        """
        # Check circuit breaker before attempting parallel processing
        current_time = time.time()
        if self.circuit_open:
            # Check if circuit breaker timeout has passed
            if current_time - self.last_failure_time > self.circuit_breaker_timeout:
                self.circuit_open = False
                self.parallel_failure_count = 0
                if not self.quiet_mode:
                    print("🔄 Circuit breaker reset - attempting parallel processing")
            else:
                # Circuit is still open, use sequential mode
                return self._process_sequential_mode(sensory_input)
        
        if self.parallel_mode_enabled:
            return self._process_parallel_mode(sensory_input)
        else:
            return self._process_sequential_mode(sensory_input)
    
    def _process_parallel_mode(self, sensory_input: List[float]) -> Tuple[List[float], Dict[str, Any]]:
        """Process using parallel streams with biological coordination."""
        start_time = time.time()
        
        try:
            # Get biological timing
            timing = self.biological_oscillator.get_current_timing()
            processing_budget = self.biological_oscillator.estimate_processing_budget() * 1000  # Convert to ms
            
            # Run parallel processing in async thread
            future = asyncio.run_coroutine_threadsafe(
                self._async_parallel_processing(sensory_input, processing_budget),
                self.async_loop
            )
            
            # Wait for completion with timeout
            timeout_ms = max(100.0, min(processing_budget, 200.0))  # Min 100ms, Max 200ms for parallel coordination
            try:
                result = future.result(timeout=timeout_ms / 1000)
                processing_time = (time.time() - start_time) * 1000
                
                # Update performance tracking
                self.parallel_cycles += 1
                self.performance_comparison['parallel_times'].append(processing_time)
                if result.coordination_success:
                    self.performance_comparison['parallel_successes'] += 1
                
                return result.motor_output, result.brain_state
                
            except asyncio.TimeoutError:
                self._record_parallel_failure()
                if not self.quiet_mode:
                    print(f"⚠️ Parallel processing timeout ({timeout_ms:.1f}ms), falling back to sequential")
                return self._process_sequential_mode(sensory_input)
            
        except Exception as e:
            self._record_parallel_failure()
            if not self.quiet_mode:
                print(f"⚠️ Parallel processing error: {e}, falling back to sequential")
            return self._process_sequential_mode(sensory_input)
    
    def _record_parallel_failure(self):
        """Record a parallel processing failure for circuit breaker logic."""
        self.parallel_failure_count += 1
        self.last_failure_time = time.time()
        
        if self.parallel_failure_count >= self.parallel_failure_threshold:
            self.circuit_open = True
            if not self.quiet_mode:
                print(f"⚠️ Circuit breaker opened - {self.parallel_failure_count} failures, switching to sequential mode for {self.circuit_breaker_timeout}s")
    
    async def _async_parallel_processing(self, sensory_input: List[float], 
                                       processing_budget_ms: float) -> ParallelProcessingResult:
        """Async parallel processing implementation using ParallelStreamCoordinator."""
        start_time = time.time()
        
        # Ensure stream coordinator is initialized
        self._ensure_stream_coordinator()
        
        # Generate motor prediction and temporal context
        motor_prediction = self._generate_motor_prediction(sensory_input)
        temporal_context = self._generate_temporal_context(time.time())
        
        # Use the ParallelStreamCoordinator with constraint detection
        parallel_results = await self.stream_coordinator.process_parallel_cycle(
            sensory_input,
            motor_prediction, 
            temporal_context,
            processing_budget_ms
        )
        
        # Extract results
        sensory_result = None
        motor_result = None
        temporal_result = None
        
        if 'stream_results' in parallel_results:
            stream_results = parallel_results['stream_results']
            if StreamType.SENSORY in stream_results and stream_results[StreamType.SENSORY]:
                sensory_result = {
                    'stream_type': 'sensory',
                    'active_patterns': stream_results[StreamType.SENSORY].output_patterns,
                    'activation_strength': stream_results[StreamType.SENSORY].activation_strength
                }
            if StreamType.MOTOR in stream_results and stream_results[StreamType.MOTOR]:
                motor_result = {
                    'stream_type': 'motor', 
                    'active_patterns': stream_results[StreamType.MOTOR].output_patterns,
                    'activation_strength': stream_results[StreamType.MOTOR].activation_strength
                }
            if StreamType.TEMPORAL in stream_results and stream_results[StreamType.TEMPORAL]:
                temporal_result = {
                    'stream_type': 'temporal',
                    'active_patterns': stream_results[StreamType.TEMPORAL].output_patterns,
                    'activation_strength': stream_results[StreamType.TEMPORAL].activation_strength
                }
        
        # Generate motor output from motor stream result
        motor_output = self._synthesize_motor_output(sensory_result, motor_result, temporal_result)
        
        # Compile brain state including constraint propagation results  
        processing_time = (time.time() - start_time) * 1000
        
        # Get constraint propagation stats from shared state
        shared_stats = self.shared_state.get_shared_state_stats()
        
        # Get prediction confidence from vector brain
        prediction_confidence = 0.5  # Default fallback
        if hasattr(self.vector_brain, '_estimate_prediction_confidence'):
            try:
                prediction_confidence = self.vector_brain._estimate_prediction_confidence()
            except:
                prediction_confidence = 0.5
        elif hasattr(self.vector_brain, 'emergent_confidence'):
            try:
                prediction_confidence = self.vector_brain.emergent_confidence.current_confidence
            except:
                prediction_confidence = 0.5
        
        brain_state = {
            'total_cycles': getattr(self.vector_brain, 'total_cycles', 0),
            'cycle_time_ms': processing_time,
            'architecture': 'parallel_sparse_goldilocks',
            'prediction_confidence': prediction_confidence,
            'parallel_processing': True,
            'processing_time_ms': processing_time,
            'sensory_result': sensory_result,
            'motor_result': motor_result, 
            'temporal_result': temporal_result,
            'cross_stream_bindings': parallel_results.get('cross_stream_bindings', []),
            'coordination_signals': parallel_results.get('coordination_signals', {}),
            'constraint_propagation': shared_stats.get('constraint_propagation', {}),
            'resource_usage': shared_stats.get('shared_resources', {}),
            'biological_timing': parallel_results.get('biological_timing', {})
        }
        
        parallel_stats = {
            'parallel_processing_time_ms': processing_time,
            'cycle_time_ms': parallel_results.get('cycle_time_ms', processing_time),
            'streams_processed': len([r for r in [sensory_result, motor_result, temporal_result] if r]),
            'cross_stream_bindings': len(parallel_results.get('cross_stream_bindings', [])),
            'coordination_signals': len(parallel_results.get('coordination_signals', {})),
            'constraint_propagation_stats': shared_stats.get('constraint_propagation', {})
        }
        
        coordination_success = (len(parallel_results.get('cross_stream_bindings', [])) > 0 or 
                              len(parallel_results.get('coordination_signals', {})) > 0)
        
        return ParallelProcessingResult(
            motor_output=motor_output,
            brain_state=brain_state, 
            parallel_stats=parallel_stats,
            processing_time_ms=processing_time,
            coordination_success=coordination_success
        )
    
    async def _process_sensory_stream_async(self, sensory_input: List[float], budget_ms: float) -> Dict[str, Any]:
        """Process sensory stream asynchronously."""
        await asyncio.sleep(0.002)  # 2ms processing simulation
        
        # Use existing sensory stream
        if hasattr(self.vector_brain, 'sensory_stream'):
            activation_info = self.vector_brain.sensory_stream.update(sensory_input, time.time())
            active_patterns = self.vector_brain.sensory_stream.get_active_pattern_indices(k=5)
            
            return {
                'stream_type': 'sensory',
                'active_patterns': active_patterns,
                'activation_strength': activation_info.get('activation_strength', 0.5),
                'processing_time_ms': 2.0
            }
        
        # Fallback for testing
        return {
            'stream_type': 'sensory',
            'active_patterns': [1, 2, 3],
            'activation_strength': 0.7,
            'processing_time_ms': 2.0
        }
    
    async def _process_motor_stream_async(self, sensory_input: List[float], budget_ms: float) -> Dict[str, Any]:
        """Process motor stream asynchronously."""
        await asyncio.sleep(0.003)  # 3ms processing simulation
        
        # Generate motor prediction based on sensory input
        motor_prediction = self._generate_motor_prediction(sensory_input)
        
        # Use existing motor stream
        if hasattr(self.vector_brain, 'motor_stream'):
            activation_info = self.vector_brain.motor_stream.update(motor_prediction, time.time())
            active_patterns = self.vector_brain.motor_stream.get_active_pattern_indices(k=5)
            
            return {
                'stream_type': 'motor',
                'active_patterns': active_patterns,
                'activation_strength': activation_info.get('activation_strength', 0.5),
                'motor_output': motor_prediction,
                'processing_time_ms': 3.0
            }
        
        # Fallback for testing
        return {
            'stream_type': 'motor',
            'active_patterns': [4, 5, 6],
            'activation_strength': 0.6,
            'motor_output': motor_prediction,
            'processing_time_ms': 3.0
        }
    
    async def _process_temporal_stream_async(self, current_time: float, budget_ms: float) -> Dict[str, Any]:
        """Process temporal stream asynchronously."""
        await asyncio.sleep(0.002)  # 2ms processing simulation
        
        # Generate temporal context
        temporal_vector = self._generate_temporal_context(current_time)
        
        # Use existing temporal stream
        if hasattr(self.vector_brain, 'temporal_stream'):
            activation_info = self.vector_brain.temporal_stream.update(temporal_vector, current_time)
            active_patterns = self.vector_brain.temporal_stream.get_active_pattern_indices(k=5)
            
            return {
                'stream_type': 'temporal',
                'active_patterns': active_patterns,
                'activation_strength': activation_info.get('activation_strength', 0.5),
                'temporal_context': temporal_vector,
                'processing_time_ms': 2.0
            }
        
        # Fallback for testing
        return {
            'stream_type': 'temporal',
            'active_patterns': [7, 8, 9],
            'activation_strength': 0.8,
            'temporal_context': temporal_vector,
            'processing_time_ms': 2.0
        }
    
    async def _create_cross_stream_bindings(self, sensory_result, motor_result, temporal_result) -> List[Dict[str, Any]]:
        """Create cross-stream bindings during gamma windows."""
        bindings = []
        
        # Check if we're in a binding window
        timing = self.biological_oscillator.get_current_timing()
        if not timing.binding_window_active:
            return bindings
        
        results = [r for r in [sensory_result, motor_result, temporal_result] if r is not None]
        
        # Create bindings between streams with strong activations
        for i, result_a in enumerate(results):
            for result_b in results[i+1:]:
                if (result_a['activation_strength'] > 0.5 and 
                    result_b['activation_strength'] > 0.5 and
                    result_a['active_patterns'] and result_b['active_patterns']):
                    
                    binding_strength = (result_a['activation_strength'] + result_b['activation_strength']) / 2
                    
                    bindings.append({
                        'source_stream': result_a['stream_type'],
                        'target_stream': result_b['stream_type'],
                        'binding_strength': binding_strength,
                        'source_patterns': result_a['active_patterns'][:3],
                        'target_patterns': result_b['active_patterns'][:3],
                        'gamma_cycle': self.biological_oscillator.cycle_count
                    })
        
        return bindings
    
    def _generate_motor_prediction(self, sensory_input: List[float]) -> List[float]:
        """Generate motor prediction from sensory input."""
        # Simple transformation for testing - use existing brain logic when available
        if hasattr(self.vector_brain, '_predict_motor_output'):
            # Use existing motor prediction logic
            return self.vector_brain._predict_motor_output(sensory_input, [])
        
        # Fallback: simple linear transformation
        motor_dim = getattr(self.vector_brain, 'motor_dim', 4)
        sensory_sum = sum(sensory_input[:motor_dim])
        return [sensory_sum * 0.1] * motor_dim
    
    def _generate_temporal_context(self, current_time: float) -> List[float]:
        """Generate temporal context vector."""
        if hasattr(self.vector_brain, '_generate_temporal_context'):
            return self.vector_brain._generate_temporal_context(current_time)
        
        # Fallback: simple time-based vector
        temporal_dim = getattr(self.vector_brain, 'temporal_dim', 4)
        time_features = [
            current_time % 1.0,  # Sub-second timing
            (current_time % 60.0) / 60.0,  # Minute cycle
            (current_time % 3600.0) / 3600.0,  # Hour cycle
            0.5  # Constant
        ]
        return time_features[:temporal_dim]
    
    def _synthesize_motor_output(self, sensory_result, motor_result, temporal_result) -> List[float]:
        """Synthesize final motor output from parallel stream results."""
        if motor_result and 'motor_output' in motor_result:
            return motor_result['motor_output']
        
        # Fallback: generate from sensory input
        motor_dim = getattr(self.vector_brain, 'motor_dim', 4)
        return [0.5] * motor_dim
    
    def _compile_parallel_brain_state(self, sensory_result, motor_result, temporal_result, 
                                    bindings, processing_time_ms) -> Dict[str, Any]:
        """Compile brain state from parallel processing results."""
        # Calculate prediction confidence from stream activations
        activation_strengths = [
            r['activation_strength'] for r in [sensory_result, motor_result, temporal_result] 
            if r is not None
        ]
        prediction_confidence = np.mean(activation_strengths) if activation_strengths else 0.5
        
        brain_state = {
            'parallel_processing': True,
            'prediction_confidence': prediction_confidence,
            'processing_time_ms': processing_time_ms,
            'streams_processed': {
                'sensory': sensory_result is not None,
                'motor': motor_result is not None,
                'temporal': temporal_result is not None
            },
            'cross_stream_bindings': len(bindings),
            'biological_timing': self.biological_oscillator.get_coordination_signal(),
            'parallel_coordination_success': len(bindings) > 0,
            'architecture': 'parallel_vector_stream'
        }
        
        # Add stream-specific information
        if sensory_result:
            brain_state['sensory_activation'] = sensory_result['activation_strength']
            brain_state['sensory_patterns'] = len(sensory_result['active_patterns'])
        
        if motor_result:
            brain_state['motor_activation'] = motor_result['activation_strength']
            brain_state['motor_patterns'] = len(motor_result['active_patterns'])
        
        if temporal_result:
            brain_state['temporal_activation'] = temporal_result['activation_strength']
            brain_state['temporal_patterns'] = len(temporal_result['active_patterns'])
        
        return brain_state
    
    def _process_sequential_mode(self, sensory_input: List[float]) -> Tuple[List[float], Dict[str, Any]]:
        """Process using original sequential brain architecture."""
        start_time = time.time()
        
        # Use existing vector brain processing
        motor_output, brain_state = self.vector_brain.process_sensory_input(sensory_input)
        
        processing_time = (time.time() - start_time) * 1000
        
        # Update performance tracking
        self.sequential_cycles += 1
        self.performance_comparison['sequential_times'].append(processing_time)
        self.performance_comparison['sequential_successes'] += 1
        
        # Add sequential processing markers
        brain_state.update({
            'parallel_processing': False,
            'sequential_processing_time_ms': processing_time
        })
        
        # Ensure prediction_confidence is always present
        if 'prediction_confidence' not in brain_state:
            brain_state['prediction_confidence'] = 0.5
        
        return motor_output, brain_state
    
    def get_performance_comparison(self) -> Dict[str, Any]:
        """Get performance comparison between parallel and sequential modes."""
        parallel_times = self.performance_comparison['parallel_times']
        sequential_times = self.performance_comparison['sequential_times']
        
        comparison = {
            'parallel_cycles': self.parallel_cycles,
            'sequential_cycles': self.sequential_cycles,
            'parallel_success_rate': (
                self.performance_comparison['parallel_successes'] / max(1, self.parallel_cycles)
            ),
            'sequential_success_rate': (
                self.performance_comparison['sequential_successes'] / max(1, self.sequential_cycles)
            )
        }
        
        if parallel_times:
            comparison.update({
                'parallel_avg_time_ms': sum(parallel_times) / len(parallel_times),
                'parallel_min_time_ms': min(parallel_times),
                'parallel_max_time_ms': max(parallel_times)
            })
        
        if sequential_times:
            comparison.update({
                'sequential_avg_time_ms': sum(sequential_times) / len(sequential_times),
                'sequential_min_time_ms': min(sequential_times),
                'sequential_max_time_ms': max(sequential_times)
            })
        
        # Calculate performance ratio
        if parallel_times and sequential_times:
            parallel_avg = sum(parallel_times) / len(parallel_times)
            sequential_avg = sum(sequential_times) / len(sequential_times)
            comparison['speedup_ratio'] = sequential_avg / parallel_avg if parallel_avg > 0 else 0
        
        return comparison
    
    def set_parallel_mode(self, enabled: bool):
        """Enable or disable parallel processing mode."""
        self.parallel_mode_enabled = enabled
        if not self.quiet_mode:
            mode = "parallel" if enabled else "sequential"
            print(f"🧠 Switched to {mode} processing mode")
    
    def cleanup(self):
        """Cleanup async resources."""
        if self.async_loop:
            self.async_loop.call_soon_threadsafe(self.async_loop.stop)
        if self.async_thread:
            self.async_thread.join(timeout=1.0)