#!/usr/bin/env python3
"""
Slow Path Profiler - Identify exactly what takes 2.2 seconds in the slow path

This profiler traces each component of the slow path to identify
the specific bottlenecks that the fast path bypasses.
"""

import time
import sys
import os
import signal
from contextlib import contextmanager
sys.path.append(os.path.join(os.path.dirname(__file__), 'server'))

from src.brain import MinimalBrain
import torch

@contextmanager
def timeout(seconds):
    """Context manager for timing out operations."""
    def timeout_handler(signum, frame):
        raise TimeoutError(f"Operation timed out after {seconds} seconds")
    
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)
    
    try:
        yield
    finally:
        signal.alarm(0)

class SlowPathProfiler:
    """Profile the slow path to identify bottlenecks."""
    
    def __init__(self):
        self.timings = {}
        self.total_time = 0
    
    def time_operation(self, name, operation):
        """Time a specific operation."""
        start_time = time.time()
        result = operation()
        elapsed = (time.time() - start_time) * 1000
        self.timings[name] = elapsed
        return result
    
    def get_report(self):
        """Generate timing report."""
        report = "🔍 SLOW PATH TIMING BREAKDOWN\n"
        report += "=" * 40 + "\n"
        
        # Sort by time descending
        sorted_timings = sorted(self.timings.items(), key=lambda x: x[1], reverse=True)
        
        total_accounted = sum(self.timings.values())
        
        for name, elapsed in sorted_timings:
            percentage = (elapsed / total_accounted) * 100 if total_accounted > 0 else 0
            report += f"  {name}: {elapsed:.1f}ms ({percentage:.1f}%)\n"
        
        report += f"\nTotal accounted: {total_accounted:.1f}ms\n"
        
        return report

def profile_slow_path():
    """Profile the slow path to identify bottlenecks."""
    print("🔍 SLOW PATH PROFILER")
    print("=" * 35)
    
    # Create brain
    try:
        with timeout(30):
            brain = MinimalBrain(quiet_mode=True)
        print("✅ Brain created successfully")
    except TimeoutError:
        print("❌ Brain creation timed out!")
        return
    
    # Force first prediction to go through slow path
    # (it should be slow because cache is empty)
    novel_input = [1.0, 2.0, 3.0, 4.0]
    
    print("\nProfiling slow path prediction...")
    
    # Get direct access to the vector brain for detailed profiling
    vector_brain = brain.vector_brain
    
    profiler = SlowPathProfiler()
    
    try:
        with timeout(60):
            total_start = time.time()
            
            # Time the full prediction
            def full_prediction():
                return brain.process_sensory_input(novel_input, action_dimensions=2)
            
            action, brain_state = profiler.time_operation("TOTAL_PREDICTION", full_prediction)
            
            total_elapsed = (time.time() - total_start) * 1000
            
            print(f"Total prediction time: {total_elapsed:.1f}ms")
            print(f"Fast path used: {brain_state.get('fast_path_used', False)}")
            
    except TimeoutError:
        print("❌ Slow path profiling timed out!")
        return
    
    # Now let's manually trace the components to see what's slow
    print("\n🔍 MANUAL COMPONENT TRACING")
    print("=" * 40)
    
    # Create a fresh brain for component tracing
    try:
        with timeout(30):
            trace_brain = MinimalBrain(quiet_mode=True)
    except TimeoutError:
        print("❌ Trace brain creation timed out!")
        return
    
    # Manually trace each component
    component_times = {}
    
    try:
        with timeout(60):
            # Time tensor conversion
            start_time = time.time()
            sensory_tensor = torch.tensor(novel_input, dtype=torch.float32)
            component_times['tensor_conversion'] = (time.time() - start_time) * 1000
            
            # Time temporal context generation
            start_time = time.time()
            temporal_vector = trace_brain.vector_brain._generate_temporal_context(time.time())
            component_times['temporal_context'] = (time.time() - start_time) * 1000
            
            # Time sensory stream update
            start_time = time.time()
            sensory_activation = trace_brain.vector_brain.sensory_stream.update(sensory_tensor, time.time())
            component_times['sensory_stream_update'] = (time.time() - start_time) * 1000
            
            # Time temporal stream update
            start_time = time.time()
            temporal_activation = trace_brain.vector_brain.temporal_stream.update(temporal_vector, time.time())
            component_times['temporal_stream_update'] = (time.time() - start_time) * 1000
            
            # Time motor prediction (this is likely the bottleneck)
            start_time = time.time()
            motor_prediction = trace_brain.vector_brain._predict_motor_output_emergent(
                sensory_activation, temporal_activation, time.time()
            )
            component_times['motor_prediction_emergent'] = (time.time() - start_time) * 1000
            
            # Time motor stream update
            start_time = time.time()
            motor_activation = trace_brain.vector_brain.motor_stream.update(motor_prediction, time.time())
            component_times['motor_stream_update'] = (time.time() - start_time) * 1000
            
            # Time combined pattern creation
            start_time = time.time()
            combined_pattern = trace_brain.vector_brain._create_combined_pattern(
                sensory_activation, motor_activation, temporal_activation
            )
            component_times['combined_pattern_creation'] = (time.time() - start_time) * 1000
            
            # Time sparse encoding
            start_time = time.time()
            from server.src.vector_stream.sparse_representations import SparsePatternEncoder
            encoder = SparsePatternEncoder(
                trace_brain.vector_brain.emergent_competition.unified_storage.pattern_dim, 
                sparsity=0.02, 
                quiet_mode=True
            )
            sparse_combined = encoder.encode_top_k(combined_pattern, f"competition_test")
            component_times['sparse_encoding'] = (time.time() - start_time) * 1000
            
            # Time competitive dynamics
            start_time = time.time()
            competitive_result = trace_brain.vector_brain.emergent_competition.process_with_competition(
                sparse_combined, time.time()
            )
            component_times['competitive_dynamics'] = (time.time() - start_time) * 1000
            
            # Time brain state compilation
            start_time = time.time()
            brain_state = trace_brain._compile_brain_state(
                trace_brain.vector_brain.get_brain_statistics()
            )
            component_times['brain_state_compilation'] = (time.time() - start_time) * 1000
            
    except TimeoutError:
        print("❌ Component tracing timed out!")
        return
    except Exception as e:
        print(f"❌ Component tracing failed: {e}")
        return
    
    # Display component breakdown
    print("\n📊 COMPONENT TIMING BREAKDOWN")
    print("=" * 40)
    
    # Sort by time descending
    sorted_components = sorted(component_times.items(), key=lambda x: x[1], reverse=True)
    total_component_time = sum(component_times.values())
    
    for name, elapsed in sorted_components:
        percentage = (elapsed / total_component_time) * 100 if total_component_time > 0 else 0
        print(f"  {name}: {elapsed:.1f}ms ({percentage:.1f}%)")
    
    print(f"\nTotal component time: {total_component_time:.1f}ms")
    
    # Identify the bottleneck
    if sorted_components:
        bottleneck_name, bottleneck_time = sorted_components[0]
        print(f"\n🚨 PRIMARY BOTTLENECK IDENTIFIED:")
        print(f"   {bottleneck_name}: {bottleneck_time:.1f}ms")
        print(f"   This is {bottleneck_time/total_component_time*100:.1f}% of slow path time")
    
    # Show what fast path bypasses
    print(f"\n⚡ FAST PATH BYPASSES:")
    bypassed_components = [
        'sensory_stream_update', 'temporal_stream_update', 'motor_prediction_emergent',
        'motor_stream_update', 'combined_pattern_creation', 'sparse_encoding',
        'competitive_dynamics', 'brain_state_compilation'
    ]
    
    bypassed_time = sum(component_times.get(comp, 0) for comp in bypassed_components)
    print(f"   Total bypassed time: {bypassed_time:.1f}ms")
    print(f"   Fast path keeps only: tensor_conversion, temporal_context, cache_lookup")
    
    return sorted_components

if __name__ == "__main__":
    print("Profiling slow path to identify 2.2s bottleneck...")
    components = profile_slow_path()
    
    if components:
        print(f"\n🏁 PROFILING COMPLETE")
        print(f"Primary bottleneck: {components[0][0]} ({components[0][1]:.1f}ms)")
    else:
        print(f"\n❌ PROFILING FAILED")