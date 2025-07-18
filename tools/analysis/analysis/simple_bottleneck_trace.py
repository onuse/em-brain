#!/usr/bin/env python3
"""
Simple Bottleneck Trace - Identify what takes 2.2s in the slow path

This uses a simpler approach to trace the slow path bottleneck
by instrumenting the actual brain processing pipeline.
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

def trace_bottleneck():
    """Trace the slow path bottleneck by timing key operations."""
    print("🔍 SIMPLE BOTTLENECK TRACE")
    print("=" * 35)
    
    # Create brain
    try:
        with timeout(30):
            brain = MinimalBrain(quiet_mode=True)
        print("✅ Brain created successfully")
    except TimeoutError:
        print("❌ Brain creation timed out!")
        return
    
    # Test input that will force slow path
    novel_input = [1.0, 2.0, 3.0, 4.0]
    
    print("\nTracing slow path components...")
    
    # Time the key operations by calling them directly
    vector_brain = brain.vector_brain
    
    try:
        with timeout(60):
            # 1. Time tensor conversion
            start_time = time.time()
            sensory_tensor = torch.tensor(novel_input, dtype=torch.float32)
            tensor_time = (time.time() - start_time) * 1000
            print(f"  1. Tensor conversion: {tensor_time:.2f}ms")
            
            # 2. Time temporal context generation
            start_time = time.time()
            temporal_vector = vector_brain._generate_temporal_context(time.time())
            temporal_time = (time.time() - start_time) * 1000
            print(f"  2. Temporal context: {temporal_time:.2f}ms")
            
            # 3. Time the motor prediction (likely bottleneck)
            start_time = time.time()
            # Create fake sensory and temporal activations
            sensory_activation = sensory_tensor
            temporal_activation = temporal_vector
            
            # This is likely the bottleneck - the emergent temporal hierarchy
            motor_prediction = vector_brain._predict_motor_output_emergent(
                sensory_activation, temporal_activation, time.time()
            )
            motor_prediction_time = (time.time() - start_time) * 1000
            print(f"  3. Motor prediction (emergent): {motor_prediction_time:.1f}ms")
            
            # 4. Time competitive dynamics
            start_time = time.time()
            # Create combined pattern
            combined_pattern = vector_brain._create_combined_pattern(
                sensory_activation, motor_prediction, temporal_activation
            )
            
            # Encode as sparse
            from server.src.vector_stream.sparse_representations import SparsePatternEncoder
            encoder = SparsePatternEncoder(
                vector_brain.emergent_competition.unified_storage.pattern_dim, 
                sparsity=0.02, 
                quiet_mode=True
            )
            sparse_combined = encoder.encode_top_k(combined_pattern, f"test")
            
            # Process through competition
            competitive_result = vector_brain.emergent_competition.process_with_competition(
                sparse_combined, time.time()
            )
            competitive_time = (time.time() - start_time) * 1000
            print(f"  4. Competitive dynamics: {competitive_time:.1f}ms")
            
            # 5. Time brain state compilation
            start_time = time.time()
            brain_statistics = vector_brain.get_brain_statistics()
            brain_state = brain._compile_brain_state(brain_statistics)
            brain_state_time = (time.time() - start_time) * 1000
            print(f"  5. Brain state compilation: {brain_state_time:.1f}ms")
            
    except TimeoutError:
        print("❌ Bottleneck tracing timed out!")
        return
    except Exception as e:
        print(f"❌ Bottleneck tracing failed: {e}")
        return
    
    # Calculate totals
    total_traced = tensor_time + temporal_time + motor_prediction_time + competitive_time + brain_state_time
    
    print(f"\n📊 BOTTLENECK ANALYSIS")
    print("=" * 30)
    print(f"Total traced time: {total_traced:.1f}ms")
    
    # Identify the bottleneck
    components = [
        ("Tensor conversion", tensor_time),
        ("Temporal context", temporal_time),
        ("Motor prediction", motor_prediction_time),
        ("Competitive dynamics", competitive_time),
        ("Brain state compilation", brain_state_time)
    ]
    
    # Sort by time
    components.sort(key=lambda x: x[1], reverse=True)
    
    print("\nComponents by time:")
    for name, elapsed in components:
        percentage = (elapsed / total_traced) * 100 if total_traced > 0 else 0
        print(f"  {name}: {elapsed:.1f}ms ({percentage:.1f}%)")
    
    # Identify primary bottleneck
    if components:
        bottleneck_name, bottleneck_time = components[0]
        print(f"\n🚨 PRIMARY BOTTLENECK:")
        print(f"   {bottleneck_name}: {bottleneck_time:.1f}ms")
        
        if bottleneck_time > 1000:
            print(f"   This is the main cause of the 2.2s delay!")
        
        # Show what fast path avoids
        print(f"\n⚡ FAST PATH ADVANTAGE:")
        print(f"   Fast path skips: {bottleneck_name}")
        print(f"   Time saved: {bottleneck_time:.1f}ms")
        print(f"   Speedup potential: {bottleneck_time/2:.0f}x")
    
    # Test actual full prediction for comparison
    print(f"\n🔄 FULL PREDICTION TEST")
    print("=" * 30)
    
    try:
        with timeout(30):
            start_time = time.time()
            action, brain_state = brain.process_sensory_input(novel_input, action_dimensions=2)
            full_prediction_time = (time.time() - start_time) * 1000
            
            fast_path_used = brain_state.get('fast_path_used', False)
            
            print(f"Full prediction time: {full_prediction_time:.1f}ms")
            print(f"Fast path used: {fast_path_used}")
            
            if not fast_path_used:
                print(f"Difference from traced: {abs(full_prediction_time - total_traced):.1f}ms")
    except TimeoutError:
        print("❌ Full prediction test timed out!")
    
    return components

if __name__ == "__main__":
    print("Tracing bottleneck in 2.2s slow path...")
    components = trace_bottleneck()
    
    if components:
        print(f"\n🏁 BOTTLENECK IDENTIFIED")
        print(f"Primary culprit: {components[0][0]} ({components[0][1]:.1f}ms)")
        print(f"This explains the 2.2s delay!")
    else:
        print(f"\n❌ BOTTLENECK TRACE FAILED")