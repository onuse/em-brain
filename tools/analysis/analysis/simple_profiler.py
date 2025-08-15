#!/usr/bin/env python3
"""
Simple profiler to identify the exact bottleneck in brain processing.
"""

import time
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'server'))

from src.brain import MinimalBrain

def profile_brain_processing():
    """Profile brain processing to find bottlenecks."""
    print("🔍 PROFILING BRAIN PROCESSING")
    print("=" * 40)
    
    # Create brain
    print("Creating brain...")
    brain = MinimalBrain(quiet_mode=True)
    
    # Single prediction timing
    print("\nTiming single prediction...")
    sensory_input = [1.0, 2.0, 3.0, 4.0]
    
    # Warm up
    brain.process_sensory_input(sensory_input, action_dimensions=2)
    
    # Profile multiple predictions
    num_predictions = 10
    total_start = time.time()
    
    for i in range(num_predictions):
        start = time.time()
        action, brain_state = brain.process_sensory_input(sensory_input, action_dimensions=2)
        end = time.time()
        
        prediction_time = (end - start) * 1000
        print(f"  Prediction {i+1}: {prediction_time:.1f}ms")
    
    total_end = time.time()
    avg_time = ((total_end - total_start) / num_predictions) * 1000
    
    print(f"\nAverage prediction time: {avg_time:.1f}ms")
    
    # Now let's manually trace the key operations
    print("\n🔍 MANUAL OPERATION TRACING")
    print("-" * 30)
    
    # Trace individual operations
    sensory_input = [1.0, 2.0, 3.0, 4.0]
    
    # Time tensor conversion
    start = time.time()
    processed_input = brain._preprocess_sensory_input(sensory_input)
    tensor_time = (time.time() - start) * 1000
    print(f"Tensor conversion: {tensor_time:.2f}ms")
    
    # Time vector brain processing
    start = time.time()
    predicted_action, vector_brain_state = brain.vector_brain.process_sensory_input(processed_input)
    vector_time = (time.time() - start) * 1000
    print(f"Vector brain processing: {vector_time:.1f}ms")
    
    # Time brain state compilation
    start = time.time()
    brain_state = brain._compile_brain_state(vector_brain_state)
    compile_time = (time.time() - start) * 1000
    print(f"Brain state compilation: {compile_time:.2f}ms")
    
    print(f"\nTotal accounted for: {tensor_time + vector_time + compile_time:.1f}ms")
    print(f"Vector brain is {vector_time/avg_time*100:.1f}% of total time")
    
    return avg_time, vector_time

if __name__ == "__main__":
    avg_time, vector_time = profile_brain_processing()
    
    print(f"\n📊 BOTTLENECK ANALYSIS")
    print("=" * 25)
    print(f"Average total time: {avg_time:.1f}ms")
    print(f"Vector brain time: {vector_time:.1f}ms")
    print(f"Vector brain percentage: {vector_time/avg_time*100:.1f}%")
    
    if vector_time > avg_time * 0.8:
        print("\n🚨 BOTTLENECK IDENTIFIED: Vector brain processing")
        print("   Focus optimization efforts on sparse_goldilocks_brain.py")
    else:
        print("\n✅ Bottleneck is elsewhere in the pipeline")