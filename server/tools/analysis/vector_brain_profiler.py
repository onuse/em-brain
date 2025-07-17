#!/usr/bin/env python3
"""
Vector Brain Profiler - Find exactly what inside vector_brain.process_sensory_input takes 2.05s

This profiler traces each component inside the vector brain processing
to identify the specific bottleneck within the 2,050.9ms.
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

class VectorBrainProfiler:
    """Profile components inside vector brain processing."""
    
    def __init__(self):
        self.component_times = {}
    
    def time_component(self, name, operation):
        """Time a specific component."""
        start_time = time.time()
        result = operation()
        elapsed = (time.time() - start_time) * 1000
        
        self.component_times[name] = elapsed
        
        if elapsed > 500:
            print(f"  🚨 VERY SLOW: {name}: {elapsed:.1f}ms")
        elif elapsed > 100:
            print(f"  🐌 SLOW: {name}: {elapsed:.1f}ms")
        elif elapsed > 10:
            print(f"  ⚠️  {name}: {elapsed:.1f}ms")
        else:
            print(f"  ✅ {name}: {elapsed:.2f}ms")
        
        return result
    
    def get_report(self):
        """Generate component timing report."""
        if not self.component_times:
            return "No timing data collected"
        
        sorted_components = sorted(self.component_times.items(), key=lambda x: x[1], reverse=True)
        
        report = "\n📊 VECTOR BRAIN COMPONENT BREAKDOWN\n"
        report += "=" * 50 + "\n"
        
        total_time = sum(self.component_times.values())
        
        for name, elapsed in sorted_components:
            percentage = (elapsed / total_time) * 100 if total_time > 0 else 0
            report += f"  {name}: {elapsed:.1f}ms ({percentage:.1f}%)\n"
        
        report += f"\nTotal component time: {total_time:.1f}ms\n"
        
        if sorted_components:
            bottleneck_name, bottleneck_time = sorted_components[0]
            report += f"\n🚨 INTERNAL BOTTLENECK: {bottleneck_name} ({bottleneck_time:.1f}ms)\n"
        
        return report

def profile_vector_brain_internals():
    """Profile the internal components of vector brain processing."""
    print("🔍 VECTOR BRAIN INTERNAL PROFILER")
    print("=" * 45)
    
    # Create brain
    try:
        with timeout(30):
            brain = MinimalBrain(quiet_mode=True)
        print("✅ Brain created successfully")
    except TimeoutError:
        print("❌ Brain creation timed out!")
        return
    
    profiler = VectorBrainProfiler()
    novel_input = [1.0, 2.0, 3.0, 4.0]
    
    print("\nProfiling vector brain internal components...")
    
    try:
        with timeout(120):  # 2 minute timeout since we know this is slow
            vector_brain = brain.vector_brain
            
            # Manually step through the vector brain processing
            # Based on the process_sensory_input method in sparse_goldilocks_brain.py
            
            current_time = time.time()
            
            print("\n1. Tensor conversion...")
            
            def tensor_conversion():
                return torch.tensor(novel_input, dtype=torch.float32)
            
            sensory_tensor = profiler.time_component("tensor_conversion", tensor_conversion)
            
            print("\n2. Fast path check...")
            
            def fast_path_check():
                return vector_brain._try_fast_reflex_path(sensory_tensor, current_time)
            
            fast_path_result = profiler.time_component("fast_path_check", fast_path_check)
            
            if fast_path_result is not None:
                print("⚡ Fast path was used - not the bottleneck we're looking for")
                print("Let's force slow path by clearing the cache...")
                # Clear the reflex cache to force slow path
                vector_brain.emergent_hierarchy.predictor.reflex_cache.clear()
            
            print("\n3. Temporal context generation...")
            
            def temporal_context():
                return vector_brain._generate_temporal_context(current_time)
            
            temporal_vector = profiler.time_component("temporal_context_generation", temporal_context)
            
            print("\n4. Parallel stream processing...")
            
            def stream_processing():
                # Time the parallel stream updates
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                    sensory_future = executor.submit(vector_brain.sensory_stream.update, sensory_tensor, current_time)
                    temporal_future = executor.submit(vector_brain.temporal_stream.update, temporal_vector, current_time)
                    
                    sensory_activation = sensory_future.result()
                    temporal_activation = temporal_future.result()
                    
                return sensory_activation, temporal_activation
            
            sensory_activation, temporal_activation = profiler.time_component("parallel_stream_processing", stream_processing)
            
            print("\n5. Motor prediction (emergent temporal constraints)...")
            
            def motor_prediction():
                return vector_brain._predict_motor_output_emergent(
                    sensory_activation, temporal_activation, current_time
                )
            
            motor_prediction = profiler.time_component("motor_prediction_emergent", motor_prediction)
            
            print("\n6. Motor stream update...")
            
            def motor_stream_update():
                return vector_brain.motor_stream.update(motor_prediction, current_time)
            
            motor_activation = profiler.time_component("motor_stream_update", motor_stream_update)
            
            print("\n7. Combined pattern creation...")
            
            def combined_pattern():
                return vector_brain._create_combined_pattern(
                    sensory_activation, motor_activation, temporal_activation
                )
            
            combined_pattern = profiler.time_component("combined_pattern_creation", combined_pattern)
            
            print("\n8. Sparse encoding...")
            
            def sparse_encoding():
                from server.src.vector_stream.sparse_representations import SparsePatternEncoder
                encoder = SparsePatternEncoder(
                    vector_brain.emergent_competition.unified_storage.pattern_dim, 
                    sparsity=0.02, 
                    quiet_mode=True
                )
                return encoder.encode_top_k(combined_pattern, f"competition_{vector_brain.total_cycles}")
            
            sparse_combined = profiler.time_component("sparse_encoding", sparse_encoding)
            
            print("\n9. Competitive dynamics processing...")
            
            def competitive_dynamics():
                return vector_brain.emergent_competition.process_with_competition(
                    sparse_combined, current_time
                )
            
            competitive_result = profiler.time_component("competitive_dynamics_processing", competitive_dynamics)
            
            print("\n10. Cross-stream coactivation...")
            
            def coactivation_tracking():
                active_pattern_ids = {
                    'sensory': vector_brain.sensory_stream.get_active_pattern_ids(k=5),
                    'motor': vector_brain.motor_stream.get_active_pattern_ids(k=5),
                    'temporal': vector_brain.temporal_stream.get_active_pattern_ids(k=5)
                }
                
                active_indices = {}
                for stream_name, pattern_ids in active_pattern_ids.items():
                    indices = [hash(pid) % 10000 for pid in pattern_ids]
                    active_indices[stream_name] = indices
                
                vector_brain.coactivation.record_coactivation(active_indices)
                return True
            
            profiler.time_component("coactivation_tracking", coactivation_tracking)
            
            print("\n11. Brain state compilation...")
            
            def brain_state_compilation():
                return vector_brain.get_brain_statistics()
            
            brain_statistics = profiler.time_component("brain_state_compilation", brain_state_compilation)
            
    except TimeoutError:
        print("❌ Vector brain internal profiling timed out!")
        return
    except Exception as e:
        print(f"❌ Vector brain internal profiling failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Print detailed report
    print(profiler.get_report())
    
    return profiler

if __name__ == "__main__":
    print("Profiling vector brain internals to find the 2.05s bottleneck...")
    profiler = profile_vector_brain_internals()
    
    if profiler:
        print(f"\n🏁 VECTOR BRAIN PROFILING COMPLETE")
        print(f"The exact internal bottleneck has been identified!")
    else:
        print(f"\n❌ VECTOR BRAIN PROFILING FAILED")