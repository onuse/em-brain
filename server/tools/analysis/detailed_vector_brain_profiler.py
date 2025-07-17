#!/usr/bin/env python3
"""
Detailed Vector Brain Profiler - Break down vector_brain.process_sensory_input timing

This profiler instruments the actual vector brain code to time each component
and identify exactly what takes the 2+ seconds.
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

# Monkey patch the vector brain to add timing
class VectorBrainTimer:
    """Timer to instrument vector brain processing."""
    
    def __init__(self):
        self.timings = {}
        self.start_time = None
    
    def start_timer(self, name):
        """Start timing a component."""
        self.start_time = time.time()
        print(f"    Starting {name}...")
    
    def end_timer(self, name):
        """End timing a component."""
        if self.start_time:
            elapsed = (time.time() - self.start_time) * 1000
            self.timings[name] = elapsed
            
            if elapsed > 1000:
                print(f"    🚨 {name}: {elapsed:.1f}ms (VERY SLOW)")
            elif elapsed > 100:
                print(f"    🐌 {name}: {elapsed:.1f}ms (SLOW)")
            elif elapsed > 10:
                print(f"    ⚠️  {name}: {elapsed:.1f}ms")
            else:
                print(f"    ✅ {name}: {elapsed:.2f}ms")
            
            self.start_time = None
    
    def get_breakdown(self):
        """Get timing breakdown."""
        if not self.timings:
            return "No timing data collected"
        
        sorted_timings = sorted(self.timings.items(), key=lambda x: x[1], reverse=True)
        
        report = "\n📊 DETAILED VECTOR BRAIN BREAKDOWN\n"
        report += "=" * 50 + "\n"
        
        total_time = sum(self.timings.values())
        
        for name, elapsed in sorted_timings:
            percentage = (elapsed / total_time) * 100 if total_time > 0 else 0
            report += f"  {name}: {elapsed:.1f}ms ({percentage:.1f}%)\n"
        
        report += f"\nTotal measured time: {total_time:.1f}ms\n"
        
        if sorted_timings:
            bottleneck_name, bottleneck_time = sorted_timings[0]
            report += f"\n🚨 SLOWEST COMPONENT: {bottleneck_name} ({bottleneck_time:.1f}ms)\n"
        
        return report

def profile_vector_brain_detailed():
    """Profile vector brain with detailed component timing."""
    print("🔍 DETAILED VECTOR BRAIN PROFILER")
    print("=" * 45)
    
    # Create brain
    try:
        with timeout(30):
            brain = MinimalBrain(quiet_mode=True)
        print("✅ Brain created successfully")
    except TimeoutError:
        print("❌ Brain creation timed out!")
        return
    
    # Clear cache to force slow path
    brain.vector_brain.emergent_hierarchy.predictor.reflex_cache.clear()
    brain.vector_brain.emergent_hierarchy.predictor.reflex_cache_hits = 0
    brain.vector_brain.emergent_hierarchy.predictor.reflex_cache_misses = 0
    
    novel_input = [1.0, 2.0, 3.0, 4.0]
    timer = VectorBrainTimer()
    
    print(f"\nProfiling vector brain components (input: {novel_input})...")
    
    # Apply MinimalBrain dimension adaptation logic
    sensory_dim = brain.sensory_dim
    if len(novel_input) > sensory_dim:
        processed_input = novel_input[:sensory_dim]
    elif len(novel_input) < sensory_dim:
        processed_input = novel_input + [0.0] * (sensory_dim - len(novel_input))
    else:
        processed_input = novel_input
    
    print(f"Adapted input: {processed_input} (length: {len(processed_input)})")
    
    try:
        with timeout(180):  # 3 minute timeout
            vector_brain = brain.vector_brain
            
            # Manually step through process_sensory_input with timing
            cycle_start = time.time()
            current_time = cycle_start
            
            print("\n🔄 VECTOR BRAIN COMPONENT TIMING:")
            
            # 1. Convert to tensors
            timer.start_timer("tensor_conversion")
            sensory_tensor = torch.tensor(processed_input, dtype=torch.float32)
            timer.end_timer("tensor_conversion")
            
            # 2. Fast path check (we know this will fail first time)
            timer.start_timer("fast_path_check")
            fast_path_result = vector_brain._try_fast_reflex_path(sensory_tensor, current_time)
            timer.end_timer("fast_path_check")
            
            if fast_path_result is not None:
                print("⚡ Fast path succeeded (unexpected!)")
                return timer.get_breakdown()
            
            # 3. Generate temporal context
            timer.start_timer("temporal_context_generation")
            temporal_vector = vector_brain._generate_temporal_context(current_time)
            timer.end_timer("temporal_context_generation")
            
            # 4. Stream processing - let's break this down further
            print("\n  📡 STREAM PROCESSING BREAKDOWN:")
            
            # 4a. Sensory stream update
            timer.start_timer("sensory_stream_update")
            try:
                sensory_activation = vector_brain.sensory_stream.update(sensory_tensor, current_time)
                timer.end_timer("sensory_stream_update")
            except Exception as e:
                timer.end_timer("sensory_stream_update")
                print(f"    ❌ Sensory stream failed: {e}")
                # Create dummy activation
                sensory_activation = sensory_tensor
            
            # 4b. Temporal stream update
            timer.start_timer("temporal_stream_update")
            try:
                temporal_activation = vector_brain.temporal_stream.update(temporal_vector, current_time)
                timer.end_timer("temporal_stream_update")
            except Exception as e:
                timer.end_timer("temporal_stream_update")
                print(f"    ❌ Temporal stream failed: {e}")
                # Create dummy activation
                temporal_activation = temporal_vector
            
            # 5. Motor prediction (this is likely the bottleneck)
            print("\n  🧠 MOTOR PREDICTION BREAKDOWN:")
            
            timer.start_timer("motor_prediction_emergent")
            try:
                motor_prediction = vector_brain._predict_motor_output_emergent(
                    sensory_activation, temporal_activation, current_time
                )
                timer.end_timer("motor_prediction_emergent")
            except Exception as e:
                timer.end_timer("motor_prediction_emergent")
                print(f"    ❌ Motor prediction failed: {e}")
                # Create dummy prediction
                motor_prediction = torch.zeros(vector_brain.motor_config.dim)
            
            # 6. Motor stream update
            timer.start_timer("motor_stream_update")
            try:
                motor_activation = vector_brain.motor_stream.update(motor_prediction, current_time)
                timer.end_timer("motor_stream_update")
            except Exception as e:
                timer.end_timer("motor_stream_update")
                print(f"    ❌ Motor stream failed: {e}")
                motor_activation = motor_prediction
            
            # 7. Combined pattern creation
            timer.start_timer("combined_pattern_creation")
            try:
                combined_pattern = vector_brain._create_combined_pattern(
                    sensory_activation, motor_activation, temporal_activation
                )
                timer.end_timer("combined_pattern_creation")
            except Exception as e:
                timer.end_timer("combined_pattern_creation")
                print(f"    ❌ Combined pattern failed: {e}")
                combined_pattern = torch.cat([sensory_activation, motor_activation, temporal_activation])
            
            # 8. Sparse encoding
            timer.start_timer("sparse_encoding")
            try:
                from server.src.vector_stream.sparse_representations import SparsePatternEncoder
                encoder = SparsePatternEncoder(
                    vector_brain.emergent_competition.unified_storage.pattern_dim, 
                    sparsity=0.02, 
                    quiet_mode=True
                )
                sparse_combined = encoder.encode_top_k(
                    combined_pattern, 
                    f"competition_{vector_brain.total_cycles}"
                )
                timer.end_timer("sparse_encoding")
            except Exception as e:
                timer.end_timer("sparse_encoding")
                print(f"    ❌ Sparse encoding failed: {e}")
                sparse_combined = None
            
            # 9. Competitive dynamics
            timer.start_timer("competitive_dynamics")
            try:
                if sparse_combined:
                    competitive_result = vector_brain.emergent_competition.process_with_competition(
                        sparse_combined, current_time
                    )
                timer.end_timer("competitive_dynamics")
            except Exception as e:
                timer.end_timer("competitive_dynamics")
                print(f"    ❌ Competitive dynamics failed: {e}")
            
            # 10. Cross-stream coactivation
            timer.start_timer("cross_stream_coactivation")
            try:
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
                timer.end_timer("cross_stream_coactivation")
            except Exception as e:
                timer.end_timer("cross_stream_coactivation")
                print(f"    ❌ Cross-stream coactivation failed: {e}")
            
            # 11. Brain state compilation
            timer.start_timer("brain_state_compilation")
            try:
                brain_statistics = vector_brain.get_brain_statistics()
                timer.end_timer("brain_state_compilation")
            except Exception as e:
                timer.end_timer("brain_state_compilation")
                print(f"    ❌ Brain state compilation failed: {e}")
            
            total_manual_time = (time.time() - cycle_start) * 1000
            print(f"\nTotal manual timing: {total_manual_time:.1f}ms")
            
    except TimeoutError:
        print("❌ Detailed profiling timed out!")
        return timer.get_breakdown()
    except Exception as e:
        print(f"❌ Detailed profiling failed: {e}")
        import traceback
        traceback.print_exc()
        return timer.get_breakdown()
    
    # Generate and print the detailed breakdown
    breakdown = timer.get_breakdown()
    print(breakdown)
    
    # Test a comparison with actual vector brain call
    print(f"\n🔄 COMPARISON WITH ACTUAL CALL")
    print("=" * 40)
    
    try:
        with timeout(60):
            # Clear cache again
            brain.vector_brain.emergent_hierarchy.predictor.reflex_cache.clear()
            
            start_time = time.time()
            predicted_action, vector_brain_state = brain.vector_brain.process_sensory_input(processed_input)
            actual_time = (time.time() - start_time) * 1000
            
            print(f"Actual vector brain call: {actual_time:.1f}ms")
            print(f"Manual component sum: {sum(timer.timings.values()):.1f}ms")
            print(f"Difference: {abs(actual_time - sum(timer.timings.values())):.1f}ms")
            
    except Exception as e:
        print(f"Actual call failed: {e}")
    
    return timer

if __name__ == "__main__":
    print("Profiling vector brain components in detail...")
    timer = profile_vector_brain_detailed()
    
    if timer:
        print(f"\n🏁 DETAILED PROFILING COMPLETE")
        print(f"Use the breakdown above to identify bottlenecks!")
    else:
        print(f"\n❌ DETAILED PROFILING FAILED")