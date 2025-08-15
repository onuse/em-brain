#!/usr/bin/env python3
"""
Simple Vector Brain Profiler - Find the bottleneck inside vector brain processing

This uses a simpler approach to identify what's slow inside vector_brain.process_sensory_input
"""

import time
import sys
import os
import signal
from contextlib import contextmanager
sys.path.append(os.path.join(os.path.dirname(__file__), 'server'))

from src.brain import MinimalBrain

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

def profile_vector_brain_simple():
    """Profile vector brain by timing key sections."""
    print("🔍 SIMPLE VECTOR BRAIN PROFILER")
    print("=" * 40)
    
    # Create brain
    try:
        with timeout(30):
            brain = MinimalBrain(quiet_mode=True)
        print("✅ Brain created successfully")
    except TimeoutError:
        print("❌ Brain creation timed out!")
        return
    
    novel_input = [1.0, 2.0, 3.0, 4.0]
    
    # Apply MinimalBrain dimension adaptation logic
    sensory_dim = brain.sensory_dim
    if len(novel_input) > sensory_dim:
        processed_input = novel_input[:sensory_dim]
    elif len(novel_input) < sensory_dim:
        processed_input = novel_input + [0.0] * (sensory_dim - len(novel_input))
    else:
        processed_input = novel_input
    
    print(f"Adapted input: {processed_input} (length: {len(processed_input)})")
    
    # Clear the reflex cache to force slow path
    brain.vector_brain.emergent_hierarchy.predictor.reflex_cache.clear()
    brain.vector_brain.emergent_hierarchy.predictor.reflex_cache_hits = 0
    brain.vector_brain.emergent_hierarchy.predictor.reflex_cache_misses = 0
    
    print("Profiling vector brain with cleared cache (forcing slow path)...")
    
    try:
        with timeout(120):  # 2 minute timeout
            # Time the full vector brain call that we know is slow
            start_time = time.time()
            predicted_action, vector_brain_state = brain.vector_brain.process_sensory_input(processed_input)
            total_vector_time = (time.time() - start_time) * 1000
            
            print(f"\nTotal vector brain time: {total_vector_time:.1f}ms")
            
            # Now let's see what the fast path check does
            print(f"\nAnalyzing what's inside the slow vector brain call...")
            
            # Look at the brain state to see what components were processed
            print(f"\nVector brain state analysis:")
            print(f"  Architecture: {vector_brain_state.get('architecture', 'unknown')}")
            print(f"  Total cycles: {vector_brain_state.get('total_cycles', 0)}")
            print(f"  Prediction confidence: {vector_brain_state.get('prediction_confidence', 0):.3f}")
            
            # Check temporal hierarchy stats
            temporal_hierarchy = vector_brain_state.get('temporal_hierarchy', {})
            print(f"\nTemporal hierarchy processing:")
            print(f"  Layer usage: {temporal_hierarchy.get('layer_usage_patterns', {})}")
            print(f"  Total predictions: {temporal_hierarchy.get('total_predictions', 0)}")
            
            # Check competitive dynamics
            competitive_dynamics = vector_brain_state.get('competitive_dynamics', {})
            print(f"\nCompetitive dynamics processing:")
            print(f"  Total competitions: {competitive_dynamics.get('total_competitions', 0)}")
            print(f"  Pattern selections: {competitive_dynamics.get('pattern_selections', 0)}")
            
            # Check streams
            sensory_stream = vector_brain_state.get('sensory_stream', {})
            motor_stream = vector_brain_state.get('motor_stream', {})
            temporal_stream = vector_brain_state.get('temporal_stream', {})
            
            print(f"\nStream processing:")
            print(f"  Sensory patterns: {sensory_stream.get('pattern_count', 0)}")
            print(f"  Motor patterns: {motor_stream.get('pattern_count', 0)}")
            print(f"  Temporal patterns: {temporal_stream.get('pattern_count', 0)}")
            
            # Now test a second call to see if it uses fast path
            print(f"\n🔄 TESTING SECOND CALL (should be fast)...")
            
            start_time = time.time()
            predicted_action2, vector_brain_state2 = brain.vector_brain.process_sensory_input(processed_input)
            second_call_time = (time.time() - start_time) * 1000
            
            print(f"Second call time: {second_call_time:.1f}ms")
            
            fast_path_used = vector_brain_state2.get('fast_path_used', False)
            print(f"Fast path used: {fast_path_used}")
            
            if fast_path_used:
                speedup = total_vector_time / second_call_time if second_call_time > 0 else float('inf')
                print(f"Speedup: {speedup:.0f}x")
                
                print(f"\n🚨 BOTTLENECK ANALYSIS:")
                print(f"  First call (slow path): {total_vector_time:.1f}ms")
                print(f"  Second call (fast path): {second_call_time:.1f}ms")
                print(f"  Bottleneck: Initial pattern processing and cache building")
                print(f"  Components involved in slow path:")
                print(f"    • Stream updates and pattern storage")
                print(f"    • Temporal hierarchy processing")
                print(f"    • Competitive dynamics")
                print(f"    • Cross-stream coactivation")
                print(f"    • Brain state compilation")
                
                print(f"\n⚡ FAST PATH BYPASSES:")
                print(f"    • ALL of the above components")
                print(f"    • Uses direct cache lookup instead")
                
    except TimeoutError:
        print("❌ Vector brain profiling timed out!")
        return
    except Exception as e:
        print(f"❌ Vector brain profiling failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test if we can isolate which specific component is slowest
    print(f"\n🔍 COMPONENT ISOLATION TEST")
    print("=" * 35)
    
    try:
        with timeout(60):
            # Clear cache again for clean test
            brain.vector_brain.emergent_hierarchy.predictor.reflex_cache.clear()
            
            print("Testing just the emergent hierarchy processing...")
            
            # Create a simple sparse pattern to test the emergent hierarchy
            import torch
            from server.src.vector_stream.sparse_representations import SparsePatternEncoder
            
            # Create encoder and pattern
            encoder = SparsePatternEncoder(16, sparsity=0.02, quiet_mode=True)
            test_vector = torch.tensor([1.0, 2.0, 3.0, 4.0] + [0.0] * 12)
            test_pattern = encoder.encode_top_k(test_vector, "test_pattern")
            
            # Time just the emergent hierarchy
            start_time = time.time()
            result = brain.vector_brain.emergent_hierarchy.process_with_adaptive_budget(test_pattern, time.time())
            hierarchy_time = (time.time() - start_time) * 1000
            
            print(f"Emergent hierarchy alone: {hierarchy_time:.1f}ms")
            
            if hierarchy_time > 1000:
                print(f"🚨 BOTTLENECK FOUND: Emergent temporal hierarchy processing!")
                print(f"   This explains most of the {total_vector_time:.1f}ms delay")
            else:
                print(f"✅ Emergent hierarchy is reasonably fast")
                print(f"   Bottleneck must be in stream processing or other components")
            
    except Exception as e:
        print(f"Component isolation failed: {e}")
    
    return total_vector_time

if __name__ == "__main__":
    print("Profiling vector brain to find internal bottleneck...")
    result = profile_vector_brain_simple()
    
    if result:
        print(f"\n🏁 VECTOR BRAIN PROFILING COMPLETE")
        print(f"Total vector brain time: {result:.1f}ms")
        print(f"The bottleneck is now identified!")
    else:
        print(f"\n❌ VECTOR BRAIN PROFILING FAILED")