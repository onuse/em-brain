#!/usr/bin/env python3
"""
Tensor Rebuild Analysis Tool

Analyzes where and when tensor rebuilding happens in the brain systems
and identifies optimization opportunities.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import numpy as np
from server.src.brain_factory import MinimalBrain
from server.src.utils.hardware_adaptation import get_hardware_adaptation


def create_test_experience(dim_sensory=16, dim_action=4):
    """Create a test experience with random data."""
    return {
        'sensory_input': np.random.randn(dim_sensory).tolist(),
        'action_taken': np.random.randn(dim_action).tolist(),
        'outcome': np.random.randn(dim_sensory).tolist()
    }


def track_tensor_rebuilds(brain, num_experiences=50):
    """Track when tensor rebuilds happen during experience processing."""
    
    print("\n=== Tensor Rebuild Analysis ===")
    print(f"Testing with {num_experiences} experiences")
    
    # Track rebuild events
    rebuild_events = {
        'similarity': [],
        'activation': [],
        'pattern': []
    }
    
    # Monitor initial state
    initial_gpu_state = {
        'similarity': hasattr(brain.similarity_engine.learnable_similarity, '_using_gpu') and 
                     brain.similarity_engine.learnable_similarity._using_gpu,
        'activation': brain.activation_dynamics.use_gpu,
        'pattern': brain.prediction_engine.pattern_analyzer.use_gpu if 
                  hasattr(brain.prediction_engine, 'pattern_analyzer') else False
    }
    
    print(f"\nInitial GPU state:")
    for system, using_gpu in initial_gpu_state.items():
        print(f"  {system}: {'GPU' if using_gpu else 'CPU'}")
    
    # Process experiences and track rebuilds
    start_time = time.time()
    
    for i in range(num_experiences):
        exp = create_test_experience()
        
        # Store pre-process state
        pre_state = {
            'similarity_gpu': hasattr(brain.similarity_engine.learnable_similarity, '_using_gpu') and 
                             brain.similarity_engine.learnable_similarity._using_gpu,
            'activation_gpu': brain.activation_dynamics.use_gpu,
            'pattern_gpu': brain.prediction_engine.pattern_analyzer.use_gpu if 
                          hasattr(brain.prediction_engine, 'pattern_analyzer') else False
        }
        
        # Get predicted action
        predicted_action, brain_state = brain.process_sensory_input(
            exp['sensory_input']
        )
        
        # Store experience
        brain.store_experience(
            exp['sensory_input'],
            exp['action_taken'],
            exp['outcome'],
            predicted_action
        )
        
        # Check post-process state
        post_state = {
            'similarity_gpu': hasattr(brain.similarity_engine.learnable_similarity, '_using_gpu') and 
                             brain.similarity_engine.learnable_similarity._using_gpu,
            'activation_gpu': brain.activation_dynamics.use_gpu,
            'pattern_gpu': brain.prediction_engine.pattern_analyzer.use_gpu if 
                          hasattr(brain.prediction_engine, 'pattern_analyzer') else False
        }
        
        # Detect GPU upgrades (which trigger rebuilds)
        if not pre_state['similarity_gpu'] and post_state['similarity_gpu']:
            rebuild_events['similarity'].append(i)
            print(f"\n🔄 Similarity GPU upgrade at experience {i}")
            
        if not pre_state['activation_gpu'] and post_state['activation_gpu']:
            rebuild_events['activation'].append(i)
            print(f"\n🔄 Activation GPU upgrade at experience {i}")
            
        if not pre_state['pattern_gpu'] and post_state['pattern_gpu']:
            rebuild_events['pattern'].append(i)
            print(f"\n🔄 Pattern GPU upgrade at experience {i}")
    
    elapsed_time = time.time() - start_time
    
    # Analysis results
    print(f"\n\n=== Analysis Results ===")
    print(f"Total time: {elapsed_time:.2f}s")
    print(f"Time per experience: {elapsed_time/num_experiences*1000:.2f}ms")
    
    print(f"\nGPU Upgrade Events:")
    for system, events in rebuild_events.items():
        if events:
            print(f"  {system}: at experiences {events}")
        else:
            print(f"  {system}: no upgrades")
    
    # Check final tensor states
    print(f"\nFinal GPU state:")
    print(f"  Similarity: {'GPU' if brain.similarity_engine.learnable_similarity._using_gpu else 'CPU'}")
    print(f"  Activation: {'GPU' if brain.activation_dynamics.use_gpu else 'CPU'}")
    if hasattr(brain.prediction_engine, 'pattern_analyzer'):
        print(f"  Pattern: {'GPU' if brain.prediction_engine.pattern_analyzer.use_gpu else 'CPU'}")
    
    # Check for continuous rebuilding
    print(f"\n=== Continuous Rebuild Analysis ===")
    
    # Test rapid experience addition
    rapid_start = time.time()
    for i in range(20):
        exp = create_test_experience()
        predicted_action, _ = brain.process_sensory_input(exp['sensory_input'])
        brain.store_experience(exp['sensory_input'], exp['action_taken'], 
                             exp['outcome'], predicted_action)
    rapid_time = time.time() - rapid_start
    
    print(f"Rapid addition test (20 experiences):")
    print(f"  Total time: {rapid_time:.2f}s")
    print(f"  Time per experience: {rapid_time/20*1000:.2f}ms")
    
    # Hardware adaptation thresholds
    hw_adapt = get_hardware_adaptation()
    print(f"\n=== Hardware Adaptation Thresholds ===")
    print(f"GPU activation thresholds:")
    print(f"  Base threshold: {hw_adapt.base_gpu_threshold}")
    print(f"  Similarity threshold: ~{hw_adapt.base_gpu_threshold}")
    print(f"  Activation threshold: ~{hw_adapt.base_gpu_threshold // 2}")
    print(f"  Pattern threshold: ~{hw_adapt.base_gpu_threshold // 5}")
    
    return rebuild_events


def analyze_batch_opportunities(brain):
    """Analyze opportunities for batch processing."""
    
    print("\n\n=== Batch Processing Opportunities ===")
    
    # Test single vs batch experience processing
    num_test = 10
    
    # Single experience processing
    single_times = []
    for i in range(num_test):
        exp = create_test_experience()
        start = time.time()
        predicted_action, _ = brain.process_sensory_input(exp['sensory_input'])
        brain.store_experience(exp['sensory_input'], exp['action_taken'], 
                             exp['outcome'], predicted_action)
        single_times.append(time.time() - start)
    
    avg_single_time = np.mean(single_times) * 1000  # Convert to ms
    
    print(f"\nSingle experience processing:")
    print(f"  Average time: {avg_single_time:.2f}ms")
    print(f"  Min time: {np.min(single_times)*1000:.2f}ms")
    print(f"  Max time: {np.max(single_times)*1000:.2f}ms")
    
    # Identify bottlenecks
    print(f"\n=== Bottleneck Analysis ===")
    
    # Profile a single experience processing
    exp = create_test_experience()
    
    # Time each component
    times = {}
    
    # Similarity search
    start = time.time()
    similar = brain.similarity_engine.find_similar_experiences(
        exp['sensory_input'],
        [e.get_context_vector() for e in brain.experience_storage.get_all_experiences()],
        [e.experience_id for e in brain.experience_storage.get_all_experiences()],
        max_results=10
    )
    times['similarity_search'] = time.time() - start
    
    # Activation update
    start = time.time()
    brain.activation_dynamics.update_all_activations(brain.experience_storage._experiences)
    times['activation_update'] = time.time() - start
    
    # Pattern analysis
    if hasattr(brain.prediction_engine, 'pattern_analyzer'):
        start = time.time()
        brain.prediction_engine.pattern_analyzer.add_experience_to_stream({
            'sensory_input': exp['sensory_input'],
            'action_taken': exp['action_taken'],
            'outcome': exp['outcome'],
            'timestamp': time.time()
        })
        times['pattern_analysis'] = time.time() - start
    
    print(f"\nComponent timing breakdown:")
    for component, timing in times.items():
        print(f"  {component}: {timing*1000:.2f}ms ({timing/sum(times.values())*100:.1f}%)")
    
    print(f"\n=== Optimization Recommendations ===")
    print("1. Batch experience addition to reduce tensor rebuild frequency")
    print("2. Implement incremental tensor updates instead of full rebuilds")
    print("3. Cache tensor states between operations")
    print("4. Use lazy tensor synchronization")
    print("5. Consider memory pooling for tensor allocations")


def main():
    """Main analysis function."""
    print("Tensor Rebuild Analysis Tool")
    print("=" * 50)
    
    # Create brain instance
    print("\nInitializing brain...")
    brain = MinimalBrain(enable_logging=False, enable_persistence=False)
    
    # Run analysis
    rebuild_events = track_tensor_rebuilds(brain)
    analyze_batch_opportunities(brain)
    
    print("\n\nAnalysis complete!")


if __name__ == "__main__":
    main()