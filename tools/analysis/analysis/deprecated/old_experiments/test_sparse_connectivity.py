#!/usr/bin/env python3
"""
Test Sparse Connectivity Implementation

Verifies that the brain actually uses sparse connectivity and doesn't
fall back to O(n²) operations despite having sparse connections stored.
"""

import sys
import os
import time
import numpy as np
from typing import List, Dict, Any

# Set up path to access brain modules
brain_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(brain_root, 'server', 'src'))
sys.path.append(os.path.join(brain_root, 'server'))

from src.brain import MinimalBrain
from src.experience import Experience

def test_sparse_connectivity_storage():
    """Test that sparse connectivity is properly stored in experiences."""
    print("🧪 TESTING SPARSE CONNECTIVITY STORAGE")
    print("=" * 50)
    
    # Create a fresh brain for testing
    brain = MinimalBrain(enable_logging=False, enable_persistence=False)
    
    # Create 20 experiences to test sparse connectivity
    test_experiences = []
    for i in range(20):
        sensory = [0.1 * i, 0.2 * i, 0.3 * i, 0.4 * i]
        action = [0.5 * i, 0.6 * i, 0.7 * i, 0.8 * i]
        outcome = [0.9 * i, 0.8 * i, 0.7 * i, 0.6 * i]
        
        exp_id = brain.store_experience(
            sensory_input=sensory,
            action_taken=action,
            outcome=outcome,
            predicted_action=action
        )
        test_experiences.append(exp_id)
    
    print(f"✅ Created {len(test_experiences)} experiences")
    
    # Check how many connections each experience has
    print(f"\n📊 ANALYZING SPARSE CONNECTIVITY:")
    total_connections = 0
    max_connections = 0
    min_connections = float('inf')
    
    for exp_id in test_experiences:
        exp = brain.experience_storage._experiences[exp_id]
        
        # Check both possible attribute names
        if hasattr(exp, 'similarity_connections'):
            num_connections = len(exp.similarity_connections)
        elif hasattr(exp, 'similar_experiences'):
            num_connections = len(exp.similar_experiences)
        else:
            num_connections = 0
        
        total_connections += num_connections
        max_connections = max(max_connections, num_connections)
        min_connections = min(min_connections, num_connections)
        
        print(f"  Experience {exp_id[:8]}...: {num_connections} connections")
    
    avg_connections = total_connections / len(test_experiences)
    theoretical_dense = len(test_experiences) * (len(test_experiences) - 1)  # n*(n-1)
    
    print(f"\n📈 CONNECTIVITY STATISTICS:")
    print(f"  - Total connections: {total_connections}")
    print(f"  - Average connections per experience: {avg_connections:.1f}")
    print(f"  - Max connections: {max_connections}")
    print(f"  - Min connections: {min_connections}")
    print(f"  - Theoretical dense (n*(n-1)): {theoretical_dense}")
    print(f"  - Sparsity ratio: {total_connections / theoretical_dense:.3f}")
    
    # Test result
    if avg_connections <= 10:  # Should be around 5 based on top-5 logic
        print(f"  ✅ SUCCESS: Sparse connectivity is working (avg {avg_connections:.1f} ≤ 10)")
        return True
    else:
        print(f"  ❌ FAILURE: Still dense connectivity (avg {avg_connections:.1f} > 10)")
        return False

def test_activation_processing_complexity():
    """Test if activation processing actually uses sparse connectivity."""
    print(f"\n🧪 TESTING ACTIVATION PROCESSING COMPLEXITY")
    print("=" * 50)
    
    brain = MinimalBrain(enable_logging=False, enable_persistence=False)
    
    # Create different dataset sizes to test scaling
    dataset_sizes = [10, 50, 100]
    processing_times = []
    
    for size in dataset_sizes:
        print(f"\n📊 Testing with {size} experiences:")
        
        # Create fresh brain for each test
        brain = MinimalBrain(enable_logging=False, enable_persistence=False)
        
        # Create experiences
        for i in range(size):
            sensory = [0.1 * i + 0.01 * np.random.random() for _ in range(4)]
            action = [0.5 * i + 0.01 * np.random.random() for _ in range(4)]
            outcome = [0.9 * i + 0.01 * np.random.random() for _ in range(4)]
            
            brain.store_experience(
                sensory_input=sensory,
                action_taken=action,
                outcome=outcome,
                predicted_action=action
            )
        
        # Test processing time for activation
        test_sensory = [0.5, 0.5, 0.5, 0.5]
        
        start_time = time.time()
        for _ in range(10):  # Multiple iterations for better measurement
            predicted_action, brain_state = brain.process_sensory_input(test_sensory)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 10
        processing_times.append(avg_time)
        
        print(f"  - Average processing time: {avg_time*1000:.2f}ms")
    
    # Analyze scaling pattern
    print(f"\n📈 SCALING ANALYSIS:")
    for i, (size, time_taken) in enumerate(zip(dataset_sizes, processing_times)):
        print(f"  - {size} experiences: {time_taken*1000:.2f}ms")
        
        if i > 0:
            scale_factor = size / dataset_sizes[i-1]
            time_ratio = time_taken / processing_times[i-1]
            print(f"    Scale factor: {scale_factor:.1f}x, Time ratio: {time_ratio:.1f}x")
    
    # Check if scaling is reasonable (should be sub-quadratic)
    if len(processing_times) >= 2:
        final_scale = dataset_sizes[-1] / dataset_sizes[0]
        final_time_ratio = processing_times[-1] / processing_times[0]
        
        print(f"\n🎯 OVERALL SCALING:")
        print(f"  - Dataset scaled by: {final_scale:.1f}x")
        print(f"  - Time scaled by: {final_time_ratio:.1f}x")
        
        if final_time_ratio < final_scale * final_scale:  # Better than O(n²)
            print(f"  ✅ SUCCESS: Sub-quadratic scaling detected")
            return True
        else:
            print(f"  ❌ FAILURE: Appears to be O(n²) or worse scaling")
            return False
    
    return False

def test_similarity_search_operations():
    """Test if similarity search operations use sparse connectivity."""
    print(f"\n🧪 TESTING SIMILARITY SEARCH OPERATIONS")
    print("=" * 50)
    
    brain = MinimalBrain(enable_logging=False, enable_persistence=False)
    
    # Create 30 experiences
    for i in range(30):
        sensory = [0.1 * i, 0.2 * i, 0.3 * i, 0.4 * i]
        action = [0.5 * i, 0.6 * i, 0.7 * i, 0.8 * i]
        outcome = [0.9 * i, 0.8 * i, 0.7 * i, 0.6 * i]
        
        brain.store_experience(
            sensory_input=sensory,
            action_taken=action,
            outcome=outcome,
            predicted_action=action
        )
    
    # Test similarity search with different thresholds
    test_vector = [0.5, 1.0, 1.5, 2.0]
    
    print(f"🔍 Testing similarity search with different thresholds:")
    
    # Get all experience vectors
    experience_vectors = []
    experience_ids = []
    for exp_id, exp in brain.experience_storage._experiences.items():
        experience_vectors.append(exp.get_context_vector())
        experience_ids.append(exp_id)
    
    thresholds = [0.0, 0.3, 0.5, 0.7, 0.9]
    
    for threshold in thresholds:
        similar_experiences = brain.similarity_engine.find_similar_experiences(
            test_vector, experience_vectors, experience_ids,
            max_results=50, min_similarity=threshold
        )
        
        print(f"  - Threshold {threshold}: {len(similar_experiences)} results")
    
    # Test if the brain actually uses thresholds in activation
    print(f"\n🔍 Testing if brain uses similarity thresholds in activation:")
    
    # Mock the activation process to see what gets activated
    brain._activate_by_utility([0.5, 1.0, 1.5, 2.0])
    
    # Check how many experiences are activated
    activated_count = 0
    if hasattr(brain.activation_dynamics, 'current_activations'):
        activated_count = len(brain.activation_dynamics.current_activations)
    
    print(f"  - Activated experiences: {activated_count} out of 30")
    
    if activated_count < 20:  # Should be sparse
        print(f"  ✅ SUCCESS: Activation uses sparse connectivity")
        return True
    else:
        print(f"  ❌ FAILURE: Activation may still be dense")
        return False

def main():
    """Run sparse connectivity tests."""
    print("🕸️ SPARSE CONNECTIVITY IMPLEMENTATION TEST")
    print("=" * 60)
    print("Testing if the brain actually uses sparse connectivity...")
    
    # Test 1: Sparse connectivity storage
    storage_sparse = test_sparse_connectivity_storage()
    
    # Test 2: Activation processing complexity
    processing_sparse = test_activation_processing_complexity()
    
    # Test 3: Similarity search operations
    search_sparse = test_similarity_search_operations()
    
    print(f"\n" + "=" * 60)
    print(f"🎯 OVERALL TEST RESULTS:")
    print(f"  - Sparse connectivity storage: {'✅ PASS' if storage_sparse else '❌ FAIL'}")
    print(f"  - Activation processing complexity: {'✅ PASS' if processing_sparse else '❌ FAIL'}")
    print(f"  - Similarity search operations: {'✅ PASS' if search_sparse else '❌ FAIL'}")
    
    all_passed = storage_sparse and processing_sparse and search_sparse
    
    if all_passed:
        print(f"\n✅ SUCCESS: Sparse connectivity is properly implemented!")
        print(f"   The brain uses sparse connections and avoids O(n²) operations.")
    else:
        print(f"\n❌ FAILURE: Sparse connectivity has issues.")
        print(f"   Need to implement engineered sparse connectivity optimizations.")
        
        print(f"\n🔧 RECOMMENDATIONS:")
        if not storage_sparse:
            print(f"  - Fix sparse connection storage (should be ~5 connections per experience)")
        if not processing_sparse:
            print(f"  - Fix activation processing to use sparse connectivity")
        if not search_sparse:
            print(f"  - Fix similarity search to respect thresholds")
    
    return all_passed

if __name__ == "__main__":
    main()