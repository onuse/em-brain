#!/usr/bin/env python3
"""
Quick Emergence Test - Rapid validation of brain architectures

Tests emergence in minutes instead of hours by focusing on the core hypothesis:
Do sparse representations actually improve emergent intelligence?

Key tests:
1. Pattern learning rate
2. Motor coordination development  
3. Pattern capacity scaling
4. Memory efficiency

Designed for fast iteration and objective comparison.
"""

import sys
import os
import time
import torch
import numpy as np
from typing import Dict, List, Tuple

# Add server src to path
server_src = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'server', 'src')
sys.path.insert(0, server_src)


def test_pattern_learning_rate(brain, brain_name: str, num_trials: int = 3) -> Dict:
    """Test how quickly brain learns to associate patterns."""
    print(f"  Testing pattern learning rate for {brain_name}...")
    
    results = []
    
    for trial in range(num_trials):
        # Reset brain if possible
        if hasattr(brain, 'reset_brain'):
            brain.reset_brain()
        
        # Simple pattern association: Red→Right, Blue→Left
        patterns = [
            ([1.0, 0.0, 0.0] + [0.0] * 13, [1.0, 0.0]),  # Red → Right
            ([0.0, 0.0, 1.0] + [0.0] * 13, [-1.0, 0.0]), # Blue → Left
        ]
        
        learning_curve = []
        
        # Train and test learning
        for cycle in range(100):
            # Train on patterns
            for sensory, target_motor in patterns:
                motor_output, _ = brain.process_sensory_input(sensory)
            
            # Test accuracy every 10 cycles
            if cycle % 10 == 0:
                total_accuracy = 0.0
                for sensory, target_motor in patterns:
                    motor_output, _ = brain.process_sensory_input(sensory)
                    
                    # Calculate accuracy (cosine similarity)
                    if len(motor_output) >= 2 and len(target_motor) >= 2:
                        motor_vec = np.array(motor_output[:2])
                        target_vec = np.array(target_motor[:2])
                        
                        if np.linalg.norm(motor_vec) > 0 and np.linalg.norm(target_vec) > 0:
                            similarity = np.dot(motor_vec, target_vec) / (
                                np.linalg.norm(motor_vec) * np.linalg.norm(target_vec)
                            )
                            accuracy = (similarity + 1) / 2  # 0-1 scale
                            total_accuracy += accuracy
                
                avg_accuracy = total_accuracy / len(patterns)
                learning_curve.append(avg_accuracy)
        
        # Calculate learning metrics
        if len(learning_curve) > 2:
            final_performance = learning_curve[-1]
            learning_rate = (learning_curve[-1] - learning_curve[0]) / len(learning_curve)
            max_performance = max(learning_curve)
        else:
            final_performance = 0.0
            learning_rate = 0.0
            max_performance = 0.0
        
        results.append({
            'final_performance': final_performance,
            'learning_rate': learning_rate,
            'max_performance': max_performance,
            'learning_curve': learning_curve
        })
    
    # Average across trials
    avg_final = np.mean([r['final_performance'] for r in results])
    avg_learning_rate = np.mean([r['learning_rate'] for r in results])
    avg_max = np.mean([r['max_performance'] for r in results])
    
    print(f"    Final performance: {avg_final:.3f}")
    print(f"    Learning rate: {avg_learning_rate:.3f}")
    print(f"    Max performance: {avg_max:.3f}")
    
    return {
        'final_performance': avg_final,
        'learning_rate': avg_learning_rate,
        'max_performance': avg_max,
        'num_trials': num_trials
    }


def test_pattern_capacity(brain, brain_name: str) -> Dict:
    """Test how well brain scales with pattern count."""
    print(f"  Testing pattern capacity for {brain_name}...")
    
    # Reset brain if possible
    if hasattr(brain, 'reset_brain'):
        brain.reset_brain()
    
    response_times = []
    pattern_counts = []
    
    # Store increasing numbers of patterns and measure response time
    for i in range(200):  # Quick test with 200 patterns
        pattern = np.random.randn(16).tolist()
        
        start_time = time.time()
        motor_output, brain_state = brain.process_sensory_input(pattern)
        response_time = time.time() - start_time
        
        response_times.append(response_time * 1000)  # Convert to ms
        
        if 'total_patterns' in brain_state:
            pattern_counts.append(brain_state['total_patterns'])
        else:
            pattern_counts.append(i + 1)
    
    # Calculate scaling metrics
    avg_response_time = np.mean(response_times)
    response_time_std = np.std(response_times)
    final_pattern_count = pattern_counts[-1] if pattern_counts else 0
    
    # Performance metric (higher = better)
    performance = 1.0 / (avg_response_time + 0.001)
    
    print(f"    Average response time: {avg_response_time:.2f}ms")
    print(f"    Pattern count: {final_pattern_count}")
    print(f"    Performance score: {performance:.1f}")
    
    return {
        'avg_response_time_ms': avg_response_time,
        'response_time_std': response_time_std,
        'final_pattern_count': final_pattern_count,
        'performance_score': performance
    }


def test_memory_efficiency(brain, brain_name: str) -> Dict:
    """Test memory usage efficiency."""
    print(f"  Testing memory efficiency for {brain_name}...")
    
    # Get brain statistics
    if hasattr(brain, 'get_brain_statistics'):
        stats = brain.get_brain_statistics()
        
        total_patterns = stats.get('total_patterns', 0)
        memory_usage = stats.get('memory_usage_mb', 0.0)
        
        if memory_usage > 0:
            patterns_per_mb = total_patterns / memory_usage
        else:
            patterns_per_mb = 0.0
        
        print(f"    Patterns stored: {total_patterns}")
        print(f"    Memory usage: {memory_usage:.1f}MB")
        print(f"    Patterns per MB: {patterns_per_mb:.0f}")
        
        return {
            'total_patterns': total_patterns,
            'memory_usage_mb': memory_usage,
            'patterns_per_mb': patterns_per_mb
        }
    else:
        print(f"    Brain statistics not available")
        return {
            'total_patterns': 0,
            'memory_usage_mb': 0.0,
            'patterns_per_mb': 0.0
        }


def create_test_brains() -> Dict:
    """Create different brain architectures for testing."""
    brains = {}
    
    print("🧠 Creating test brain architectures...")
    
    # Try to create different brain types
    try:
        from vector_stream.goldilocks_brain import GoldilocksBrain
        brains['goldilocks'] = GoldilocksBrain(
            sensory_dim=16, motor_dim=8, temporal_dim=4, 
            max_patterns=5000, quiet_mode=True
        )
        print("  ✅ Goldilocks brain created")
    except ImportError as e:
        print(f"  ❌ Goldilocks brain not available: {e}")
    
    try:
        from vector_stream.sparse_goldilocks_brain import SparseGoldilocksBrain
        brains['sparse'] = SparseGoldilocksBrain(
            sensory_dim=16, motor_dim=8, temporal_dim=4,
            max_patterns=5000, quiet_mode=True
        )
        print("  ✅ Sparse Goldilocks brain created")
    except ImportError as e:
        print(f"  ❌ Sparse Goldilocks brain not available: {e}")
    
    try:
        from vector_stream.minimal_brain import MinimalVectorStreamBrain
        brains['minimal'] = MinimalVectorStreamBrain(
            sensory_dim=16, motor_dim=8, temporal_dim=4
        )
        print("  ✅ Minimal brain created")
    except ImportError as e:
        print(f"  ❌ Minimal brain not available: {e}")
    
    return brains


def run_quick_emergence_validation():
    """Run quick emergence validation on available brain architectures."""
    print("\n🧪 QUICK EMERGENCE VALIDATION")
    print("=" * 50)
    print("Rapid testing of emergent intelligence across brain architectures")
    
    # Create test brains
    brains = create_test_brains()
    
    if not brains:
        print("❌ No brain architectures available for testing")
        return
    
    print(f"\n🔬 Testing {len(brains)} brain architectures")
    
    # Run tests on each brain
    results = {}
    
    for brain_name, brain in brains.items():
        print(f"\n📊 TESTING {brain_name.upper()}")
        print("-" * 30)
        
        try:
            start_time = time.time()
            
            # Run tests
            pattern_learning = test_pattern_learning_rate(brain, brain_name)
            pattern_capacity = test_pattern_capacity(brain, brain_name)
            memory_efficiency = test_memory_efficiency(brain, brain_name)
            
            test_time = time.time() - start_time
            
            results[brain_name] = {
                'pattern_learning': pattern_learning,
                'pattern_capacity': pattern_capacity,
                'memory_efficiency': memory_efficiency,
                'test_time_seconds': test_time
            }
            
            print(f"  ✅ {brain_name} tests completed in {test_time:.1f}s")
            
        except Exception as e:
            print(f"  ❌ {brain_name} tests failed: {e}")
            results[brain_name] = {'error': str(e)}
    
    # Compare results
    print(f"\n📈 COMPARATIVE ANALYSIS")
    print("=" * 50)
    
    # Pattern learning comparison
    print("\n🧠 Pattern Learning Performance:")
    for brain_name, result in results.items():
        if 'pattern_learning' in result:
            pl = result['pattern_learning']
            print(f"  {brain_name:15} Final: {pl['final_performance']:.3f}, "
                  f"Learning Rate: {pl['learning_rate']:.3f}, "
                  f"Max: {pl['max_performance']:.3f}")
    
    # Capacity comparison
    print("\n⚡ Pattern Capacity Performance:")
    for brain_name, result in results.items():
        if 'pattern_capacity' in result:
            pc = result['pattern_capacity']
            print(f"  {brain_name:15} Response: {pc['avg_response_time_ms']:.2f}ms, "
                  f"Patterns: {pc['final_pattern_count']}, "
                  f"Score: {pc['performance_score']:.1f}")
    
    # Memory efficiency comparison
    print("\n💾 Memory Efficiency:")
    for brain_name, result in results.items():
        if 'memory_efficiency' in result:
            me = result['memory_efficiency']
            print(f"  {brain_name:15} Memory: {me['memory_usage_mb']:.1f}MB, "
                  f"Patterns: {me['total_patterns']}, "
                  f"Efficiency: {me['patterns_per_mb']:.0f} patterns/MB")
    
    # Determine winner
    print(f"\n🏆 EMERGENCE EVALUATION")
    print("-" * 30)
    
    best_learning = max(results.items(), 
                       key=lambda x: x[1].get('pattern_learning', {}).get('final_performance', 0))
    best_capacity = max(results.items(),
                       key=lambda x: x[1].get('pattern_capacity', {}).get('performance_score', 0))
    best_efficiency = max(results.items(),
                         key=lambda x: x[1].get('memory_efficiency', {}).get('patterns_per_mb', 0))
    
    print(f"Best learning performance: {best_learning[0]} ({best_learning[1].get('pattern_learning', {}).get('final_performance', 0):.3f})")
    print(f"Best capacity performance: {best_capacity[0]} ({best_capacity[1].get('pattern_capacity', {}).get('performance_score', 0):.1f})")
    print(f"Best memory efficiency: {best_efficiency[0]} ({best_efficiency[1].get('memory_efficiency', {}).get('patterns_per_mb', 0):.0f} patterns/MB)")
    
    # Check if sparse representations show advantages
    if 'sparse' in results and 'goldilocks' in results:
        print(f"\n🧬 SPARSE REPRESENTATION ANALYSIS")
        print("-" * 40)
        
        sparse_results = results['sparse']
        dense_results = results['goldilocks']
        
        if 'memory_efficiency' in sparse_results and 'memory_efficiency' in dense_results:
            sparse_efficiency = sparse_results['memory_efficiency']['patterns_per_mb']
            dense_efficiency = dense_results['memory_efficiency']['patterns_per_mb']
            
            if sparse_efficiency > dense_efficiency:
                efficiency_gain = sparse_efficiency / dense_efficiency if dense_efficiency > 0 else float('inf')
                print(f"✅ Sparse representations show {efficiency_gain:.1f}x memory efficiency gain")
            else:
                print(f"⚠️  Sparse representations do not show clear memory advantage")
        
        if 'pattern_learning' in sparse_results and 'pattern_learning' in dense_results:
            sparse_learning = sparse_results['pattern_learning']['final_performance']
            dense_learning = dense_results['pattern_learning']['final_performance']
            
            if sparse_learning > dense_learning:
                learning_gain = sparse_learning / dense_learning if dense_learning > 0 else float('inf')
                print(f"✅ Sparse representations show {learning_gain:.1f}x learning advantage")
            else:
                print(f"⚠️  Sparse representations do not show clear learning advantage")
    
    print(f"\n✅ QUICK EMERGENCE VALIDATION COMPLETE")
    return results


if __name__ == "__main__":
    results = run_quick_emergence_validation()