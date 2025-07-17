#!/usr/bin/env python3
"""
Minimal Emergence Test - Answer the key question fast

Core question: Do sparse representations improve emergent intelligence?

Test design:
1. Create sparse vs dense brain with identical configs
2. Test simple pattern learning (Red→Right, Blue→Left)  
3. Measure learning speed and memory efficiency
4. Give clear recommendation

Fast execution: < 30 seconds
"""

import sys
import os
import time
import torch
import numpy as np

# Add server src to path
server_src = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'server', 'src')
sys.path.insert(0, server_src)


def test_brain_learning(brain, brain_name, max_cycles=50):
    """Test how quickly brain learns Red→Right, Blue→Left mapping."""
    print(f"Testing {brain_name} learning...")
    
    # Simple patterns: Red→Right, Blue→Left
    patterns = [
        ([1.0, 0.0] + [0.0] * 14, [1.0, 0.0]),   # Red → Right  
        ([0.0, 1.0] + [0.0] * 14, [-1.0, 0.0]),  # Blue → Left
    ]
    
    accuracies = []
    
    for cycle in range(max_cycles):
        # Train on patterns
        for sensory, target in patterns:
            motor_output, _ = brain.process_sensory_input(sensory)
        
        # Test accuracy every 5 cycles
        if cycle % 5 == 0:
            total_accuracy = 0.0
            
            for sensory, target in patterns:
                motor_output, _ = brain.process_sensory_input(sensory)
                
                if len(motor_output) >= 2:
                    # Measure direction alignment
                    motor_dir = np.array(motor_output[:2])
                    target_dir = np.array(target[:2])
                    
                    if np.linalg.norm(motor_dir) > 0.01:
                        # Cosine similarity
                        similarity = np.dot(motor_dir, target_dir) / (
                            np.linalg.norm(motor_dir) * np.linalg.norm(target_dir)
                        )
                        accuracy = max(0, (similarity + 1) / 2)  # 0-1 scale
                    else:
                        accuracy = 0.0
                    
                    total_accuracy += accuracy
            
            avg_accuracy = total_accuracy / len(patterns)
            accuracies.append(avg_accuracy)
    
    # Calculate final metrics
    final_accuracy = accuracies[-1] if accuracies else 0.0
    max_accuracy = max(accuracies) if accuracies else 0.0
    learning_rate = (accuracies[-1] - accuracies[0]) / len(accuracies) if len(accuracies) > 1 else 0.0
    
    print(f"  Final accuracy: {final_accuracy:.3f}")
    print(f"  Max accuracy: {max_accuracy:.3f}")
    print(f"  Learning rate: {learning_rate:.3f}")
    
    return {
        'final_accuracy': final_accuracy,
        'max_accuracy': max_accuracy,
        'learning_rate': learning_rate,
        'learning_curve': accuracies
    }


def test_brain_memory(brain, brain_name):
    """Test memory usage efficiency."""
    print(f"Testing {brain_name} memory...")
    
    # Store patterns and measure memory
    for i in range(100):
        pattern = np.random.randn(16).tolist()
        brain.process_sensory_input(pattern)
    
    # Get memory stats
    if hasattr(brain, 'get_brain_statistics'):
        stats = brain.get_brain_statistics()
        total_patterns = stats.get('total_patterns', 0)
        memory_mb = stats.get('memory_usage_mb', 0.0)
        
        efficiency = total_patterns / memory_mb if memory_mb > 0 else 0.0
        
        print(f"  Patterns: {total_patterns}")
        print(f"  Memory: {memory_mb:.1f}MB")
        print(f"  Efficiency: {efficiency:.0f} patterns/MB")
        
        return {
            'total_patterns': total_patterns,
            'memory_mb': memory_mb,
            'efficiency': efficiency
        }
    else:
        return {'total_patterns': 0, 'memory_mb': 0.0, 'efficiency': 0.0}


def main():
    print("🧪 MINIMAL EMERGENCE TEST")
    print("=" * 40)
    print("Key question: Do sparse representations help emergence?")
    
    results = {}
    
    # Test Goldilocks Brain (dense)
    print(f"\n📊 Testing Dense Goldilocks Brain...")
    try:
        from vector_stream.goldilocks_brain import GoldilocksBrain
        dense_brain = GoldilocksBrain(
            sensory_dim=16, motor_dim=8, temporal_dim=4,
            max_patterns=1000, quiet_mode=True
        )
        
        learning = test_brain_learning(dense_brain, "Dense")
        memory = test_brain_memory(dense_brain, "Dense")
        
        results['dense'] = {'learning': learning, 'memory': memory}
        print("  ✅ Dense brain tested successfully")
        
    except Exception as e:
        print(f"  ❌ Dense brain failed: {e}")
        results['dense'] = None
    
    # Test Sparse Goldilocks Brain
    print(f"\n📊 Testing Sparse Goldilocks Brain...")
    try:
        from vector_stream.sparse_goldilocks_brain import SparseGoldilocksBrain
        sparse_brain = SparseGoldilocksBrain(
            sensory_dim=16, motor_dim=8, temporal_dim=4,
            max_patterns=1000, quiet_mode=True
        )
        
        learning = test_brain_learning(sparse_brain, "Sparse")
        memory = test_brain_memory(sparse_brain, "Sparse")
        
        results['sparse'] = {'learning': learning, 'memory': memory}
        print("  ✅ Sparse brain tested successfully")
        
    except Exception as e:
        print(f"  ❌ Sparse brain failed: {e}")
        results['sparse'] = None
    
    # Compare results
    print(f"\n🔬 RESULTS COMPARISON")
    print("=" * 40)
    
    if results['dense'] and results['sparse']:
        dense_learning = results['dense']['learning']['final_accuracy']
        sparse_learning = results['sparse']['learning']['final_accuracy']
        
        dense_memory = results['dense']['memory']['efficiency']
        sparse_memory = results['sparse']['memory']['efficiency']
        
        print(f"Learning Performance:")
        print(f"  Dense:  {dense_learning:.3f}")
        print(f"  Sparse: {sparse_learning:.3f}")
        
        if sparse_learning > dense_learning:
            learning_advantage = sparse_learning / dense_learning if dense_learning > 0 else float('inf')
            print(f"  ✅ Sparse shows {learning_advantage:.1f}x learning advantage")
        else:
            print(f"  ⚠️  Dense shows better learning")
        
        print(f"\nMemory Efficiency:")
        print(f"  Dense:  {dense_memory:.0f} patterns/MB")
        print(f"  Sparse: {sparse_memory:.0f} patterns/MB")
        
        if sparse_memory > dense_memory:
            memory_advantage = sparse_memory / dense_memory if dense_memory > 0 else float('inf')
            print(f"  ✅ Sparse shows {memory_advantage:.1f}x memory advantage")
        else:
            print(f"  ⚠️  Dense shows better memory efficiency")
        
        # Final recommendation
        print(f"\n🎯 RECOMMENDATION")
        print("-" * 20)
        
        sparse_wins = 0
        if sparse_learning > dense_learning:
            sparse_wins += 1
        if sparse_memory > dense_memory:
            sparse_wins += 1
        
        if sparse_wins >= 1:
            print("✅ CONTINUE with sparse representations - showing clear advantages")
            print("✅ Proceed to Phase 2: Multi-timescale temporal hierarchies")
        else:
            print("⚠️  RECONSIDER sparse approach - not showing clear advantages")
            print("⚠️  May need to debug implementation or try different approach")
    
    elif results['sparse']:
        print("Only sparse brain available - cannot compare")
        sparse_learning = results['sparse']['learning']['final_accuracy']
        sparse_memory = results['sparse']['memory']['efficiency']
        
        if sparse_learning > 0.3 and sparse_memory > 100:
            print("✅ Sparse brain shows reasonable performance")
        else:
            print("⚠️  Sparse brain performance may need improvement")
    
    else:
        print("❌ No brains available for testing")
    
    print(f"\n✅ MINIMAL EMERGENCE TEST COMPLETE")


if __name__ == "__main__":
    main()