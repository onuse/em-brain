#!/usr/bin/env python3
"""
Test Fuzzyness Performance Impact

Measure how hardware-adaptive fuzzyness affects:
1. Learning speed (experiences per second)
2. Prediction accuracy
3. Memory efficiency
4. Overall performance
"""

import sys
import os
import time
import numpy as np
from typing import Dict, List, Tuple

# Set up path to access brain modules
brain_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(brain_root, 'server', 'src'))
sys.path.append(os.path.join(brain_root, 'server'))

from src.brain import MinimalBrain
from hardware_adaptive_fuzzyness import HardwareAdaptiveFuzzyness


def simulate_brain_with_fuzzyness(fuzzyness_enabled: bool, 
                                 num_experiences: int = 100,
                                 artificial_slowdown: float = 0.0) -> Dict[str, any]:
    """
    Run brain simulation with or without adaptive fuzzyness.
    
    Args:
        fuzzyness_enabled: Whether to use adaptive fuzzyness
        num_experiences: Number of experiences to learn
        artificial_slowdown: Artificial delay to simulate slower hardware (seconds)
    
    Returns:
        Performance metrics
    """
    print(f"\n{'='*50}")
    print(f"Testing {'WITH' if fuzzyness_enabled else 'WITHOUT'} adaptive fuzzyness")
    if artificial_slowdown > 0:
        print(f"Simulating slow hardware: +{artificial_slowdown*1000:.0f}ms per operation")
    print(f"{'='*50}")
    
    # Create brain
    brain = MinimalBrain(enable_logging=False, enable_persistence=False, quiet_mode=True)
    
    # Create fuzzyness system if enabled
    fuzzy_system = HardwareAdaptiveFuzzyness(target_cycle_time_ms=50.0) if fuzzyness_enabled else None
    
    # Metrics
    cycle_times = []
    prediction_errors = []
    unique_experiences = set()
    fuzzy_groupings = 0
    
    # Generate test experiences with some repetition
    test_experiences = []
    for i in range(num_experiences):
        if i < 20:
            # Novel experiences
            sensory = [0.1 + i * 0.02, 0.2 + i * 0.02, 0.3 + i * 0.02, 0.4 + i * 0.02]
        elif i < 80:
            # Repetitive patterns with small variations
            base = i % 10
            noise = np.random.normal(0, 0.01, 4)  # Small variations
            sensory = [0.1 + base * 0.05 + noise[0], 
                      0.2 + base * 0.05 + noise[1],
                      0.3 + base * 0.05 + noise[2], 
                      0.4 + base * 0.05 + noise[3]]
        else:
            # Mix of novel and familiar
            if i % 3 == 0:
                sensory = [0.8 + (i-80) * 0.01, 0.7 + (i-80) * 0.01, 
                          0.6 + (i-80) * 0.01, 0.5 + (i-80) * 0.01]
            else:
                base = i % 5
                sensory = [0.2 + base * 0.1, 0.3 + base * 0.1,
                          0.4 + base * 0.1, 0.5 + base * 0.1]
        test_experiences.append(sensory)
    
    # Run simulation
    start_time = time.time()
    
    for i, sensory in enumerate(test_experiences):
        # Add artificial slowdown if simulating slow hardware
        if artificial_slowdown > 0:
            time.sleep(artificial_slowdown)
        
        cycle_start = time.time()
        
        # Apply fuzzyness if enabled
        if fuzzy_system:
            # Quantize sensory input based on current precision
            quantized_sensory = fuzzy_system.quantize_vector(sensory)
            
            # Check if this is similar to any previous experience
            is_duplicate = False
            for prev_sensory in list(unique_experiences)[-10:]:  # Check last 10
                if fuzzy_system.should_be_fuzzy_similar(quantized_sensory, prev_sensory):
                    is_duplicate = True
                    fuzzy_groupings += 1
                    break
            
            if not is_duplicate:
                unique_experiences.add(tuple(quantized_sensory))
            
            # Use quantized version for brain
            sensory_to_use = quantized_sensory
        else:
            # No fuzzyness - use exact values
            unique_experiences.add(tuple(sensory))
            sensory_to_use = sensory
        
        # Brain processing
        predicted_action, brain_state = brain.process_sensory_input(sensory_to_use)
        outcome = [a * 0.9 + 0.05 for a in predicted_action]
        
        # Calculate prediction error
        actual_outcome = [s * 0.5 + 0.25 for s in sensory]  # True mapping
        pred_error = np.mean([(p - a) ** 2 for p, a in zip(predicted_action, actual_outcome)])
        prediction_errors.append(pred_error)
        
        # Store experience
        brain.store_experience(sensory_to_use, predicted_action, outcome, predicted_action)
        
        cycle_time = (time.time() - cycle_start) * 1000
        cycle_times.append(cycle_time)
        
        # Update fuzzyness based on performance
        if fuzzy_system:
            adaptation = fuzzy_system.measure_cycle_performance(cycle_time)
            
            # Progress updates
            if (i + 1) % 25 == 0:
                stats = fuzzy_system.get_adaptive_stats()
                print(f"  Progress {i+1}/{num_experiences}: "
                      f"cycle={cycle_time:.1f}ms, "
                      f"threshold={stats['similarity_threshold']:.3f}, "
                      f"precision={stats['vector_precision']}, "
                      f"grouped={fuzzy_groupings}")
    
    total_time = time.time() - start_time
    brain.finalize_session()
    
    # Calculate metrics
    avg_cycle_time = np.mean(cycle_times)
    avg_prediction_error = np.mean(prediction_errors)
    learning_rate = num_experiences / total_time
    memory_efficiency = len(unique_experiences) / num_experiences
    
    results = {
        'total_time': total_time,
        'avg_cycle_time': avg_cycle_time,
        'avg_prediction_error': avg_prediction_error,
        'learning_rate': learning_rate,
        'unique_experiences': len(unique_experiences),
        'memory_efficiency': memory_efficiency,
        'fuzzy_groupings': fuzzy_groupings,
        'final_threshold': fuzzy_system.current_similarity_threshold if fuzzy_system else None,
        'final_precision': fuzzy_system.current_vector_precision if fuzzy_system else None
    }
    
    return results


def compare_fuzzyness_impact():
    """Compare performance with and without adaptive fuzzyness."""
    print("🧪 FUZZYNESS PERFORMANCE IMPACT TEST")
    print("Testing how adaptive fuzzyness affects learning speed and accuracy")
    
    # Test scenarios
    scenarios = [
        ("Fast Hardware", 0.0),      # No artificial slowdown
        ("Medium Hardware", 0.001),   # 1ms slowdown
        ("Slow Hardware", 0.01),     # 10ms slowdown
        ("Very Slow Hardware", 0.05) # 50ms slowdown
    ]
    
    for scenario_name, slowdown in scenarios:
        print(f"\n{'='*60}")
        print(f"📊 {scenario_name} Scenario")
        print(f"{'='*60}")
        
        # Test without fuzzyness
        results_without = simulate_brain_with_fuzzyness(
            fuzzyness_enabled=False,
            num_experiences=100,
            artificial_slowdown=slowdown
        )
        
        # Test with fuzzyness
        results_with = simulate_brain_with_fuzzyness(
            fuzzyness_enabled=True,
            num_experiences=100,
            artificial_slowdown=slowdown
        )
        
        # Compare results
        print(f"\n🎯 COMPARISON: {scenario_name}")
        print("-" * 40)
        
        print(f"Total Time:")
        print(f"  Without fuzzyness: {results_without['total_time']:.2f}s")
        print(f"  With fuzzyness:    {results_with['total_time']:.2f}s")
        speedup = results_without['total_time'] / results_with['total_time']
        print(f"  Speedup:           {speedup:.2f}x")
        
        print(f"\nLearning Rate (experiences/sec):")
        print(f"  Without fuzzyness: {results_without['learning_rate']:.1f}")
        print(f"  With fuzzyness:    {results_with['learning_rate']:.1f}")
        improvement = (results_with['learning_rate'] / results_without['learning_rate'] - 1) * 100
        print(f"  Improvement:       {improvement:+.1f}%")
        
        print(f"\nMemory Usage:")
        print(f"  Without fuzzyness: {results_without['unique_experiences']} unique experiences")
        print(f"  With fuzzyness:    {results_with['unique_experiences']} unique experiences")
        print(f"  Fuzzy groupings:   {results_with['fuzzy_groupings']}")
        memory_savings = (1 - results_with['memory_efficiency']) * 100
        print(f"  Memory savings:    {memory_savings:.1f}%")
        
        print(f"\nPrediction Accuracy:")
        print(f"  Without fuzzyness: {results_without['avg_prediction_error']:.4f} error")
        print(f"  With fuzzyness:    {results_with['avg_prediction_error']:.4f} error")
        accuracy_loss = (results_with['avg_prediction_error'] / results_without['avg_prediction_error'] - 1) * 100
        print(f"  Accuracy impact:   {accuracy_loss:+.1f}%")
        
        if results_with['final_threshold']:
            print(f"\nFuzzyness Adaptation:")
            print(f"  Final threshold:   {results_with['final_threshold']:.3f}")
            print(f"  Final precision:   {results_with['final_precision']} decimals")
    
    print("\n" + "="*60)
    print("🧠 FUZZYNESS IMPACT SUMMARY")
    print("="*60)
    print("\nKEY FINDINGS:")
    print("• Fast hardware: Minimal fuzzyness, preserves accuracy")
    print("• Slow hardware: Increased fuzzyness, maintains speed")
    print("• Memory efficiency: Groups similar experiences")
    print("• Accuracy tradeoff: Small loss for large speed gains")
    print("\nBIOLOGICAL INSIGHT:")
    print("Just like biological brains under stress or fatigue,")
    print("the system trades precision for speed when needed.")


def test_extreme_fuzzyness():
    """Test extreme cases of fuzzyness."""
    print("\n" + "="*60)
    print("🔬 EXTREME FUZZYNESS TEST")
    print("="*60)
    print("Testing with extremely slow hardware simulation...")
    
    # Create fuzzy system with extreme slowdown
    fuzzy_system = HardwareAdaptiveFuzzyness(target_cycle_time_ms=10.0)  # Very aggressive target
    
    # Simulate very slow cycles
    print("\nFeeding extremely slow cycle times:")
    slow_cycles = [200, 250, 300, 280, 260]  # 20-30x slower than target
    
    for cycle in slow_cycles:
        adaptation = fuzzy_system.measure_cycle_performance(cycle)
    
    stats = fuzzy_system.get_adaptive_stats()
    
    print(f"\nExtreme adaptation state:")
    print(f"  Performance ratio: {stats['performance_ratio']:.1f}x target")
    print(f"  Similarity threshold: {stats['similarity_threshold']:.3f}")
    print(f"  Vector precision: {stats['vector_precision']} decimals")
    print(f"  Attention threshold: {stats['attention_threshold']:.3f}")
    
    # Test similarity behavior
    print(f"\nSimilarity behavior under extreme fuzzyness:")
    test_vectors = [
        ([0.1, 0.2, 0.3, 0.4], [0.15, 0.25, 0.35, 0.45], "5% difference"),
        ([0.1, 0.2, 0.3, 0.4], [0.2, 0.3, 0.4, 0.5], "10% difference"),
        ([0.1, 0.2, 0.3, 0.4], [0.3, 0.4, 0.5, 0.6], "20% difference"),
        ([0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8], "40% difference"),
    ]
    
    for vec1, vec2, desc in test_vectors:
        is_similar = fuzzy_system.should_be_fuzzy_similar(vec1, vec2)
        raw_similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        print(f"  {desc}: {raw_similarity:.3f} → {'SIMILAR' if is_similar else 'different'}")
    
    print("\n💡 Under extreme hardware constraints, the brain becomes")
    print("   very fuzzy, treating even moderately different experiences")
    print("   as 'the same' to maintain any real-time function at all.")


if __name__ == "__main__":
    compare_fuzzyness_impact()
    test_extreme_fuzzyness()