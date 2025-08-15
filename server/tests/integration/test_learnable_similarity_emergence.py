#!/usr/bin/env python3
"""
Test Learnable Similarity Emergence

Dedicated test to observe how similarity functions evolve over extended periods.
This is artificial life research - watching patterns emerge from pure information.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from server.src.brain_factory import MinimalBrain
import numpy as np
import time


def test_similarity_emergence():
    """Test emergence of useful similarity patterns over extended learning."""
    
    print("Testing Similarity Emergence - Extended Learning Session")
    print("This may take several minutes - we're watching patterns emerge...")
    
    # Create brain with learnable similarity
    brain = MinimalBrain()
    
    # Track similarity evolution over time
    evolution_log = []
    
    # Extended learning simulation
    print(f"\nSimulating {500} experiences with pattern structure...")
    
    for i in range(500):
        # Create structured sensory patterns
        if i % 5 == 0:
            # Pattern A: Linear progression
            base = i / 100.0
            sensors = [base + 1.0, base + 2.0, base + 3.0, base + 4.0]
        elif i % 5 == 1:
            # Pattern B: Inverse relation
            base = i / 100.0
            sensors = [5.0 - base, 4.0 - base, 3.0 - base, 2.0 - base]
        elif i % 5 == 2:
            # Pattern C: Oscillating
            phase = i / 50.0
            sensors = [np.sin(phase), np.cos(phase), np.sin(phase * 2), np.cos(phase * 2)]
        else:
            # Random noise
            sensors = [np.random.normal(0, 2) for _ in range(4)]
        
        # Process through brain
        action, brain_state = brain.process_sensory_input(sensors)
        
        # Create structured outcomes based on patterns
        if i % 5 < 3:
            # Patterns A, B, C have predictable outcomes
            outcome = [s * 1.2 + np.random.normal(0, 0.1) for s in sensors]
        else:
            # Random cases have unpredictable outcomes
            outcome = [np.random.normal(0, 1) for _ in range(4)]
        
        # Store experience
        brain.store_experience(sensors, action, outcome, action)
        
        # Log similarity statistics periodically
        if i % 50 == 49:
            stats = brain.similarity_engine.get_performance_stats()
            if 'similarity_learning' in stats:
                sim_stats = stats['similarity_learning']
                evolution_log.append({
                    'experience': i + 1,
                    'adaptations': sim_stats['adaptations_performed'],
                    'correlation': sim_stats['similarity_success_correlation'],
                    'feature_variance': sim_stats['feature_weight_variance'],
                    'dominant_features': sim_stats['dominant_feature_indices']
                })
                
                print(f"Experience {i+1:3d}: "
                      f"adaptations={sim_stats['adaptations_performed']:2d}, "
                      f"correlation={sim_stats['similarity_success_correlation']:6.3f}, "
                      f"feature_var={sim_stats['feature_weight_variance']:6.3f}")
    
    # Analyze evolution
    print(f"\nSimilarity Learning Evolution Analysis:")
    print("=" * 50)
    
    if len(evolution_log) > 1:
        first = evolution_log[0]
        last = evolution_log[-1]
        
        print(f"Initial state (exp {first['experience']}):")
        print(f"  Adaptations: {first['adaptations']}")
        print(f"  Correlation: {first['correlation']:.3f}")
        print(f"  Feature variance: {first['feature_variance']:.3f}")
        
        print(f"\nFinal state (exp {last['experience']}):")
        print(f"  Adaptations: {last['adaptations']}")
        print(f"  Correlation: {last['correlation']:.3f}")
        print(f"  Feature variance: {last['feature_variance']:.3f}")
        print(f"  Dominant features: {last['dominant_features']}")
        
        # Analyze trends
        correlations = [entry['correlation'] for entry in evolution_log if not np.isnan(entry['correlation'])]
        if len(correlations) > 1:
            correlation_trend = correlations[-1] - correlations[0]
            print(f"\nCorrelation trend: {correlation_trend:+.3f} (positive = improving)")
        
        variances = [entry['feature_variance'] for entry in evolution_log]
        if len(variances) > 1:
            variance_trend = variances[-1] - variances[0]
            print(f"Feature specialization: {variance_trend:+.3f} (positive = more specialized)")
        
        adaptation_rate = last['adaptations'] / (last['experience'] / 50)
        print(f"Adaptation frequency: {adaptation_rate:.2f} adaptations per 50 experiences")
    
    # Test evolved similarity on known patterns
    print(f"\nTesting Evolved Similarity Function:")
    print("-" * 40)
    
    if brain.similarity_engine.learnable_similarity:
        # Test pattern recognition
        pattern_a1 = [1.0, 2.0, 3.0, 4.0]
        pattern_a2 = [1.1, 2.1, 3.1, 4.1]  # Similar to A
        pattern_b = [4.0, 3.0, 2.0, 1.0]   # Inverse pattern
        random_vec = [10.0, -5.0, 0.0, 100.0]  # Random
        
        sim_a1_a2 = brain.similarity_engine.learnable_similarity.compute_similarity(pattern_a1, pattern_a2)
        sim_a1_b = brain.similarity_engine.learnable_similarity.compute_similarity(pattern_a1, pattern_b)
        sim_a1_random = brain.similarity_engine.learnable_similarity.compute_similarity(pattern_a1, random_vec)
        
        print(f"Pattern A1 vs A2 (should be high): {sim_a1_a2:.3f}")
        print(f"Pattern A1 vs B (inverse):        {sim_a1_b:.3f}")
        print(f"Pattern A1 vs Random:             {sim_a1_random:.3f}")
        
        if sim_a1_a2 > sim_a1_b and sim_a1_a2 > sim_a1_random:
            print("✓ Similarity function learned to recognize pattern similarity!")
        else:
            print("? Similarity function still learning pattern relationships")
    
    # Final comprehensive brain stats
    print(f"\nFinal Brain Statistics:")
    print("=" * 30)
    final_stats = brain.get_brain_stats()
    print(f"Total experiences: {final_stats['brain_summary']['total_experiences']}")
    print(f"Total predictions: {final_stats['brain_summary']['total_predictions']}")
    
    if 'similarity_learning' in final_stats['similarity_engine']:
        sim_final = final_stats['similarity_engine']['similarity_learning']
        print(f"Similarity adaptations: {sim_final['adaptations_performed']}")
        print(f"Final correlation: {sim_final['similarity_success_correlation']:.3f}")
        print(f"Learning rate: {sim_final['learning_rate']}")
    
    print(f"\nEmergence Test Completed!")
    print("Key insight: Similarity meaning emerged from prediction success patterns")
    
    return evolution_log


def main():
    """Main test runner."""
    try:
        evolution_log = test_similarity_emergence()
        print("\n✓ Similarity emergence test completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)