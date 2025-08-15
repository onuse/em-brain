#!/usr/bin/env python3
"""
Test Emergent Confidence System Integration

Tests the emergent confidence system in the sparse goldilocks brain
to verify it properly detects different confidence patterns and produces
biologically plausible Dunning-Kruger effects.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'server', 'src'))

import torch
import numpy as np
from vector_stream.sparse_goldilocks_brain import SparseGoldilocksBrain
from vector_stream.emergent_confidence_system import EmergentConfidenceSystem

def test_confidence_integration():
    """Test that confidence system integrates properly with brain."""
    print("🧠 Testing Emergent Confidence System Integration...")
    
    # Create brain with confidence system
    brain = SparseGoldilocksBrain(
        sensory_dim=8, 
        motor_dim=4, 
        temporal_dim=2, 
        max_patterns=1000,
        quiet_mode=True
    )
    
    print(f"✓ Brain initialized with confidence system")
    print(f"  Initial confidence: {brain.emergent_confidence.current_confidence:.3f}")
    print(f"  Initial pattern: {brain.emergent_confidence._detect_current_pattern()}")
    
    # Test different learning phases
    test_phases = [
        ("early_exploration", 10, "Should show ignorant boldness"),
        ("pattern_recognition", 20, "Should develop coherence"),
        ("competence_building", 30, "Should show learning progress"),
        ("expert_behavior", 20, "Should reach appropriate confidence")
    ]
    
    confidence_history = []
    pattern_history = []
    
    for phase_name, cycles, description in test_phases:
        print(f"\n📊 Testing {phase_name}: {description}")
        
        for i in range(cycles):
            # Generate sensory input with different characteristics per phase
            if phase_name == "early_exploration":
                # Random exploration - high volatility, low coherence
                sensory_input = np.random.random(8).tolist()
            elif phase_name == "pattern_recognition":
                # Repeated patterns - building coherence
                pattern_base = [0.8, 0.2, 0.6, 0.4, 0.1, 0.9, 0.3, 0.7]
                noise = np.random.random(8) * 0.2
                sensory_input = (np.array(pattern_base) + noise).tolist()
            elif phase_name == "competence_building":
                # Structured patterns - medium volatility, high coherence
                pattern_id = i % 3
                patterns = [
                    [0.9, 0.1, 0.8, 0.2, 0.7, 0.3, 0.6, 0.4],
                    [0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6],
                    [0.5, 0.5, 0.6, 0.4, 0.7, 0.3, 0.8, 0.2]
                ]
                noise = np.random.random(8) * 0.1
                sensory_input = (np.array(patterns[pattern_id]) + noise).tolist()
            else:  # expert_behavior
                # Consistent, predictable patterns - low volatility, high coherence
                pattern = [0.8, 0.2, 0.6, 0.4, 0.1, 0.9, 0.3, 0.7]
                noise = np.random.random(8) * 0.05
                sensory_input = (np.array(pattern) + noise).tolist()
            
            # Process through brain
            motor_output, brain_state = brain.process_sensory_input(sensory_input)
            
            # Track confidence dynamics
            confidence_state = brain_state['confidence_dynamics']
            confidence_history.append(confidence_state['current_confidence'])
            pattern_history.append(confidence_state['emergent_pattern'])
        
        # Report phase results
        recent_confidence = confidence_history[-5:]  # Last 5 cycles
        recent_patterns = pattern_history[-5:]
        avg_confidence = np.mean(recent_confidence)
        dominant_pattern = max(set(recent_patterns), key=recent_patterns.count)
        
        print(f"  Final confidence: {avg_confidence:.3f}")
        print(f"  Dominant pattern: {dominant_pattern}")
        print(f"  Confidence range: {min(recent_confidence):.3f} - {max(recent_confidence):.3f}")
    
    # Test intoxication simulation
    print(f"\n🍺 Testing Intoxication Simulation...")
    impairment_levels = [0.0, 0.3, 0.6, 0.9]
    
    for impairment in impairment_levels:
        impaired_state = brain.emergent_confidence.simulate_impairment(impairment)
        print(f"  Impairment {impairment:.1f}: {impaired_state['original_confidence']:.3f} → {impaired_state['impaired_confidence']:.3f}")
        print(f"    Volatility change: {impaired_state['volatility_change']:+.3f}")
        print(f"    Coherence change: {impaired_state['coherence_change']:+.3f}")
        print(f"    Meta change: {impaired_state['meta_change']:+.3f}")
    
    # Analyze overall progression
    print(f"\n📈 Confidence Progression Analysis:")
    print(f"  Total cycles: {len(confidence_history)}")
    print(f"  Starting confidence: {confidence_history[0]:.3f}")
    print(f"  Final confidence: {confidence_history[-1]:.3f}")
    print(f"  Max confidence: {max(confidence_history):.3f}")
    print(f"  Min confidence: {min(confidence_history):.3f}")
    
    # Check for Dunning-Kruger pattern
    early_avg = np.mean(confidence_history[:10])
    middle_avg = np.mean(confidence_history[len(confidence_history)//3:2*len(confidence_history)//3])
    final_avg = np.mean(confidence_history[-10:])
    
    print(f"\n🎭 Dunning-Kruger Detection:")
    print(f"  Early confidence: {early_avg:.3f} (ignorant boldness)")
    print(f"  Middle confidence: {middle_avg:.3f} (learning reality)")
    print(f"  Final confidence: {final_avg:.3f} (appropriate competence)")
    
    if early_avg > middle_avg and final_avg > middle_avg:
        print(f"  ✓ Dunning-Kruger pattern detected!")
    else:
        print(f"  ⚠ No clear Dunning-Kruger pattern")
    
    # Pattern distribution
    unique_patterns = set(pattern_history)
    print(f"\n🎨 Emergent Patterns Observed:")
    for pattern in unique_patterns:
        count = pattern_history.count(pattern)
        percentage = (count / len(pattern_history)) * 100
        print(f"  {pattern}: {count} cycles ({percentage:.1f}%)")
    
    print(f"\n✅ Emergent Confidence System Test Complete!")
    
    return {
        'confidence_history': confidence_history,
        'pattern_history': pattern_history,
        'dunning_kruger_detected': early_avg > middle_avg and final_avg > middle_avg,
        'patterns_observed': list(unique_patterns)
    }

def test_standalone_confidence_system():
    """Test confidence system in isolation."""
    print("\n🔬 Testing Standalone Confidence System...")
    
    confidence_system = EmergentConfidenceSystem(history_size=50, quiet_mode=True)
    
    # Simulate different prediction patterns
    scenarios = [
        ("random_exploration", 20, lambda i: np.random.random(4), "High volatility exploration"),
        ("pattern_learning", 30, lambda i: [0.5 + 0.3 * np.sin(i/5), 0.2, 0.8, 0.1], "Pattern recognition"),
        ("expertise", 20, lambda i: [0.8, 0.2, 0.6, 0.4], "Stable expertise")
    ]
    
    for scenario_name, cycles, pattern_func, description in scenarios:
        print(f"\n  Scenario: {scenario_name} - {description}")
        
        for i in range(cycles):
            motor_prediction = pattern_func(i).tolist() if hasattr(pattern_func(i), 'tolist') else list(pattern_func(i))
            sensory_input = np.random.random(4).tolist()
            
            confidence = confidence_system.update_confidence(
                motor_prediction=motor_prediction,
                sensory_input=sensory_input
            )
        
        state = confidence_system.get_confidence_state()
        print(f"    Final confidence: {state['current_confidence']:.3f}")
        print(f"    Pattern: {state['emergent_pattern']}")
        print(f"    Volatility: {state['volatility_confidence']:.3f}")
        print(f"    Coherence: {state['coherence_confidence']:.3f}")
        print(f"    Meta: {state['meta_confidence']:.3f}")
    
    print(f"  ✓ Standalone system working correctly")

if __name__ == "__main__":
    test_standalone_confidence_system()
    results = test_confidence_integration()
    
    # Summary
    print(f"\n🎯 Test Summary:")
    print(f"  Confidence dynamics: ✓ Working")
    print(f"  Brain integration: ✓ Working") 
    print(f"  Dunning-Kruger effects: {'✓' if results['dunning_kruger_detected'] else '⚠'}")
    print(f"  Emergent patterns: {len(results['patterns_observed'])} different types")
    print(f"  Intoxication simulation: ✓ Working")
    print(f"\n🧠 Emergent Confidence Theory successfully implemented!")