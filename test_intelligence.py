#!/usr/bin/env python3
"""
Test for emergent intelligence in the field brain.

Tests whether the brain can:
1. Form discrete concepts from patterns
2. Bind concepts compositionally  
3. Plan sequences of actions
"""

import torch
import numpy as np
import time
from server.src.brains.field.intelligent_field_brain import IntelligentFieldBrain


def test_concept_formation():
    """Test if the brain can form discrete concepts from repeated patterns."""
    print("\n=== Testing Concept Formation ===")
    
    brain = IntelligentFieldBrain(
        sensory_dim=8,
        motor_dim=4,
        spatial_size=16,
        channels=32,
        quiet_mode=False
    )
    
    print(f"\nPresenting distinct patterns to form concepts...")
    
    # Create three distinct patterns
    pattern_A = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]  # Alternating
    pattern_B = [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0]  # Pairs
    pattern_C = [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]  # Split
    
    patterns = [pattern_A, pattern_B, pattern_C]
    pattern_names = ["Alternating", "Pairs", "Split"]
    
    # Present each pattern multiple times
    for epoch in range(10):
        for i, pattern in enumerate(patterns):
            # Present pattern several times to strengthen it
            for _ in range(3):
                motors, telemetry = brain.process(pattern)
            
            if epoch % 3 == 0:
                print(f"  Epoch {epoch}, Pattern {pattern_names[i]}: "
                      f"{telemetry['n_concepts']} concepts formed")
    
    # Check results
    n_concepts = len(brain.attractors.concepts)
    print(f"\nResults:")
    print(f"  Total concepts formed: {n_concepts}")
    print(f"  Active concepts: {len(brain.active_concepts)}")
    
    # Inspect concepts
    for i, concept in enumerate(brain.active_concepts[:5]):
        print(f"  Concept {i}: strength={concept.strength:.2f}, age={concept.age}")
    
    success = n_concepts >= 2  # Should form at least 2 distinct concepts
    print(f"  Concept formation: {'✓ SUCCESS' if success else '✗ FAILED'}")
    
    return success, brain


def test_compositional_binding(brain=None):
    """Test if the brain can bind concepts together."""
    print("\n=== Testing Compositional Binding ===")
    
    if brain is None:
        brain = IntelligentFieldBrain(
            sensory_dim=8,
            motor_dim=4,
            spatial_size=16,
            channels=32,
            quiet_mode=False
        )
    
    print(f"\nTesting concept binding...")
    
    # Present compound pattern (combination of two patterns)
    compound_pattern = [1.0, 0.5, 1.0, 0.5, 0.0, 0.5, 0.0, 0.5]  # Mix
    
    for i in range(20):
        motors, telemetry = brain.process(compound_pattern)
        
        if i % 5 == 0:
            print(f"  Step {i}: {telemetry['n_bindings']} bindings, "
                  f"complexity={telemetry['binding_complexity']}")
    
    # Check if binding occurred
    n_bindings = len(brain.current_bindings)
    max_complexity = brain.binding_complexity
    
    print(f"\nResults:")
    print(f"  Total bindings created: {n_bindings}")
    print(f"  Maximum binding complexity: {max_complexity}")
    
    # Check binding strength between concepts
    if len(brain.active_concepts) >= 2:
        c1, c2 = brain.active_concepts[:2]
        binding_strength = brain.binding.get_binding_strength(c1.position, c2.position)
        print(f"  Binding strength between first two concepts: {binding_strength:.2f}")
    
    success = n_bindings > 0 or max_complexity >= 2
    print(f"  Compositional binding: {'✓ SUCCESS' if success else '✗ FAILED'}")
    
    return success, brain


def test_temporal_planning(brain=None):
    """Test if the brain can plan sequences."""
    print("\n=== Testing Temporal Planning ===")
    
    if brain is None:
        brain = IntelligentFieldBrain(
            sensory_dim=8,
            motor_dim=4,
            spatial_size=16,
            channels=32,
            quiet_mode=False
        )
    
    # Set a goal
    brain.set_goal("explore")
    
    print(f"\nTesting planning with exploration goal...")
    
    planning_events = []
    
    # Run with changing patterns to require planning
    patterns = [
        [0.1] * 8,  # Low activity
        [0.9] * 8,  # High activity
        [0.5] * 8,  # Medium activity
    ]
    
    for i in range(30):
        pattern = patterns[i % len(patterns)]
        motors, telemetry = brain.process(pattern)
        
        if telemetry['planning_confidence'] > 0:
            planning_events.append({
                'step': i,
                'confidence': telemetry['planning_confidence'],
                'horizon': telemetry['planning_horizon']
            })
        
        if i % 10 == 0:
            print(f"  Step {i}: Planning confidence={telemetry['planning_confidence']:.2f}, "
                  f"horizon={telemetry['planning_horizon']}")
    
    print(f"\nResults:")
    print(f"  Total planning events: {len(planning_events)}")
    
    if planning_events:
        avg_confidence = np.mean([e['confidence'] for e in planning_events])
        max_horizon = max(e['horizon'] for e in planning_events)
        print(f"  Average planning confidence: {avg_confidence:.2f}")
        print(f"  Maximum planning horizon: {max_horizon}")
    else:
        avg_confidence = 0
        max_horizon = 0
    
    # Test reasoning
    print(f"\n  Brain's self-report:")
    print(f"    {brain.reason_about('concepts')}")
    print(f"    {brain.reason_about('plan')}")
    print(f"    {brain.reason_about('feeling')}")
    
    success = len(planning_events) > 0 and max_horizon >= 3
    print(f"\n  Temporal planning: {'✓ SUCCESS' if success else '✗ FAILED'}")
    
    return success


def test_problem_solving():
    """Test if the brain can solve a simple problem."""
    print("\n=== Testing Problem Solving ===")
    
    brain = IntelligentFieldBrain(
        sensory_dim=8,
        motor_dim=4,
        spatial_size=16,
        channels=32,
        quiet_mode=True
    )
    
    print(f"\nPresenting a pattern recognition problem...")
    
    # Problem: Learn that pattern A leads to comfort, pattern B leads to discomfort
    pattern_good = [1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0]
    pattern_bad = [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0]
    
    # Training phase
    print("  Training phase...")
    for epoch in range(20):
        # Good pattern - reinforce with low error
        for _ in range(3):
            motors, telemetry = brain.process(pattern_good)
            # Artificially reduce error for good pattern
            brain.selective_persistence.update_stability(brain.field, 0.01)
        
        # Bad pattern - high error
        motors, telemetry = brain.process(pattern_bad)
        brain.selective_persistence.update_stability(brain.field, 0.9)
    
    # Testing phase
    print("  Testing phase...")
    
    # Present good pattern
    good_motors = []
    for _ in range(5):
        motors, telemetry = brain.process(pattern_good)
        good_motors.append(motors)
    
    good_activity = np.mean([sum(abs(m) for m in motor) for motor in good_motors])
    
    # Present bad pattern
    bad_motors = []
    for _ in range(5):
        motors, telemetry = brain.process(pattern_bad)
        bad_motors.append(motors)
    
    bad_activity = np.mean([sum(abs(m) for m in motor) for motor in bad_motors])
    
    print(f"\nResults:")
    print(f"  Motor activity for good pattern: {good_activity:.3f}")
    print(f"  Motor activity for bad pattern: {bad_activity:.3f}")
    print(f"  Difference: {abs(good_activity - bad_activity):.3f}")
    
    # Brain should respond differently to good vs bad patterns
    success = abs(good_activity - bad_activity) > 0.05
    print(f"  Problem solving: {'✓ SUCCESS' if success else '✗ FAILED'}")
    
    return success


def benchmark_intelligence():
    """Benchmark the intelligent brain."""
    print("\n=== Benchmarking Intelligent Brain ===")
    
    brain = IntelligentFieldBrain(
        sensory_dim=24,
        motor_dim=6,
        spatial_size=32,
        channels=64,
        quiet_mode=True
    )
    
    print(f"Brain size: {32**3 * 64:,} parameters")
    print(f"Device: {brain.device}")
    
    # Random sensory input
    sensory_input = [np.random.random() for _ in range(24)]
    
    # Warmup
    for _ in range(10):
        brain.process(sensory_input)
    
    # Benchmark
    times = []
    intelligence_active = []
    
    for i in range(50):
        start = time.time()
        motors, telemetry = brain.process(sensory_input)
        elapsed = (time.time() - start) * 1000
        times.append(elapsed)
        intelligence_active.append(telemetry['intelligence_active'])
        
        if i % 10 == 0:
            print(f"  Cycle {i}: {elapsed:.1f}ms, "
                  f"Concepts: {telemetry['n_concepts']}, "
                  f"Planning: {telemetry['planning_confidence']:.2f}")
    
    avg_time = np.mean(times)
    intelligence_rate = sum(intelligence_active) / len(intelligence_active)
    
    print(f"\nPerformance:")
    print(f"  Average: {avg_time:.2f}ms")
    print(f"  Frequency: {1000/avg_time:.1f}Hz")
    print(f"  Intelligence active: {intelligence_rate:.1%} of cycles")
    
    return 1000/avg_time >= 10  # At least 10Hz with intelligence


if __name__ == "__main__":
    print("=" * 60)
    print("Intelligence Emergence Test Suite")
    print("Testing if true intelligence emerges from field dynamics")
    print("=" * 60)
    
    results = []
    
    # Test concept formation first and pass brain to next test
    success1, brain = test_concept_formation()
    results.append(("Concept Formation", success1))
    
    # Use same brain for binding test
    success2, brain = test_compositional_binding(brain)
    results.append(("Compositional Binding", success2))
    
    # Use same brain for planning test
    success3 = test_temporal_planning(brain)
    results.append(("Temporal Planning", success3))
    
    # Fresh brain for problem solving
    success4 = test_problem_solving()
    results.append(("Problem Solving", success4))
    
    # Benchmark
    success5 = benchmark_intelligence()
    results.append(("Performance", success5))
    
    # Summary
    print("\n" + "=" * 60)
    print("INTELLIGENCE TEST SUMMARY")
    print("=" * 60)
    
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {test_name}: {status}")
    
    all_passed = all(r[1] for r in results)
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🧠✨ TRUE INTELLIGENCE HAS EMERGED!")
        print("The field brain can form concepts, bind them, and plan ahead.")
    else:
        print("🧠💭 Intelligence is emerging but needs refinement.")
        print("The architecture is sound but parameters may need tuning.")
    print("=" * 60)