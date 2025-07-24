#!/usr/bin/env python3
"""Test prediction system using behavioral test framework."""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../server'))

from tools.testing.behavioral_test_framework import (
    BehavioralTestFramework, 
    IntelligenceProfile,
    BehavioralTarget,
    IntelligenceMetric
)

print("=== BEHAVIORAL TEST: PREDICTION ADDICTION SYSTEM ===\n")

# Create test framework
framework = BehavioralTestFramework(quiet_mode=False)

# Define intelligence profile focused on prediction
prediction_profile = IntelligenceProfile(
    name="Prediction-Driven Intelligence",
    targets=[
        BehavioralTarget(
            metric=IntelligenceMetric.PREDICTION_LEARNING,
            target_value=0.7,
            tolerance=0.1,
            description="Should improve prediction accuracy over time"
        ),
        BehavioralTarget(
            metric=IntelligenceMetric.EXPLORATION_EXPLOITATION,
            target_value=0.6,
            tolerance=0.1,
            description="Should balance exploration and exploitation based on confidence"
        ),
        BehavioralTarget(
            metric=IntelligenceMetric.PATTERN_RECOGNITION,
            target_value=0.65,
            tolerance=0.1,
            description="Should recognize and predict patterns"
        ),
        BehavioralTarget(
            metric=IntelligenceMetric.FIELD_STABILIZATION,
            target_value=0.5,
            tolerance=0.1,
            description="Field should stabilize with familiar patterns"
        )
    ]
)

# Run behavioral assessment
print("Running behavioral assessment...")
brain = framework.create_brain({'quiet_mode': True})
results = framework.run_intelligence_assessment(brain, prediction_profile)

print("\n=== ASSESSMENT RESULTS ===")
print(f"Overall Achievement: {results['overall_achievement']:.1%}\n")

for metric, score in results['metric_scores'].items():
    print(f"{metric.value}: {score:.3f}")
    
print("\n=== BEHAVIORAL OBSERVATIONS ===")
if 'behavioral_log' in results:
    for obs in results['behavioral_log'][-5:]:  # Last 5 observations
        print(f"  {obs}")

# Test specific prediction behaviors
print("\n=== PREDICTION-SPECIFIC TESTS ===")

# Test 1: Pattern prediction
print("\n1. Pattern Prediction Test:")
pattern_score = framework.test_prediction_learning(brain, num_cycles=50)
print(f"   Score: {pattern_score:.3f}")

# Test 2: Exploration vs Exploitation
print("\n2. Exploration/Exploitation Test:")
explore_score = framework.test_exploration_exploitation(brain, num_cycles=50)
print(f"   Score: {explore_score:.3f}")

# Get final brain state
final_state = brain._brain._get_field_brain_state(0.0)
print("\n=== FINAL BRAIN STATE ===")
print(f"Prediction confidence: {final_state['prediction_confidence']:.3f}")
print(f"Learning modifier: {final_state['learning_addiction_modifier']:.2f}")
print(f"Intrinsic reward: {final_state['intrinsic_reward']:.3f}")
print(f"Prediction efficiency: {final_state['prediction_efficiency']:.3f}")
print(f"Brain cycles: {final_state['brain_cycles']}")

# Analyze improvement
if len(brain._brain._prediction_confidence_history) > 10:
    early_conf = sum(brain._brain._prediction_confidence_history[:5]) / 5
    late_conf = sum(brain._brain._prediction_confidence_history[-5:]) / 5
    improvement = late_conf - early_conf
    print(f"\nConfidence improvement: {improvement:+.3f}")
    
    if improvement > 0.1:
        print("✓ Brain is learning to predict better!")
    elif improvement < -0.1:
        print("✗ Prediction accuracy declining")
    else:
        print("~ Prediction accuracy stable")

print("\n=== SUMMARY ===")
if results['overall_achievement'] > 0.6:
    print("✓ Prediction addiction system is driving intelligent behavior!")
else:
    print("~ Prediction system needs further tuning")

framework.shutdown()