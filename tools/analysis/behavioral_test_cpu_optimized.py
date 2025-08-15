#!/usr/bin/env python3
"""
CPU-Optimized Behavioral Test Framework
Reduced cycle counts and spatial resolution for practical M1 Mac development.
"""

import sys
import os
sys.path.append('server/tools/testing')

from behavioral_test_framework import BehavioralTestFramework, IntelligenceProfile, BehavioralTarget, IntelligenceMetric

# CPU-optimized intelligence profile with reduced cycle counts
CPU_OPTIMIZED_PROFILE = IntelligenceProfile(
    name="CPU Optimized Intelligence",
    targets=[
        BehavioralTarget(IntelligenceMetric.PREDICTION_LEARNING, 0.3, 0.05, 
                        "Brain should improve predictions over time", 30),  # Reduced from 100
        BehavioralTarget(IntelligenceMetric.EXPLORATION_EXPLOITATION, 0.5, 0.1,
                        "Brain should balance exploration and exploitation", 60),  # Reduced from 200
        BehavioralTarget(IntelligenceMetric.COMPUTATIONAL_EFFICIENCY, 0.7, 0.1,
                        "Brain should process efficiently", 20)  # Reduced from 50
    ]
)

def main():
    """Run CPU-optimized behavioral test."""
    print("🧠 CPU-Optimized Behavioral Test Framework")
    print("Reduced cycles for M1 Mac development")
    print("=" * 50)
    
    framework = BehavioralTestFramework(quiet_mode=True)
    
    # CPU-optimized brain configuration
    config = {
        'brain': {
            'type': 'field', 
            'sensory_dim': 16, 
            'motor_dim': 4,
            'spatial_resolution': 6  # Reduced from default 20
        },
        'memory': {'enable_persistence': False}  # Disable for speed
    }
    
    brain = framework.create_brain(config)
    
    print(f"🧪 Testing with {config['brain']['spatial_resolution']}³ field resolution")
    
    # Run optimized assessment
    results = framework.run_intelligence_assessment(brain, CPU_OPTIMIZED_PROFILE)
    
    # Summary
    achievement = results['overall_achievement']
    print(f"\n🎯 CPU-Optimized Assessment Complete!")
    print(f"Overall Intelligence Achievement: {achievement:.1%}")
    
    if achievement > 0.7:
        print("🎉 EXCELLENT: Brain performing well on CPU!")
    elif achievement > 0.5:
        print("✅ GOOD: Brain functional for development")
    else:
        print("⚠️ PARTIAL: Consider further optimization")
    
    # Show individual results
    print(f"\n📊 Detailed Results:")
    for metric_name, result in results['detailed_results'].items():
        status = "✅" if result['achieved'] else "❌"
        print(f"  {status} {metric_name}: {result['score']:.3f} / {result['target']:.3f}")
    
    return achievement > 0.5

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)