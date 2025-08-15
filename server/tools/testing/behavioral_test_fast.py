#!/usr/bin/env python3
"""
Fast behavioral test using dynamic brain architecture

Quick test to verify brain behavioral capabilities
"""

import sys
import os
from pathlib import Path

# Add brain server to path
brain_server_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(brain_server_path))

# Use the dynamic test framework
from behavioral_test_dynamic import (
    DynamicBehavioralTestFramework,
    FAST_INTELLIGENCE_PROFILE,
    IntelligenceProfile,
    BehavioralTarget,
    IntelligenceMetric
)

# Define a very fast profile for quick testing
VERY_FAST_PROFILE = IntelligenceProfile(
    name="Very Fast Test",
    targets=[
        BehavioralTarget(
            IntelligenceMetric.COMPUTATIONAL_EFFICIENCY, 0.5, 0.1,
            "Basic timing check", 5
        ),
        BehavioralTarget(
            IntelligenceMetric.PREDICTION_LEARNING, 0.1, 0.05, 
            "Minimal prediction", 10
        )
    ]
)

def main():
    """Run fast behavioral test"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Fast behavioral test")
    parser.add_argument('--very-fast', action='store_true', 
                        help='Run even faster test (5-10 cycles per test)')
    args = parser.parse_args()
    
    print("🧠 Fast Behavioral Test")
    print("=" * 50)
    
    framework = DynamicBehavioralTestFramework(
        use_simple_brain=False,
        quiet_mode=True
    )
    
    try:
        # Choose profile
        profile = VERY_FAST_PROFILE if args.very_fast else FAST_INTELLIGENCE_PROFILE
        
        # Run assessment
        results = framework.run_assessment(profile)
        
        # Quick summary
        print(f"\n✅ Test Complete")
        print(f"Overall Score: {results['overall_achievement']:.1%}")
        
        if results['overall_achievement'] >= 0.6:
            print("🎉 Brain is functioning well!")
        else:
            print("⚠️  Brain may need optimization")
            
    finally:
        framework.cleanup()

if __name__ == "__main__":
    main()