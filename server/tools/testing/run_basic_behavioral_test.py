#!/usr/bin/env python3
"""
Run just the basic intelligence assessment using dynamic brain architecture
"""

import sys
import os
from pathlib import Path

# Add brain server to path
brain_server_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(brain_server_path))

from behavioral_test_dynamic import (
    DynamicBehavioralTestFramework, 
    BASIC_INTELLIGENCE_PROFILE
)

def main():
    """Run basic intelligence assessment only"""
    print("🧠 Running Basic Intelligence Assessment")
    print("=" * 60)
    
    # Create framework
    framework = DynamicBehavioralTestFramework(quiet_mode=True)
    
    try:
        # Run basic intelligence assessment
        results = framework.run_assessment(BASIC_INTELLIGENCE_PROFILE)
        
        # Print summary
        print("\n" + "=" * 60)
        print("📊 ASSESSMENT COMPLETE")
        print("=" * 60)
        
        for metric, details in results['detailed_results'].items():
            score = details['score']
            target = details['target']
            achieved = details['achieved']
            status = "✅ PASS" if achieved else "❌ FAIL"
            print(f"{status} {metric}: {score:.3f} / {target:.3f}")
        
        print(f"\n🏆 Overall Achievement: {results['overall_achievement']:.1%}")
        
    finally:
        # Clean shutdown
        framework.cleanup()
        print("\n✅ Test completed successfully!")


if __name__ == "__main__":
    main()