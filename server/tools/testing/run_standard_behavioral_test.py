#!/usr/bin/env python3
"""
Standard behavioral test using dynamic brain architecture

Full behavioral test with standard timing and cycles
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
    BASIC_INTELLIGENCE_PROFILE
)

def main():
    """Run standard behavioral tests"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Standard behavioral test')
    parser.add_argument('--simple', action='store_true',
                        help='Use simple brain implementation')
    parser.add_argument('--quiet', action='store_true',
                        help='Quiet mode')
    
    args = parser.parse_args()
    
    print("🧠 Standard Behavioral Test")
    print("=" * 60)
    
    # Create framework
    framework = DynamicBehavioralTestFramework(
        use_simple_brain=args.simple,
        quiet_mode=args.quiet
    )
    
    try:
        # Run assessment with standard profile
        results = framework.run_assessment(BASIC_INTELLIGENCE_PROFILE)
        
        # Detailed results
        print(f"\n📊 Detailed Test Results")
        print("=" * 60)
        
        for metric, details in results['detailed_results'].items():
            print(f"\n{metric}:")
            print(f"  Score: {details['score']:.3f}")
            print(f"  Target: {details['target']:.3f}")
            print(f"  Achieved: {'Yes' if details['achieved'] else 'No'}")
            print(f"  Test time: {details['test_time_s']:.1f}s")
        
        print(f"\n🏆 Overall Achievement: {results['overall_achievement']:.1%}")
        print(f"⏱️  Total test time: {results['total_test_time']:.1f}s")
        
    finally:
        # Cleanup
        framework.cleanup()

if __name__ == "__main__":
    main()