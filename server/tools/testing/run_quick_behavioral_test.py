#!/usr/bin/env python3
"""
Quick behavioral test using dynamic brain architecture

This runs a subset of tests that can complete quickly
"""

import sys
import os
from pathlib import Path
import time
import numpy as np

# Add brain server to path
brain_server_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(brain_server_path))

# Use the dynamic test framework
from behavioral_test_dynamic import (
    DynamicBehavioralTestFramework,
    FAST_INTELLIGENCE_PROFILE
)

def main():
    """Run quick behavioral tests"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Quick behavioral test')
    parser.add_argument('--time-per-test', type=int, default=5,
                        help='Time per test in seconds (default: 5)')
    parser.add_argument('--quiet', action='store_true',
                        help='Quiet mode')
    
    args = parser.parse_args()
    
    print(f"🧠 Quick Behavioral Test")
    print(f"⏱️  Time per test: {args.time_per_test} seconds")
    print("=" * 60)
    
    # Create framework
    framework = DynamicBehavioralTestFramework(
        use_simple_brain=False,
        quiet_mode=args.quiet
    )
    
    try:
        # Run assessment with fast profile
        results = framework.run_assessment(FAST_INTELLIGENCE_PROFILE)
        
        # Print summary
        print(f"\n📊 Test Summary")
        print("=" * 60)
        
        for metric, details in results['detailed_results'].items():
            status = "✅" if details['achieved'] else "❌"
            print(f"{status} {metric}: {details['score']:.3f}")
        
        print(f"\nOverall Score: {results['overall_achievement']:.3f}")
        print(f"Total time: {results['total_test_time']:.1f} seconds")
        
    finally:
        # Cleanup
        framework.cleanup()

if __name__ == "__main__":
    main()