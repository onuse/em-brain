#!/usr/bin/env python3
"""
Time-Limited Behavioral Test Framework

Same as behavioral_test_framework.py but with built-in time limits
"""

import sys
import os
from pathlib import Path
import time
import signal
from contextlib import contextmanager

# Add brain server to path
brain_server_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(brain_server_path))

# Import the original framework
from behavioral_test_framework import (
    BehavioralTestFramework, 
    BASIC_INTELLIGENCE_PROFILE,
    ADVANCED_INTELLIGENCE_PROFILE,
    IntelligenceProfile
)

class TimeoutException(Exception):
    pass

@contextmanager
def time_limit(seconds):
    """Context manager to limit execution time"""
    def signal_handler(signum, frame):
        raise TimeoutException(f"Test timed out after {seconds} seconds")
    
    # Set the signal handler and alarm
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    
    try:
        yield
    finally:
        # Disable the alarm
        signal.alarm(0)

class TimedBehavioralTestFramework(BehavioralTestFramework):
    """Extended framework with time limits"""
    
    def __init__(self, quiet_mode: bool = False, max_test_time: int = 60):
        super().__init__(quiet_mode)
        self.max_test_time = max_test_time
        
    def run_intelligence_assessment(self, brain, profile: IntelligenceProfile) -> dict:
        """Run assessment with time limit"""
        print(f"⏱️  Running with {self.max_test_time}s time limit")
        
        try:
            with time_limit(self.max_test_time):
                return super().run_intelligence_assessment(brain, profile)
        except TimeoutException as e:
            print(f"\n⏰ {e}")
            print("Returning partial results...")
            
            # Return partial results
            return {
                'completed': False,
                'timeout': True,
                'time_limit': self.max_test_time,
                'message': str(e)
            }

def main():
    """Run timed behavioral tests"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run timed behavioral tests')
    parser.add_argument('--time-limit', type=int, default=30,
                        help='Time limit in seconds (default: 30)')
    parser.add_argument('--profile', choices=['basic', 'advanced'], default='basic',
                        help='Intelligence profile to test (default: basic)')
    parser.add_argument('--quiet', action='store_true',
                        help='Quiet mode')
    
    args = parser.parse_args()
    
    print(f"🧠 Timed Behavioral Test Framework")
    print(f"⏱️  Time limit: {args.time_limit} seconds")
    print("=" * 60)
    
    # Create framework with time limit
    framework = TimedBehavioralTestFramework(
        quiet_mode=args.quiet,
        max_test_time=args.time_limit
    )
    
    # Create brain
    print("\n🔧 Creating brain...")
    brain = framework.create_brain()
    
    # Select profile
    profile = BASIC_INTELLIGENCE_PROFILE if args.profile == 'basic' else ADVANCED_INTELLIGENCE_PROFILE
    
    # Run assessment
    start_time = time.time()
    results = framework.run_intelligence_assessment(brain, profile)
    elapsed = time.time() - start_time
    
    # Print results
    print(f"\n📊 Test completed in {elapsed:.1f} seconds")
    
    if isinstance(results, dict) and results.get('timeout'):
        print(f"⏰ Test timed out - only partial results available")
    else:
        print(f"✅ All tests completed successfully")
        framework.print_assessment_summary(results, profile)

if __name__ == "__main__":
    main()