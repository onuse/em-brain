#!/usr/bin/env python3
"""
Standard behavioral test in quiet mode using dynamic brain architecture
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
    """Run standard behavioral test in quiet mode"""
    print("🧠 Standard Behavioral Test (Quiet Mode)")
    print("=" * 60)
    
    # Create framework in quiet mode
    framework = DynamicBehavioralTestFramework(
        use_simple_brain=False,
        quiet_mode=True
    )
    
    try:
        # Run assessment with standard profile
        results = framework.run_assessment(BASIC_INTELLIGENCE_PROFILE)
        
        # Minimal output
        print("\n📊 Results:")
        for metric, details in results['detailed_results'].items():
            status = "✅" if details['achieved'] else "❌"
            print(f"{status} {metric}: {details['score']:.3f}")
        
        print(f"\nOverall: {results['overall_achievement']:.1%}")
        
    finally:
        # Cleanup
        framework.cleanup()

if __name__ == "__main__":
    main()