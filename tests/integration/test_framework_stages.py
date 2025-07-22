#!/usr/bin/env python3
"""
Test Framework in Stages
Test the behavioral framework step by step to isolate remaining issues.
"""

import sys
import time
import signal
import traceback
sys.path.append('server/tools/testing')

def timeout_handler(signum, frame):
    print("🚨 TIMEOUT!")
    traceback.print_stack()
    sys.exit(1)

def test_framework_stages():
    """Test each major component of the behavioral framework."""
    print("🔍 Testing Framework Components")
    print("=" * 50)
    
    try:
        signal.signal(signal.SIGALRM, timeout_handler)
        
        # Test paradigm shifting experiment (was hanging before)
        print("Stage 1: Testing paradigm shifting experiment...")
        signal.alarm(30)
        from behavioral_test_framework import test_paradigm_shifting_experiment
        test_paradigm_shifting_experiment()
        signal.alarm(0)
        print("✅ Paradigm shifting experiment completed")
        
        # Test main section components
        print("\nStage 2: Testing main behavioral framework components...")
        signal.alarm(30)
        from behavioral_test_framework import BehavioralTestFramework, BASIC_INTELLIGENCE_PROFILE
        
        framework = BehavioralTestFramework(quiet_mode=True)
        print("  ✅ Framework created")
        
        brain = framework.create_brain()
        print("  ✅ Brain created")
        
        signal.alarm(60)  # Give more time for intelligence assessment
        print("  Running intelligence assessment...")
        results = framework.run_intelligence_assessment(brain, BASIC_INTELLIGENCE_PROFILE)
        print("  ✅ Intelligence assessment completed")
        
        signal.alarm(0)
        
        print(f"\n🎯 Results:")
        print(f"  Overall achievement: {results['overall_achievement']:.1%}")
        for metric, result in results['detailed_results'].items():
            status = "✅" if result['achieved'] else "❌"
            print(f"  {status} {metric}: {result['score']:.3f}/{result['target']:.3f}")
        
        print("\n🎉 SUCCESS: Full behavioral framework works!")
        
    except Exception as e:
        signal.alarm(0)
        print(f"❌ Error in stage testing: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    test_framework_stages()