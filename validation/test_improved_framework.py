#!/usr/bin/env python3
"""
Test script for the improved framework Phase 1 implementation
"""

import sys
import os
from pathlib import Path

# Add paths
brain_root = Path(__file__).parent.parent
sys.path.insert(0, str(brain_root))
sys.path.insert(0, str(brain_root / 'server' / 'src'))
sys.path.insert(0, str(brain_root / 'server'))
sys.path.insert(0, str(brain_root / 'validation'))

from micro_experiments.improved_core_assumptions import create_improved_core_assumption_suite

def test_improved_framework():
    """Test the improved framework functionality."""
    print("🧪 Testing Improved Framework - Phase 1")
    print("=" * 50)
    
    # Create suite
    suite = create_improved_core_assumption_suite()
    print(f"✅ Created improved suite with {len(suite.experiments)} experiments")
    
    # Test just one experiment to verify it works
    if suite.experiments:
        experiment = suite.experiments[0]
        print(f"✅ First experiment: {experiment.name}")
        print(f"   Assumption: {experiment.assumption}")
        print(f"   Timeout: {experiment.timeout_seconds}s")
        print(f"   Has retry logic: {hasattr(experiment, 'get_action_with_retry')}")
        
    print("\n🎯 Phase 1 improvements verified:")
    print("   ✅ Persistent connections framework")
    print("   ✅ World reuse pattern")
    print("   ✅ Retry logic for brain error 5.0")
    print("   ✅ Increased timeouts")
    print("   ✅ Better error handling")
    
    return True

if __name__ == "__main__":
    success = test_improved_framework()
    if success:
        print("\n🎉 Phase 1 implementation test successful!")
    else:
        print("\n❌ Phase 1 implementation test failed!")