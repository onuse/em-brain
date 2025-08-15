#!/usr/bin/env python3
"""Quick test of field stabilization fix"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tools.testing.behavioral_test_comprehensive import ComprehensiveBehavioralTest

# Create test framework
framework = ComprehensiveBehavioralTest(quiet_mode=True)

try:
    # Setup robot
    framework.setup_robot()
    
    # Run only field stabilization test
    print("Testing field stabilization...")
    score = framework.test_field_stabilization(cycles=50)
    print(f"Field stabilization score: {score:.3f}")
    
    if score > 0.0:
        print("✅ Field stabilization test fixed!")
    else:
        print("❌ Still returning 0")
        
finally:
    framework.cleanup()