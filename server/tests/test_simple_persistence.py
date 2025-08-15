#!/usr/bin/env python3
"""
Simple Persistence Test

Test basic functionality to isolate issues.
"""

import sys
import os
import tempfile

# Add project root to path  
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from server.src.brain_factory import MinimalBrain


def test_simple_persistence():
    """Test basic persistence functionality."""
    print("🧠 Testing Simple Persistence...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        config = {
            'memory': {
                'persistent_memory_path': temp_dir,
                'enable_persistence': True,
                'save_interval_cycles': 5
            },
            'brain': {
                'type': 'sparse_goldilocks',
                'sensory_dim': 4,
                'motor_dim': 2
            }
        }
        
        print("📖 Session 1: Create brain...")
        brain1 = MinimalBrain(config=config, enable_logging=False, quiet_mode=True)
        
        print(f"   Brain created:")
        print(f"      Session count: {brain1.session_count}")
        print(f"      Architecture: {brain1.brain_type}")
        print(f"      Dimensions: {brain1.sensory_dim}D → {brain1.motor_dim}D")
        
        # Process a few cycles
        for i in range(10):
            sensory_input = [0.1 * i, 0.2 * i, 0.3 * i, 0.4 * i]
            prediction, brain_state = brain1.process_sensory_input(sensory_input)
        
        print(f"   After processing:")
        print(f"      Cycles: {brain1.total_cycles}")
        print(f"      Experiences: {brain1.total_experiences}")
        
        brain1.finalize_session()
        
        print("🌅 Session 2: Load brain...")
        brain2 = MinimalBrain(config=config, enable_logging=False, quiet_mode=True)
        
        print(f"   Brain loaded:")
        print(f"      Session count: {brain2.session_count}")
        print(f"      Experiences: {brain2.total_experiences}")
        
        # Test result
        success = (
            brain2.session_count == 2 and
            brain2.total_experiences >= brain1.total_experiences
        )
        
        brain2.finalize_session()
        
        print(f"🎯 Result: {'✅ PASSED' if success else '❌ FAILED'}")
        print(f"   Session progression: {brain1.session_count} → {brain2.session_count}")
        print(f"   Experience preservation: {brain1.total_experiences} → {brain2.total_experiences}")
        
        return success


if __name__ == "__main__":
    success = test_simple_persistence()
    exit(0 if success else 1)