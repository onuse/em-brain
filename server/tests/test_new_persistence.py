#!/usr/bin/env python3
"""
Test New Phase 6.5 Persistent Memory System

Quick validation of the cross-session continuity implementation.
"""

import sys
import os
import tempfile
from pathlib import Path

# Add project root to path  
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from server.src.brain_factory import MinimalBrain


def test_basic_cross_session_persistence():
    """Test that brain state persists across sessions."""
    print("🧠 Testing Phase 6.5 Persistent Memory System...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        config = {
            'memory': {
                'persistent_memory_path': temp_dir,
                'enable_persistence': True
            },
            'brain': {
                'type': 'sparse_goldilocks',
                'sensory_dim': 4,
                'motor_dim': 2
            }
        }
        
        print("   📖 Session 1: Creating fresh brain...")
        brain1 = MinimalBrain(config=config, enable_logging=False, quiet_mode=True)
        
        # Process some inputs to create patterns
        for i in range(10):
            sensory_input = [0.5 + i * 0.1, 0.3, 0.8, 0.2]
            prediction, brain_state = brain1.process_sensory_input(sensory_input)
        
        session1_cycles = brain1.total_cycles
        session1_experiences = brain1.total_experiences
        session1_session_count = brain1.session_count
        
        print(f"      Session: {session1_session_count}")
        print(f"      Cycles: {session1_cycles}")
        print(f"      Experiences: {session1_experiences}")
        
        # Finalize session (should save state)
        brain1.finalize_session()
        
        # Session 2: Load brain and continue
        print("   🌅 Session 2: Loading brain...")
        brain2 = MinimalBrain(config=config, enable_logging=False, quiet_mode=True)
        
        # Verify continuity
        session2_session_count = brain2.session_count
        session2_experiences = brain2.total_experiences
        
        print(f"      Session: {session2_session_count}")
        print(f"      Experiences: {session2_experiences}")
        
        # Process more inputs
        for i in range(5):
            sensory_input = [0.7, 0.4 + i * 0.1, 0.6, 0.9]
            prediction, brain_state = brain2.process_sensory_input(sensory_input)
        
        final_experiences = brain2.total_experiences
        brain2.finalize_session()
        
        print(f"      Final experiences: {final_experiences}")
        
        # Test results
        session_incremented = session2_session_count == session1_session_count + 1
        experiences_preserved = session2_experiences == session1_experiences
        learning_continued = final_experiences > session2_experiences
        
        print(f"\n   📊 Test Results:")
        print(f"      Session incremented: {session_incremented} ({session1_session_count} → {session2_session_count})")
        print(f"      Experiences preserved: {experiences_preserved} ({session1_experiences} → {session2_experiences})")
        print(f"      Learning continued: {learning_continued} ({session2_experiences} → {final_experiences})")
        
        success = session_incremented and experiences_preserved and learning_continued
        
        if success:
            print(f"\n   ✅ Persistent Memory Test PASSED!")
            print(f"      🧠 Cross-session continuity working")
            print(f"      💾 Brain state persistence validated")
        else:
            print(f"\n   ❌ Persistent Memory Test FAILED!")
        
        return success


def main():
    """Run the persistence test."""
    print("🚀 Phase 6.5 Persistent Memory Test")
    print("=" * 50)
    
    success = test_basic_cross_session_persistence()
    
    if success:
        print("\n🎉 Phase 6.5 Persistent Memory system working!")
        print("🧠 Brain now maintains state across sessions")
    else:
        print("\n⚠️ Persistent Memory test failed")
    
    return success


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)