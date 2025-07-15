#!/usr/bin/env python3
"""
Test Quiet Brain Initialization

Test the brain with quiet_mode=True to see the reduced startup verbosity.
"""

import sys
import os

# Set up path to access brain modules
brain_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(brain_root, 'server', 'src'))
sys.path.append(os.path.join(brain_root, 'server'))

from src.brain import MinimalBrain

def test_quiet_mode():
    """Test brain initialization with quiet mode."""
    print("🔇 Testing quiet mode initialization...")
    
    brain = MinimalBrain(
        enable_logging=False,
        enable_persistence=False,
        enable_storage_optimization=True,
        use_utility_based_activation=True,
        enable_phase2_adaptations=True,
        quiet_mode=True
    )
    
    # Test a few cycles to confirm it works
    for i in range(5):
        sensory = [0.1 * i, 0.2 * i, 0.3 * i, 0.4 * i]
        predicted_action, brain_state = brain.process_sensory_input(sensory)
        outcome = [a * 0.9 + 0.05 for a in predicted_action]
        brain.store_experience(sensory, predicted_action, outcome, predicted_action)
    
    print(f"✅ Quiet mode test complete: {len(brain.experience_storage._experiences)} experiences stored")
    brain.finalize_session()

if __name__ == "__main__":
    test_quiet_mode()