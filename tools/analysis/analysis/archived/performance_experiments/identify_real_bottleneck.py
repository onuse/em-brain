#!/usr/bin/env python3
"""
Identify Real Bottleneck

Based on the server logs, identify what's actually causing the 2230% performance degradation.
"""

import sys
import os
import time

# Set up path to access brain modules
brain_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(brain_root, 'server', 'src'))
sys.path.append(os.path.join(brain_root, 'server'))

from src.brain import MinimalBrain

def analyze_server_configuration():
    """Analyze the configuration that causes server performance issues."""
    print("🔍 ANALYZING SERVER CONFIGURATION ISSUES")
    print("=" * 50)
    
    print("Based on server logs, the issues appear to be:")
    print("1. Meta-learning decreasing utility learning rate: 0.010 → 0.009")
    print("2. Performance degradation detected: cycle_time +2230.4%")
    print("3. Checkpoint creation every cycle: checkpoint_000265, checkpoint_000266")
    print("4. Cognitive limits scaling DOWN: WM 16→14, Search 380→342")
    print("5. Cognitive energy scaling DOWN: 19→17")
    
    print("\n🎯 HYPOTHESIS: The issue is NOT our optimizations, but:")
    print("   - Checkpointing happening every cycle instead of every 1000")
    print("   - Aggressive cognitive limit scaling down")
    print("   - Meta-learning getting stuck in poor performance state")
    
    # Test checkpointing frequency issue
    print("\n⚠️  TESTING CHECKPOINT FREQUENCY:")
    
    # Create brain with persistence like the server
    brain = MinimalBrain(
        enable_logging=False,
        enable_persistence=True,  # This is what the server uses
        enable_storage_optimization=True,
        use_utility_based_activation=True,
        quiet_mode=False  # Show what's happening
    )
    
    print(f"   Checkpoint interval: {brain.persistence_manager.config.checkpoint_interval_experiences}")
    print(f"   Checkpoint time interval: {brain.persistence_manager.config.checkpoint_interval_seconds}")
    
    # Test if checkpointing is happening too frequently
    for i in range(10):
        sensory = [0.1 + 0.01 * i, 0.2 + 0.01 * i, 0.3 + 0.01 * i, 0.4 + 0.01 * i]
        
        start_time = time.time()
        predicted_action, brain_state = brain.process_sensory_input(sensory)
        outcome = [a * 0.9 + 0.05 for a in predicted_action]
        brain.store_experience(sensory, predicted_action, outcome, predicted_action)
        cycle_time = (time.time() - start_time) * 1000
        
        print(f"   Cycle {i}: {cycle_time:.1f}ms")
    
    brain.finalize_session()
    
    print("\n💡 RECOMMENDATIONS:")
    print("   1. Check if checkpoint_interval_experiences is set too low")
    print("   2. Disable Phase 2 adaptations if they're causing scaling issues")
    print("   3. Reset meta-learning parameters if stuck in poor state")
    print("   4. Consider running server with enable_persistence=False for testing")

if __name__ == "__main__":
    analyze_server_configuration()