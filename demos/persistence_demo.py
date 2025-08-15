#!/usr/bin/env python3
"""
Persistence Demo

Demonstrates how the brain remembers across sessions.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'server'))

import time
import numpy as np
from pathlib import Path
from src.core.dynamic_brain_factory import DynamicBrainFactory
from src.persistence.integrated_persistence import IntegratedPersistence


def run_persistence_demo():
    """
    Demonstrate persistence across multiple "sessions" of the brain.
    """
    
    print("\n🧠 BRAIN PERSISTENCE DEMO")
    print("=" * 60)
    print("This demo shows how the brain remembers across restarts")
    print()
    
    # Use a persistent location (not temp)
    memory_path = "./demo_brain_memory"
    
    # Check if we have existing memory
    memory_dir = Path(memory_path)
    has_memory = memory_dir.exists() and any(memory_dir.glob("brain_state_*.json"))
    
    if has_memory:
        print("📂 Found existing brain memory!")
        response = input("   Continue previous session (c) or start fresh (f)? [c/f]: ")
        if response.lower() == 'f':
            # Clear memory
            import shutil
            shutil.rmtree(memory_path, ignore_errors=True)
            has_memory = False
            print("   Memory cleared - starting fresh")
    
    # Create persistence manager
    persistence = IntegratedPersistence(
        memory_path=memory_path,
        save_interval_cycles=50,
        auto_save=True
    )
    
    # Create brain
    factory = DynamicBrainFactory({
        'use_dynamic_brain': True,
        'use_full_features': True,
        'quiet_mode': False
    })
    
    brain_wrapper = factory.create(
        field_dimensions=None,
        spatial_resolution=4,
        sensory_dim=17,
        motor_dim=4
    )
    
    # Try to recover previous state
    if has_memory:
        print("\n🔄 Recovering previous brain state...")
        if persistence.recover_brain_state(brain_wrapper):
            brain = brain_wrapper.brain
            print(f"\n📊 Recovered state:")
            print(f"   Total brain cycles: {brain.brain_cycles}")
            print(f"   Memory regions: {len(brain.topology_regions)}")
            print(f"   Working memory items: {len(brain.working_memory)}")
            print(f"   Prediction confidence: {brain._current_prediction_confidence:.2f}")
            
            # Show what patterns it remembers
            if brain.brain_cycles > 100:
                print("\n🧠 Testing memory of previous patterns...")
                
                # Pattern A (if it was learned)
                pattern_a = [0.8, 0.2] * 8 + [0.5]
                motors, state = brain.process_robot_cycle(pattern_a)
                print(f"   Pattern A confidence: {state['prediction_confidence']:.2f}")
                
                # Pattern B (if it was learned)
                pattern_b = [0.2, 0.8] * 8 + [0.5]
                motors, state = brain.process_robot_cycle(pattern_b)
                print(f"   Pattern B confidence: {state['prediction_confidence']:.2f}")
    else:
        brain = brain_wrapper.brain
        print("\n🆕 Starting fresh brain")
    
    # Interactive session
    print("\n\n🎮 INTERACTIVE BRAIN SESSION")
    print("-" * 60)
    print("Commands:")
    print("  a - Teach pattern A (high-low alternating)")
    print("  b - Teach pattern B (low-high alternating)")
    print("  r - Random input")
    print("  i - Idle (neutral input)")
    print("  s - Show brain state")
    print("  q - Quit and save")
    print()
    
    patterns = {
        'a': ([0.8, 0.2] * 8 + [0.5], "Pattern A (high-low)"),
        'b': ([0.2, 0.8] * 8 + [0.5], "Pattern B (low-high)"),
        'r': (None, "Random input"),
        'i': ([0.5] * 16 + [0.0], "Idle")
    }
    
    try:
        while True:
            # Get command
            cmd = input("\n> ").lower().strip()
            
            if cmd == 'q':
                break
            elif cmd == 's':
                # Show state
                print(f"\n📊 Brain State:")
                print(f"   Brain cycles: {brain.brain_cycles}")
                print(f"   Field energy: {torch.mean(torch.abs(brain.unified_field)):.4f}")
                print(f"   Memory regions: {len(brain.topology_regions)}")
                print(f"   Prediction confidence: {brain._current_prediction_confidence:.2f}")
                print(f"   Cognitive mode: {brain.cognitive_autopilot.current_mode.value}")
                if hasattr(brain, 'blended_reality'):
                    blend_state = brain.blended_reality.get_blend_state()
                    print(f"   Reality blend: {blend_state['reality_balance']}")
                continue
            elif cmd in patterns:
                pattern_data, pattern_name = patterns[cmd]
                print(f"\n🔄 Processing: {pattern_name}")
                
                # Generate pattern
                if pattern_data is None:  # Random
                    pattern = [np.random.random() for _ in range(16)] + [0.0]
                else:
                    pattern = pattern_data
                
                # Process for a few cycles
                for i in range(10):
                    motors, state = brain.process_robot_cycle(pattern)
                    
                    # Check for auto-save
                    if persistence.check_auto_save(brain_wrapper):
                        print("   💾 Auto-saved!")
                
                print(f"   Confidence: {state['prediction_confidence']:.2f}")
                print(f"   Mode: {state['cognitive_mode']}")
            else:
                print("   Unknown command. Use: a, b, r, i, s, q")
    
    except KeyboardInterrupt:
        print("\n\n⚠️ Interrupted!")
    
    # Final save
    print("\n🔚 Saving final brain state...")
    if persistence.shutdown_save(brain_wrapper):
        print("✅ Brain state saved successfully")
        
        # Show what will be remembered
        print(f"\n📝 Next session will remember:")
        print(f"   - {brain.brain_cycles} cycles of experience")
        print(f"   - {len(brain.topology_regions)} memory regions")
        print(f"   - Current confidence: {brain._current_prediction_confidence:.2f}")
        print(f"\nRun the demo again to continue where you left off!")
    else:
        print("❌ Failed to save brain state")


if __name__ == "__main__":
    import torch  # Import here to ensure it's available
    run_persistence_demo()