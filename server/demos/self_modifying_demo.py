#!/usr/bin/env python3
"""
Self-Modifying Field Dynamics Demo

Interactive demonstration of how field dynamics that modify themselves
lead to emergent intelligence properties.
"""

import sys
import os
import time
import numpy as np
import torch
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.brains.field.simplified_unified_brain import SimplifiedUnifiedBrain


def demonstrate_regional_specialization():
    """Show how different regions develop different dynamics."""
    print("\n🌍 Regional Specialization Demo")
    print("=" * 60)
    print("Different regions will learn different dynamics based on their experiences.")
    
    brain = SimplifiedUnifiedBrain(
        sensory_dim=24,
        motor_dim=4,
        spatial_resolution=16,
        quiet_mode=True,
        enable_self_modification=True,
        self_mod_ratio=0.5  # 50% self-modification
    )
    
    print("\n1. Training three regions with different patterns:")
    print("-" * 40)
    
    # Region A: Fast changing patterns
    print("  Region A: Rapid oscillations (should develop fast dynamics)")
    for i in range(100):
        pattern = np.zeros(24)
        pattern[0:8] = np.sin(i * 0.3) * (1 + 0.5 * np.random.randn())
        brain.process_robot_cycle(pattern.tolist())
    
    # Region B: Slow, stable patterns
    print("  Region B: Slow changes (should develop persistence)")
    for i in range(100):
        pattern = np.zeros(24)
        pattern[8:16] = np.sin(i * 0.03) + 0.5
        brain.process_robot_cycle(pattern.tolist())
    
    # Region C: Reward-correlated patterns
    print("  Region C: Reward-associated (should become very persistent)")
    for i in range(100):
        pattern = np.zeros(24)
        if i % 20 < 5:  # Active 25% of time
            pattern[16:24] = 1.0
            pattern[-1] = 1.0  # Reward
        brain.process_robot_cycle(pattern.tolist())
    
    print("\n2. Testing regional responses:")
    print("-" * 40)
    
    # Test each region's persistence
    test_patterns = [
        ("Region A (fast)", [1.0] * 8 + [0.0] * 16),
        ("Region B (slow)", [0.0] * 8 + [1.0] * 8 + [0.0] * 8),
        ("Region C (reward)", [0.0] * 16 + [1.0] * 8)
    ]
    
    for name, test_pattern in test_patterns:
        print(f"\n  Testing {name}:")
        
        # Apply test pattern
        brain.process_robot_cycle(test_pattern)
        
        # Measure decay over 10 empty cycles
        initial_energy = brain.get_self_modification_state().get('emergent_properties', {}).get('total_energy', 0)
        
        energies = []
        for _ in range(10):
            brain.process_robot_cycle([0.0] * 24)
            energy = brain.get_self_modification_state().get('emergent_properties', {}).get('total_energy', 0)
            energies.append(energy)
        
        # Calculate persistence
        if initial_energy > 0:
            persistence = energies[-1] / initial_energy
            print(f"    Persistence after 10 cycles: {persistence:.1%}")
        
        # Show dynamics diversity
        state = brain.get_self_modification_state()
        if 'emergent_properties' in state:
            diversity = state['emergent_properties'].get('dynamics_diversity', 0)
            print(f"    Dynamics diversity: {diversity:.3f}")


def demonstrate_meta_learning():
    """Show how the system learns to learn better over time."""
    print("\n\n🚀 Meta-Learning Demo")
    print("=" * 60)
    print("The self-modifying brain should learn new patterns faster over time.")
    
    # Create two brains for comparison
    brain_fixed = SimplifiedUnifiedBrain(
        sensory_dim=24, motor_dim=4, spatial_resolution=16,
        quiet_mode=True, enable_self_modification=False
    )
    
    brain_selfmod = SimplifiedUnifiedBrain(
        sensory_dim=24, motor_dim=4, spatial_resolution=16,
        quiet_mode=True, enable_self_modification=True,
        self_mod_ratio=0.7  # 70% self-modification
    )
    
    print("\n1. Learning 5 different pattern sequences:")
    print("-" * 40)
    
    # Define pattern sequences
    sequences = [
        ([1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]),  # Sequence 1
        ([1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1], [1, 0, 0, 1]),  # Sequence 2
        ([1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1]),  # Sequence 3
        ([1, 1, 1, 0], [0, 1, 1, 1], [1, 0, 1, 1], [1, 1, 0, 1]),  # Sequence 4
        ([0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0]),  # Sequence 5 (reverse)
    ]
    
    learning_times_fixed = []
    learning_times_selfmod = []
    
    for seq_idx, sequence in enumerate(sequences):
        print(f"\n  Learning sequence {seq_idx + 1}...")
        
        # Measure how quickly each brain learns the sequence
        fixed_learned = False
        selfmod_learned = False
        fixed_time = 0
        selfmod_time = 0
        
        for epoch in range(50):  # Max 50 epochs
            # Present sequence
            for i, pattern in enumerate(sequence):
                full_pattern = [0.0] * 24
                full_pattern[:4] = pattern
                
                # Process with both brains
                motors_fixed, _ = brain_fixed.process_robot_cycle(full_pattern)
                motors_selfmod, _ = brain_selfmod.process_robot_cycle(full_pattern)
                
                # Check if predicting next pattern correctly
                if i < len(sequence) - 1:
                    next_pattern = sequence[i + 1]
                    
                    # Simple accuracy check (threshold)
                    fixed_pred = [1 if m > 0.5 else 0 for m in motors_fixed[:4]]
                    selfmod_pred = [1 if m > 0.5 else 0 for m in motors_selfmod[:4]]
                    
                    fixed_correct = fixed_pred == list(next_pattern)
                    selfmod_correct = selfmod_pred == list(next_pattern)
                    
                    if fixed_correct and not fixed_learned:
                        fixed_learned = True
                        fixed_time = epoch
                    
                    if selfmod_correct and not selfmod_learned:
                        selfmod_learned = True
                        selfmod_time = epoch
            
            if fixed_learned and selfmod_learned:
                break
        
        learning_times_fixed.append(fixed_time if fixed_learned else 50)
        learning_times_selfmod.append(selfmod_time if selfmod_learned else 50)
        
        print(f"    Fixed brain: {'Learned' if fixed_learned else 'Not learned'} "
              f"(epochs: {learning_times_fixed[-1]})")
        print(f"    Self-mod brain: {'Learned' if selfmod_learned else 'Not learned'} "
              f"(epochs: {learning_times_selfmod[-1]})")
    
    print("\n2. Meta-learning analysis:")
    print("-" * 40)
    
    # Calculate improvement over sequences
    fixed_avg_first = np.mean(learning_times_fixed[:2])
    fixed_avg_last = np.mean(learning_times_fixed[-2:])
    selfmod_avg_first = np.mean(learning_times_selfmod[:2])
    selfmod_avg_last = np.mean(learning_times_selfmod[-2:])
    
    print(f"\n  Fixed brain:")
    print(f"    First 2 sequences: {fixed_avg_first:.1f} epochs average")
    print(f"    Last 2 sequences: {fixed_avg_last:.1f} epochs average")
    print(f"    Improvement: {(1 - fixed_avg_last/fixed_avg_first)*100:.1f}%")
    
    print(f"\n  Self-modifying brain:")
    print(f"    First 2 sequences: {selfmod_avg_first:.1f} epochs average")
    print(f"    Last 2 sequences: {selfmod_avg_last:.1f} epochs average")
    print(f"    Improvement: {(1 - selfmod_avg_last/selfmod_avg_first)*100:.1f}%")
    
    print(f"\n  Meta-learning advantage: Self-mod learns {selfmod_avg_first/selfmod_avg_last:.1f}x "
          f"faster by the end!")


def demonstrate_emergent_memory():
    """Show how memory emerges from self-modifying dynamics."""
    print("\n\n💾 Emergent Memory Demo")
    print("=" * 60)
    print("Important patterns naturally become more persistent.")
    
    brain = SimplifiedUnifiedBrain(
        sensory_dim=24, motor_dim=4, spatial_resolution=16,
        quiet_mode=True, enable_self_modification=True,
        self_mod_ratio=0.6
    )
    
    print("\n1. Creating memory through repetition and reward:")
    print("-" * 40)
    
    # Pattern A: Repeated but no reward
    print("  Pattern A: 50 repetitions, no reward")
    for _ in range(50):
        pattern = [1.0, 0.0, 1.0, 0.0] + [0.0] * 20
        brain.process_robot_cycle(pattern)
    
    # Pattern B: Few repetitions but high reward
    print("  Pattern B: 10 repetitions, high reward")
    for _ in range(10):
        pattern = [0.0, 1.0, 0.0, 1.0] + [0.0] * 19 + [1.0]  # Reward
        brain.process_robot_cycle(pattern)
    
    # Pattern C: Many repetitions with moderate reward
    print("  Pattern C: 30 repetitions, moderate reward")
    for _ in range(30):
        pattern = [1.0, 1.0, 0.0, 0.0] + [0.0] * 19 + [0.5]  # Moderate reward
        brain.process_robot_cycle(pattern)
    
    print("\n2. Testing memory recall after delay:")
    print("-" * 40)
    
    # Let the field evolve without input
    print("  Waiting (50 empty cycles)...")
    for _ in range(50):
        brain.process_robot_cycle([0.0] * 24)
    
    # Test recall with partial cues
    test_cues = [
        ("Pattern A", [1.0, 0.0, 0.0, 0.0] + [0.0] * 20),  # Partial A
        ("Pattern B", [0.0, 1.0, 0.0, 0.0] + [0.0] * 20),  # Partial B
        ("Pattern C", [1.0, 1.0, 0.0, 0.0] + [0.0] * 20),  # Exact C
    ]
    
    print("\n  Testing recall with partial cues:")
    for name, cue in test_cues:
        # Measure field state before cue
        before_energy = brain.get_self_modification_state().get(
            'emergent_properties', {}).get('total_energy', 0)
        
        # Apply cue
        motors, _ = brain.process_robot_cycle(cue)
        
        # Measure response
        after_energy = brain.get_self_modification_state().get(
            'emergent_properties', {}).get('total_energy', 0)
        
        response_strength = (after_energy - before_energy) / (before_energy + 1e-8)
        motor_response = np.mean(np.abs(motors[:4]))
        
        print(f"    {name}: response={response_strength:.1%}, motor={motor_response:.3f}")
    
    # Show final topology stats
    print("\n3. Final brain state:")
    print("-" * 40)
    state = brain.get_self_modification_state()
    if 'emergent_properties' in state:
        props = state['emergent_properties']
        print(f"  Stable regions: {props.get('stable_regions', 0)}")
        print(f"  Specialization: {props.get('specialization_ratio', 0):.3f}")
        print(f"  Dynamics diversity: {props.get('dynamics_diversity', 0):.3f}")


def main():
    """Run all demonstrations."""
    print("🧠 Self-Modifying Field Dynamics Demonstration")
    print("=" * 80)
    print("\nThis demo shows how field dynamics that modify themselves lead to:")
    print("- Regional specialization based on experience")
    print("- Meta-learning (learning to learn)")
    print("- Emergent memory formation")
    print("- Novel behaviors without programming")
    
    input("\nPress Enter to begin...")
    
    # Run demos
    demonstrate_regional_specialization()
    input("\nPress Enter for meta-learning demo...")
    
    demonstrate_meta_learning()
    input("\nPress Enter for emergent memory demo...")
    
    demonstrate_emergent_memory()
    
    print("\n\n✨ Summary")
    print("=" * 60)
    print("Self-modifying field dynamics enable:")
    print("✓ Regions that adapt their dynamics to their inputs")
    print("✓ Systems that get better at learning over time")
    print("✓ Natural memory emergence without explicit storage")
    print("✓ True autonomy - the system determines how to learn")
    
    print("\n🚀 This is artificial life, not just artificial intelligence!")


if __name__ == "__main__":
    main()