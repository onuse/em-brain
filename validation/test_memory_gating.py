#!/usr/bin/env python3
"""
Quick test of memory gating system
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'server'))

import numpy as np
from src.brain import MinimalBrain
from memory_inspector import MemoryInspector

def test_memory_gating():
    """Test memory gating with simulated data"""
    print("🧠 Testing Memory Gating System")
    print("=" * 50)
    
    # Create brain and inspector
    brain = MinimalBrain(brain_type="sparse_goldilocks", quiet_mode=True)
    inspector = MemoryInspector(brain)
    
    # Simulate 200 frames
    for i in range(200):
        # Create sensory input
        if i < 50:
            # Static scene - should form few memories
            base_pattern = np.ones(brain.sensory_dim) * 0.3
            noise = np.random.normal(0, 0.01, brain.sensory_dim)
            sensory_input = (base_pattern + noise).tolist()
        elif i < 100:
            # Dynamic scene - should form more memories
            sensory_input = np.random.uniform(0, 1, brain.sensory_dim).tolist()
        elif i < 150:
            # Back to static - few memories
            base_pattern = np.ones(brain.sensory_dim) * 0.7
            noise = np.random.normal(0, 0.01, brain.sensory_dim)
            sensory_input = (base_pattern + noise).tolist()
        else:
            # Novel patterns - many memories
            sensory_input = np.random.uniform(0, 1, brain.sensory_dim).tolist()
        
        # Process through brain
        brain_output, brain_info = brain.process_sensory_input(sensory_input)
        
        # Capture memory snapshot (with gating)
        snapshot = inspector.capture_memory_snapshot(sensory_input, brain_output, brain_info)
        
        # Print progress
        if (i + 1) % 50 == 0:
            stats = inspector.memory_gate.get_statistics()
            print(f"\nFrame {i + 1}:")
            print(f"  Memories formed: {stats['memories_formed']}")
            print(f"  Storage rate: {stats['recent_storage_rate']:.1%}")
            print(f"  Memory pressure: {stats['memory_pressure']:.2f}")
            print(f"  Avg prediction error: {stats['average_prediction_error']:.3f}")
            print(f"  Avg novelty: {stats['average_novelty']:.3f}")
    
    # Final statistics
    print("\n" + "=" * 50)
    print("FINAL STATISTICS:")
    
    final_stats = inspector.memory_gate.get_statistics()
    print(f"\nMemory Gating:")
    print(f"  Total experiences: {final_stats['total_experiences']}")
    print(f"  Memories formed: {final_stats['memories_formed']}")
    print(f"  Overall storage rate: {final_stats['overall_storage_rate']:.1%}")
    print(f"  Final memory pressure: {final_stats['memory_pressure']:.2f}")
    
    consolidation_stats = inspector.memory_consolidator.get_statistics()
    if consolidation_stats['consolidation_runs'] > 0:
        print(f"\nMemory Consolidation:")
        print(f"  Consolidation runs: {consolidation_stats['consolidation_runs']}")
        print(f"  Memories pruned: {consolidation_stats['total_memories_pruned']}")
        print(f"  Prototypes created: {consolidation_stats['prototypes_created']}")
    
    print(f"\nFinal memory count: {len(inspector.memory_samples)}")
    
    # Analyze memory distribution
    print("\nMemory Distribution:")
    memory_types = {}
    for mem in inspector.memory_samples:
        mem_type = mem.get('memory_type', 'regular')
        memory_types[mem_type] = memory_types.get(mem_type, 0) + 1
    
    for mem_type, count in memory_types.items():
        print(f"  {mem_type}: {count}")
    
    print("\n✅ Memory gating test complete!")

if __name__ == "__main__":
    test_memory_gating()