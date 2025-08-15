#!/usr/bin/env python3
"""
Quick test of adjusted emergent system
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'server'))

import time
import numpy as np
from src.brain import MinimalBrain
from memory_inspector import MemoryInspector

def test_emergent_system():
    """Test emergent system with better balance"""
    print("🌊 Testing Adjusted Emergent Memory System")
    print("=" * 50)
    
    # Create brain and inspector
    brain = MinimalBrain(brain_type="sparse_goldilocks", quiet_mode=True)
    inspector = MemoryInspector(brain, use_emergent_gate=True)
    
    print("Simulating 500 frames with mixed activity...")
    
    for i in range(500):
        # Vary novelty to test responsiveness
        if i < 100:
            # Static scene
            base = np.ones(brain.sensory_dim) * 0.5
            noise = np.random.normal(0, 0.02, brain.sensory_dim)
            sensory_input = (base + noise).tolist()
        elif i < 200:
            # Dynamic scene
            sensory_input = np.random.uniform(0, 1, brain.sensory_dim).tolist()
        elif i < 300:
            # Back to static
            base = np.ones(brain.sensory_dim) * 0.7
            noise = np.random.normal(0, 0.03, brain.sensory_dim)
            sensory_input = (base + noise).tolist()
        elif i < 400:
            # High novelty burst
            sensory_input = np.random.uniform(0, 1, brain.sensory_dim).tolist()
        else:
            # Final static period
            base = np.ones(brain.sensory_dim) * 0.3
            noise = np.random.normal(0, 0.01, brain.sensory_dim)
            sensory_input = (base + noise).tolist()
        
        # Process through brain
        brain_output, brain_info = brain.process_sensory_input(sensory_input)
        
        # Capture memory
        snapshot = inspector.capture_memory_snapshot(sensory_input, brain_output, brain_info)
        
        # Progress report
        if (i + 1) % 100 == 0:
            stats = inspector.memory_gate.get_statistics()
            pressure = stats.get('total_pressure', 0)
            storage_rate = stats.get('recent_storage_rate', 0) * 100
            
            print(f"Frame {i + 1}: {len(inspector.memory_samples)} memories, "
                  f"{storage_rate:.0f}% storage, pressure: {pressure:.2f}")
    
    # Final report
    final_stats = inspector.memory_gate.get_statistics()
    print(f"\nFinal Results:")
    print(f"Total memories: {len(inspector.memory_samples)}")
    print(f"Overall storage rate: {final_stats.get('overall_storage_rate', 0)*100:.1f}%")
    print(f"Final pressure: {final_stats.get('total_pressure', 0):.2f}")
    
    if 'pressure_breakdown' in final_stats:
        print("Pressure breakdown:")
        for pressure_type, value in final_stats['pressure_breakdown'].items():
            print(f"  {pressure_type}: {value:.2f}")
    
    inspector.cleanup()
    
    print("\n✅ Test complete!")

if __name__ == "__main__":
    test_emergent_system()