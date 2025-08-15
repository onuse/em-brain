#!/usr/bin/env python3
"""
Test async brain maintenance system
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'server'))

import time
import numpy as np
from src.brain import MinimalBrain
from memory_inspector import MemoryInspector
from src.async_brain_maintenance import BrainState

def test_async_maintenance():
    """Test async maintenance with simulated brain activity"""
    print("🧠 Testing Async Brain Maintenance")
    print("=" * 50)
    
    # Create brain and inspector
    brain = MinimalBrain(brain_type="sparse_goldilocks", quiet_mode=True)
    inspector = MemoryInspector(brain)
    
    # Enable async maintenance
    inspector.enable_async_maintenance()
    
    print("\nSimulating brain activity patterns...")
    print("Watch for brain state changes and consolidation events\n")
    
    # Simulate different activity patterns
    for phase in range(3):
        if phase == 0:
            print("\n📊 Phase 1: High activity (60 seconds)")
            activity_base = 0.8
            duration = 60
        elif phase == 1:
            print("\n😴 Phase 2: Low activity - should trigger quiet consolidation (30 seconds)")
            activity_base = 0.1
            duration = 30
        else:
            print("\n🔥 Phase 3: Extreme activity - should force rest (30 seconds)")
            activity_base = 0.95
            duration = 30
        
        start_time = time.time()
        while time.time() - start_time < duration:
            # Generate activity-appropriate input
            if activity_base > 0.5:
                # High activity - random patterns
                sensory_input = np.random.uniform(0, 1, brain.sensory_dim).tolist()
            else:
                # Low activity - similar patterns with small variation
                base = np.ones(brain.sensory_dim) * 0.5
                noise = np.random.normal(0, 0.05, brain.sensory_dim)
                sensory_input = (base + noise).tolist()
            
            # Process through brain
            brain_output, brain_info = brain.process_sensory_input(sensory_input)
            
            # Capture memory (with gating)
            inspector.capture_memory_snapshot(sensory_input, brain_output, brain_info)
            
            # Update activity level
            activity = activity_base + np.random.uniform(-0.1, 0.1)
            inspector.update_activity_level(activity)
            
            # Check maintenance status periodically
            if int(time.time() - start_time) % 10 == 0 and inspector.async_maintenance:
                stats = inspector.async_maintenance.get_maintenance_stats()
                print(f"\n   Time: {int(time.time() - start_time)}s")
                print(f"   Brain state: {stats['brain_state']}")
                print(f"   Memories: {len(inspector.memory_samples)}")
                print(f"   Memory pressure: {stats['maintenance_pressure']['memory']:.2f}")
                print(f"   Consolidation debt: {stats['maintenance_pressure']['consolidation_debt']:.2f}")
                print(f"   Total pressure: {stats['maintenance_pressure']['total']:.2f}")
            
            # Small delay to simulate real-time processing
            time.sleep(0.1)
    
    # Final report
    print("\n" + "=" * 50)
    print("FINAL REPORT:")
    
    final_stats = inspector.memory_gate.get_statistics()
    print(f"\nMemory Statistics:")
    print(f"  Total experiences: {final_stats['total_experiences']}")
    print(f"  Memories formed: {final_stats['memories_formed']}")
    print(f"  Storage rate: {final_stats['overall_storage_rate']:.1%}")
    
    if inspector.async_maintenance:
        maint_stats = inspector.async_maintenance.get_maintenance_stats()
        print(f"\nMaintenance Statistics:")
        print(f"  Final brain state: {maint_stats['brain_state']}")
        print(f"  Consolidation count: {maint_stats['consolidation_count']}")
        print(f"  Time since quiet: {maint_stats['time_since_quiet']:.1f}s")
    
    # Cleanup
    inspector.disable_async_maintenance()
    
    print("\n✅ Async maintenance test complete!")

if __name__ == "__main__":
    # Set up logging to see maintenance events
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(threadName)s] %(levelname)s: %(message)s'
    )
    
    test_async_maintenance()