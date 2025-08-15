#!/usr/bin/env python3
"""
Compare emergent constraint-based vs explicit threshold-based memory systems
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'server'))

import time
import numpy as np
from src.brain import MinimalBrain
from memory_inspector import MemoryInspector

def test_memory_system(use_emergent: bool, test_name: str):
    """Test a memory system with simulated load"""
    print(f"\n{'='*60}")
    print(f"Testing {test_name}")
    print(f"{'='*60}")
    
    # Create brain and inspector
    brain = MinimalBrain(brain_type="sparse_goldilocks", quiet_mode=True)
    inspector = MemoryInspector(brain, use_emergent_gate=use_emergent)
    
    # Simulate realistic load patterns
    results = {
        'processing_times': [],
        'memory_counts': [],
        'storage_rates': [],
        'pressure_stats': []
    }
    
    print("Simulating realistic camera-like load...")
    
    for phase in range(3):
        if phase == 0:
            print(f"\n📹 Phase {phase+1}: Static scene (low novelty)")
            base_novelty = 0.1
            frames = 200
        elif phase == 1:
            print(f"\n🏃 Phase {phase+1}: Dynamic scene (high novelty)")
            base_novelty = 0.8
            frames = 200
        else:
            print(f"\n🎯 Phase {phase+1}: Mixed activity")
            base_novelty = 0.4
            frames = 300
        
        for i in range(frames):
            start_time = time.time()
            
            # Generate sensory input with appropriate novelty
            if base_novelty > 0.5:
                # Dynamic - random patterns
                sensory_input = np.random.uniform(0, 1, brain.sensory_dim).tolist()
            else:
                # Static - similar patterns
                base = np.ones(brain.sensory_dim) * 0.5
                noise_level = base_novelty * 0.2
                noise = np.random.normal(0, noise_level, brain.sensory_dim)
                sensory_input = (base + noise).tolist()
            
            # Process through brain
            brain_output, brain_info = brain.process_sensory_input(sensory_input)
            
            # Capture memory (with gating)
            snapshot = inspector.capture_memory_snapshot(sensory_input, brain_output, brain_info)
            
            processing_time = (time.time() - start_time) * 1000  # ms
            
            # Record metrics
            results['processing_times'].append(processing_time)
            results['memory_counts'].append(len(inspector.memory_samples))
            
            # Get storage rate
            stats = inspector.memory_gate.get_statistics()
            results['storage_rates'].append(stats.get('recent_storage_rate', 0))
            
            # Get pressure stats for emergent system
            if use_emergent and hasattr(stats, 'get') and 'total_pressure' in stats:
                results['pressure_stats'].append(stats['total_pressure'])
            
            # Print progress
            if (i + 1) % 50 == 0:
                avg_time = np.mean(results['processing_times'][-50:])
                current_memories = len(inspector.memory_samples)
                current_rate = stats.get('recent_storage_rate', 0) * 100
                
                print(f"   Frame {phase*200 + i + 1}: {avg_time:.1f}ms, "
                      f"{current_memories} memories, {current_rate:.0f}% storage")
                
                if use_emergent:
                    pressure = stats.get('total_pressure', 0)
                    print(f"      Pressure: {pressure:.2f}")
    
    # Final statistics
    print(f"\n📊 {test_name} Results:")
    print(f"Total frames: {len(results['processing_times'])}")
    print(f"Final memories: {len(inspector.memory_samples)}")
    print(f"Average processing time: {np.mean(results['processing_times']):.1f}ms")
    print(f"Max processing time: {np.max(results['processing_times']):.1f}ms")
    print(f"Overall storage rate: {stats.get('overall_storage_rate', 0)*100:.1f}%")
    
    if use_emergent:
        final_stats = inspector.memory_gate.get_statistics()
        if 'pressure_breakdown' in final_stats:
            print(f"Final pressure breakdown:")
            for pressure_type, value in final_stats['pressure_breakdown'].items():
                print(f"  {pressure_type}: {value:.2f}")
        
        if 'constraint_enforcer' in final_stats:
            enforcer_stats = final_stats['constraint_enforcer']
            print(f"Emergency interventions: {enforcer_stats.get('emergency_interventions', 0)}")
            print(f"Consolidation events: {enforcer_stats.get('consolidation_events', 0)}")
    
    # Cleanup
    inspector.cleanup()
    
    return results

def compare_systems():
    """Compare emergent vs explicit memory systems"""
    print("🧠 Comparing Memory Systems")
    print("Testing constraint-based emergent vs explicit threshold systems")
    
    # Test explicit system first
    explicit_results = test_memory_system(False, "Explicit Threshold System")
    
    # Short delay between tests
    time.sleep(2)
    
    # Test emergent system
    emergent_results = test_memory_system(True, "Emergent Constraint System")
    
    # Comparison
    print(f"\n{'='*60}")
    print("COMPARISON SUMMARY")
    print(f"{'='*60}")
    
    explicit_avg_time = np.mean(explicit_results['processing_times'])
    emergent_avg_time = np.mean(emergent_results['processing_times'])
    
    explicit_max_time = np.max(explicit_results['processing_times'])
    emergent_max_time = np.max(emergent_results['processing_times'])
    
    explicit_final_memories = explicit_results['memory_counts'][-1]
    emergent_final_memories = emergent_results['memory_counts'][-1]
    
    print(f"Average Processing Time:")
    print(f"  Explicit: {explicit_avg_time:.1f}ms")
    print(f"  Emergent: {emergent_avg_time:.1f}ms")
    print(f"  Winner: {'Emergent' if emergent_avg_time < explicit_avg_time else 'Explicit'}")
    
    print(f"\nMax Processing Time:")
    print(f"  Explicit: {explicit_max_time:.1f}ms")
    print(f"  Emergent: {emergent_max_time:.1f}ms")
    print(f"  Winner: {'Emergent' if emergent_max_time < explicit_max_time else 'Explicit'}")
    
    print(f"\nFinal Memory Count:")
    print(f"  Explicit: {explicit_final_memories}")
    print(f"  Emergent: {emergent_final_memories}")
    print(f"  More efficient: {'Emergent' if emergent_final_memories < explicit_final_memories else 'Explicit'}")
    
    print(f"\nPerformance kept under 100ms:")
    explicit_violations = sum(1 for t in explicit_results['processing_times'] if t > 100)
    emergent_violations = sum(1 for t in emergent_results['processing_times'] if t > 100)
    
    print(f"  Explicit violations: {explicit_violations}")
    print(f"  Emergent violations: {emergent_violations}")
    print(f"  Better: {'Emergent' if emergent_violations < explicit_violations else 'Explicit'}")
    
    # Performance over time analysis
    print(f"\nPerformance Degradation Analysis:")
    
    # Check if performance stayed stable
    explicit_early = np.mean(explicit_results['processing_times'][:100])
    explicit_late = np.mean(explicit_results['processing_times'][-100:])
    
    emergent_early = np.mean(emergent_results['processing_times'][:100])
    emergent_late = np.mean(emergent_results['processing_times'][-100:])
    
    print(f"  Explicit: {explicit_early:.1f}ms → {explicit_late:.1f}ms "
          f"({'+' if explicit_late > explicit_early else ''}{explicit_late - explicit_early:.1f}ms)")
    print(f"  Emergent: {emergent_early:.1f}ms → {emergent_late:.1f}ms "
          f"({'+' if emergent_late > emergent_early else ''}{emergent_late - emergent_early:.1f}ms)")
    
    if abs(emergent_late - emergent_early) < abs(explicit_late - explicit_early):
        print(f"  🏆 Emergent system shows better stability!")
    else:
        print(f"  🏆 Explicit system shows better stability!")

if __name__ == "__main__":
    compare_systems()