#!/usr/bin/env python3
"""
Test field energy dissipation mechanism.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../server/src'))

import torch
import time
from brains.field.core_brain import create_unified_field_brain


def test_energy_accumulation_without_maintenance():
    """Test what happens without maintenance."""
    print("Testing Energy Accumulation (No Maintenance)")
    print("=" * 60)
    
    brain = create_unified_field_brain(
        spatial_resolution=15,
        quiet_mode=True
    )
    
    # Disable maintenance for this test
    brain.maintenance_interval = 999999
    
    energy_history = []
    
    # Run many cycles with varied input
    for i in range(500):
        # Create oscillating input
        input_data = [
            0.5 + 0.3 * torch.sin(torch.tensor(i * 0.1)).item(),
            0.5 + 0.3 * torch.cos(torch.tensor(i * 0.15)).item(),
            0.5 + 0.2 * torch.sin(torch.tensor(i * 0.05)).item()
        ] + [0.5] * 21
        
        action, state = brain.process_robot_cycle(input_data)
        
        if i % 50 == 0:
            energy = state['field_total_energy']
            energy_history.append(energy)
            print(f"  Cycle {i}: energy={energy:.3f}")
    
    # Check if energy increased
    energy_increase = energy_history[-1] / energy_history[0] if energy_history[0] > 0 else 0
    print(f"\nEnergy increased by factor of {energy_increase:.2f}")
    
    if energy_increase > 1.5:
        print("⚠️  WARNING: Significant energy accumulation detected!")
    
    return energy_history


def test_energy_dissipation_with_maintenance():
    """Test energy dissipation with maintenance enabled."""
    print("\n\nTesting Energy Dissipation (With Maintenance)")
    print("=" * 60)
    
    brain = create_unified_field_brain(
        spatial_resolution=15,
        quiet_mode=False  # Show maintenance messages
    )
    
    # Enable frequent maintenance
    brain.maintenance_interval = 50
    brain.field_energy_dissipation_rate = 0.95  # More aggressive for testing
    
    energy_history = []
    maintenance_count = 0
    
    # Run many cycles
    for i in range(500):
        # Create oscillating input
        input_data = [
            0.5 + 0.3 * torch.sin(torch.tensor(i * 0.1)).item(),
            0.5 + 0.3 * torch.cos(torch.tensor(i * 0.15)).item(),
            0.5 + 0.2 * torch.sin(torch.tensor(i * 0.05)).item()
        ] + [0.5] * 21
        
        action, state = brain.process_robot_cycle(input_data)
        
        if i % 50 == 0:
            energy = state['field_total_energy']
            energy_history.append(energy)
            
        # Track maintenance runs
        if brain.brain_cycles % brain.maintenance_interval == 0:
            maintenance_count += 1
    
    print(f"\nMaintenance ran {maintenance_count} times")
    print("\nEnergy history:")
    for i, energy in enumerate(energy_history):
        print(f"  Cycle {i*50}: energy={energy:.3f}")
    
    # Check if energy is stable
    max_energy = max(energy_history)
    min_energy = min(energy_history)
    energy_ratio = max_energy / min_energy if min_energy > 0 else float('inf')
    
    print(f"\nEnergy range: [{min_energy:.3f}, {max_energy:.3f}]")
    print(f"Max/Min ratio: {energy_ratio:.2f}")
    
    if energy_ratio < 2.0:
        print("✅ SUCCESS: Energy remains bounded!")
    else:
        print("⚠️  WARNING: Energy variation too high")
    
    return energy_history


def test_maintenance_performance():
    """Test that maintenance doesn't impact hot path."""
    print("\n\nTesting Maintenance Performance Impact")
    print("=" * 60)
    
    brain = create_unified_field_brain(
        spatial_resolution=10,
        quiet_mode=True
    )
    
    # Measure cycles without maintenance
    brain.maintenance_interval = 999999
    start_time = time.perf_counter()
    
    for i in range(100):
        input_data = [0.5] * 24
        brain.process_robot_cycle(input_data)
    
    no_maintenance_time = (time.perf_counter() - start_time) * 1000
    
    # Reset and measure with maintenance
    brain = create_unified_field_brain(
        spatial_resolution=10,
        quiet_mode=True
    )
    brain.maintenance_interval = 10  # Very frequent
    
    start_time = time.perf_counter()
    
    for i in range(100):
        input_data = [0.5] * 24
        brain.process_robot_cycle(input_data)
    
    with_maintenance_time = (time.perf_counter() - start_time) * 1000
    
    print(f"Time for 100 cycles without maintenance: {no_maintenance_time:.2f}ms")
    print(f"Time for 100 cycles with maintenance: {with_maintenance_time:.2f}ms")
    print(f"Overhead: {((with_maintenance_time - no_maintenance_time) / no_maintenance_time * 100):.1f}%")
    
    # Check individual cycle times
    print("\nChecking hot path impact...")
    brain = create_unified_field_brain(
        spatial_resolution=10,
        quiet_mode=True
    )
    brain.maintenance_interval = 5
    
    cycle_times = []
    for i in range(20):
        start = time.perf_counter()
        brain.process_robot_cycle([0.5] * 24)
        cycle_time = (time.perf_counter() - start) * 1000
        cycle_times.append(cycle_time)
        
        if i % 5 == 4:
            print(f"  Cycle {i+1} (with maintenance): {cycle_time:.2f}ms")
        else:
            print(f"  Cycle {i+1}: {cycle_time:.2f}ms")
    
    # Analyze variance
    avg_normal = sum(ct for i, ct in enumerate(cycle_times) if (i+1) % 5 != 0) / len([ct for i, ct in enumerate(cycle_times) if (i+1) % 5 != 0])
    avg_maintenance = sum(ct for i, ct in enumerate(cycle_times) if (i+1) % 5 == 0) / len([ct for i, ct in enumerate(cycle_times) if (i+1) % 5 == 0])
    
    print(f"\nAverage normal cycle: {avg_normal:.2f}ms")
    print(f"Average maintenance cycle: {avg_maintenance:.2f}ms")
    print(f"Maintenance overhead: {avg_maintenance - avg_normal:.2f}ms")


def plot_energy_comparison(no_maintenance, with_maintenance):
    """Plot energy over time comparison."""
    try:
        import matplotlib.pyplot as plt
        
        cycles = [i * 50 for i in range(len(no_maintenance))]
        
        plt.figure(figsize=(10, 6))
        plt.plot(cycles, no_maintenance, 'r-', label='No Maintenance', linewidth=2)
        plt.plot(cycles, with_maintenance, 'b-', label='With Maintenance', linewidth=2)
        
        plt.xlabel('Cycles')
        plt.ylabel('Field Total Energy')
        plt.title('Field Energy Accumulation: With vs Without Maintenance')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('field_energy_dissipation.png')
        print("\n📊 Plot saved to field_energy_dissipation.png")
        
    except ImportError:
        print("\n(Matplotlib not available for plotting)")


if __name__ == "__main__":
    # Test energy accumulation
    no_maint_energy = test_energy_accumulation_without_maintenance()
    
    # Test energy dissipation
    with_maint_energy = test_energy_dissipation_with_maintenance()
    
    # Test performance impact
    test_maintenance_performance()
    
    # Plot comparison
    plot_energy_comparison(no_maint_energy, with_maint_energy)
    
    print("\n✅ Energy dissipation testing complete!")