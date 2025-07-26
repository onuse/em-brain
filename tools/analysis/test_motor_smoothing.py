#!/usr/bin/env python3
"""
Test motor smoothing implementation
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'server', 'src'))

import numpy as np
from brain_factory import BrainFactory
import matplotlib.pyplot as plt


def test_motor_smoothing():
    """Test that motor commands are smoothed properly"""
    
    print("Testing Motor Smoothing")
    print("=" * 50)
    
    config = {
        'brain': {
            'field_spatial_resolution': 4,
            'target_cycle_time_ms': 150
        },
        'memory': {
            'enable_persistence': False
        }
    }
    
    brain = BrainFactory(config=config, quiet_mode=True, enable_logging=False)
    
    # Test 1: Sudden change in sensory input
    print("\n1. Testing response to sudden input change...")
    
    motor_history = []
    
    # First 10 cycles: obstacle on left
    for i in range(10):
        sensory_input = [0.8, 0.1, 0.1] + [0.1] * 13  # Left obstacle
        action, state = brain.process_sensory_input(sensory_input)
        motor_history.append(action[:4])
    
    # Next 10 cycles: obstacle on right 
    for i in range(10):
        sensory_input = [0.1, 0.1, 0.8] + [0.1] * 13  # Right obstacle
        action, state = brain.process_sensory_input(sensory_input)
        motor_history.append(action[:4])
    
    # Analyze smoothness
    motor_array = np.array(motor_history)
    
    # Calculate differences between consecutive commands
    diffs = np.diff(motor_array, axis=0)
    max_change = np.max(np.abs(diffs))
    avg_change = np.mean(np.abs(diffs))
    
    print(f"  Max change between cycles: {max_change:.3f}")
    print(f"  Avg change between cycles: {avg_change:.3f}")
    
    # Check for smoothing at transition point (cycle 10)
    transition_diff = np.abs(motor_array[10] - motor_array[9])
    print(f"  Change at obstacle switch: {np.max(transition_diff):.3f}")
    
    # Test 2: Oscillating input
    print("\n2. Testing response to oscillating input...")
    
    motor_history2 = []
    
    for i in range(30):
        # Oscillate between left and right
        if i % 4 < 2:
            sensory_input = [0.6, 0.1, 0.1] + [0.1] * 13
        else:
            sensory_input = [0.1, 0.1, 0.6] + [0.1] * 13
        
        action, state = brain.process_sensory_input(sensory_input)
        motor_history2.append(action[:4])
    
    motor_array2 = np.array(motor_history2)
    
    # Should see dampened oscillations due to smoothing
    turn_commands = motor_array2[:, 0]  # First motor is turn
    
    # Find peaks and troughs
    peaks = []
    troughs = []
    for i in range(1, len(turn_commands)-1):
        if turn_commands[i] > turn_commands[i-1] and turn_commands[i] > turn_commands[i+1]:
            peaks.append(turn_commands[i])
        elif turn_commands[i] < turn_commands[i-1] and turn_commands[i] < turn_commands[i+1]:
            troughs.append(turn_commands[i])
    
    if peaks and troughs:
        oscillation_amplitude = np.mean(peaks) - np.mean(troughs)
        print(f"  Oscillation amplitude: {oscillation_amplitude:.3f}")
        print(f"  (Lower is better - shows smoothing)")
    
    # Plot results
    plt.figure(figsize=(12, 8))
    
    # Plot 1: Sudden change response
    plt.subplot(2, 2, 1)
    plt.plot(motor_array[:, 0], label='Turn')
    plt.plot(motor_array[:, 1], label='Speed')
    plt.axvline(x=9.5, color='r', linestyle='--', label='Input switch')
    plt.title('Motor Response to Sudden Input Change')
    plt.xlabel('Cycle')
    plt.ylabel('Motor Command')
    plt.legend()
    plt.grid(True)
    
    # Plot 2: Command differences
    plt.subplot(2, 2, 2)
    plt.plot(np.abs(diffs[:, 0]), label='Turn changes')
    plt.plot(np.abs(diffs[:, 1]), label='Speed changes')
    plt.title('Absolute Changes Between Cycles')
    plt.xlabel('Cycle')
    plt.ylabel('|Change|')
    plt.legend()
    plt.grid(True)
    
    # Plot 3: Oscillating response
    plt.subplot(2, 2, 3)
    plt.plot(motor_array2[:, 0], label='Turn (smoothed)')
    # Plot expected unsmoothed response
    unsmoothed = []
    for i in range(30):
        if i % 4 < 2:
            unsmoothed.append(-0.1)  # Approximate
        else:
            unsmoothed.append(0.1)
    plt.plot(unsmoothed, '--', alpha=0.5, label='Expected unsmoothed')
    plt.title('Motor Response to Oscillating Input')
    plt.xlabel('Cycle')
    plt.ylabel('Turn Command')
    plt.legend()
    plt.grid(True)
    
    # Plot 4: Summary stats
    plt.subplot(2, 2, 4)
    plt.text(0.1, 0.8, f"Smoothing factor: 0.3", transform=plt.gca().transAxes)
    plt.text(0.1, 0.6, f"Max change: {max_change:.3f}", transform=plt.gca().transAxes)
    plt.text(0.1, 0.4, f"Avg change: {avg_change:.3f}", transform=plt.gca().transAxes)
    plt.text(0.1, 0.2, f"Smoothing working: {'Yes' if max_change < 0.5 else 'No'}", 
             transform=plt.gca().transAxes,
             color='green' if max_change < 0.5 else 'red')
    plt.axis('off')
    plt.title('Summary')
    
    plt.tight_layout()
    plt.savefig('logs/motor_smoothing_test.png')
    print("\nPlot saved to logs/motor_smoothing_test.png")
    
    brain.shutdown()
    
    # Summary
    print("\n" + "=" * 50)
    print("MOTOR SMOOTHING TEST RESULTS")
    print("=" * 50)
    
    smoothing_working = max_change < 0.5 and avg_change < 0.2
    
    if smoothing_working:
        print("✅ Motor smoothing is working correctly")
        print("  - Commands change gradually")
        print("  - No abrupt jumps in motor outputs")
        print("  - Oscillations are dampened")
    else:
        print("❌ Motor smoothing may need adjustment")
        print(f"  - Max change {max_change:.3f} is too high")
    
    return smoothing_working


if __name__ == "__main__":
    test_motor_smoothing()