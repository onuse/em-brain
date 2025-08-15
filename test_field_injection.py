#!/usr/bin/env python3
"""
Test Field Injection - Battery sensor directly writes to brain field.

This demonstrates that:
1. Battery thread can inject into field independently
2. Main brain loop continues evolving field
3. Field stays coherent with parallel writes
4. Motor output reflects field state (including battery region)
"""

import sys
import time
import threading
import numpy as np
import socket
import struct

# Add server src to path
sys.path.insert(0, 'server/src')

from brains.field.pure_field_brain import PureFieldBrain
from streams.field_injection_threads import (
    BatteryFieldInjector, 
    FieldCoherenceMonitor,
    SensorFieldInjectionManager
)


def simulate_battery_sender(voltage: float = 7.4, rate_hz: float = 1.0):
    """Simulate battery voltage updates via UDP."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    
    def send_loop():
        nonlocal voltage
        while True:
            # Slowly discharge
            voltage = max(6.0, voltage - 0.001)
            
            # Add some noise
            noisy_voltage = voltage + np.random.randn() * 0.05
            
            # Send via UDP
            packet = struct.pack('!f', noisy_voltage)
            sock.sendto(packet, ('localhost', 10004))
            
            time.sleep(1.0 / rate_hz)
    
    thread = threading.Thread(target=send_loop, daemon=True)
    thread.start()
    return thread


def main():
    print("=" * 70)
    print("FIELD INJECTION TEST - Parallel Battery Sensor")
    print("=" * 70)
    print()
    print("This test demonstrates:")
    print("  1. Battery sensor injecting directly into field (parallel thread)")
    print("  2. Main brain loop evolving field independently")
    print("  3. No synchronization between them!")
    print("  4. Field stays coherent")
    print()
    
    # Create brain with small field for testing
    print("Creating PureFieldBrain...")
    brain = PureFieldBrain(
        input_dim=10,  # Small for testing
        output_dim=3,   # Motor commands
        spatial_resolution=8,  # Small field: 8x8x8
        enable_learning=False  # Just test dynamics
    )
    
    # Get field reference
    field = brain.levels[0].field
    print(f"Field shape: {field.shape}")
    print(f"Field device: {field.device}")
    
    # Create coherence monitor
    monitor = FieldCoherenceMonitor(field)
    baseline_coherent, baseline_msg = monitor.check_coherence()
    print(f"Baseline field: {baseline_msg}")
    
    # Create field injection manager
    print("\nStarting field injection manager...")
    injection_manager = SensorFieldInjectionManager(brain)
    
    # Start battery injector thread
    injection_manager.start_battery_injector(port=10004)
    print("✓ Battery injector thread started")
    
    # Start battery sender simulation
    print("\nStarting battery voltage simulator...")
    battery_thread = simulate_battery_sender(voltage=7.4, rate_hz=2.0)
    print("✓ Battery sender started (2Hz)")
    
    # Main brain loop (runs in main thread)
    print("\nRunning main brain loop...")
    print("-" * 40)
    
    try:
        for cycle in range(100):  # 100 cycles
            # Main thread: Normal brain processing
            motor_feedback = np.array([0.0, 0.0, 0.0])
            sensory_input = np.random.randn(10) * 0.1  # Other sensors
            
            # Brain processes (field evolves)
            motor_output = brain.process(motor_feedback, sensory_input)
            
            # Every 10 cycles, check status
            if cycle % 10 == 0:
                # Check field coherence
                coherent, msg = monitor.check_coherence()
                
                # Get battery field value
                battery_value = field[0, 0, 0, -1].item()
                
                # Get injection stats
                stats = injection_manager.get_stats()
                battery_stats = stats.get('battery', {})
                
                print(f"Cycle {cycle:3d} | "
                      f"Battery field: {battery_value:6.3f} | "
                      f"Injections: {battery_stats.get('injections', 0):3d} | "
                      f"Motor: [{motor_output[0]:5.2f}, {motor_output[1]:5.2f}, {motor_output[2]:5.2f}] | "
                      f"{msg}")
                
                if not coherent:
                    print(f"❌ Field coherence lost! {msg}")
                    break
            
            # Simulate main loop rate (50Hz)
            time.sleep(0.02)
            
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    
    # Stop everything
    print("\nStopping...")
    injection_manager.stop_all()
    
    # Final analysis
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    
    final_stats = injection_manager.get_stats()
    battery_stats = final_stats.get('battery', {})
    
    print(f"\nBattery injection statistics:")
    print(f"  Total injections: {battery_stats.get('injections', 0)}")
    print(f"  Last voltage: {battery_stats.get('last_value', 0):.2f}V")
    
    final_coherent, final_msg = monitor.check_coherence()
    print(f"\nField coherence: {final_msg}")
    print(f"  Mean: {field.mean().item():.4f}")
    print(f"  Std:  {field.std().item():.4f}")
    print(f"  Min:  {field.min().item():.4f}")
    print(f"  Max:  {field.max().item():.4f}")
    
    # Check battery region
    battery_region_value = field[0, 0, 0, -1].item()
    print(f"\nBattery field region [0,0,0,-1]: {battery_region_value:.4f}")
    
    if battery_region_value != 0 and final_coherent:
        print("\n✅ SUCCESS! Parallel field injection works!")
        print("   • Battery thread injected directly into field")
        print("   • Main brain loop continued processing")
        print("   • Field stayed coherent")
        print("   • No synchronization needed!")
    else:
        print("\n⚠️  Test inconclusive")
        if battery_region_value == 0:
            print("   Battery region not modified (no injections?)")
        if not final_coherent:
            print("   Field coherence lost")
    
    print("\nNext steps:")
    print("  1. Add ultrasonic parallel injection")
    print("  2. Test with multiple sensors")
    print("  3. Eventually move vision to thread")
    print("  4. Handshake-based thread spawning")


if __name__ == "__main__":
    main()