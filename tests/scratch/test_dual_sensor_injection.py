#!/usr/bin/env python3
"""
Test dual sensor field injection - Battery + Ultrasonic in parallel.

This demonstrates two different sensors at different frequencies
injecting into different field regions simultaneously.
"""

import sys
import time
import threading
import socket
import struct
import random
import torch

# Add server src to path
sys.path.insert(0, 'server/src')

from streams.field_injection_threads import SensorFieldInjectionManager
from streams.ultrasonic_field_injector import UltrasonicFieldInjector


def simulate_battery(duration: float = 10.0):
    """Send battery updates at 1Hz."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    voltage = 7.6
    
    start = time.time()
    while time.time() - start < duration:
        voltage = max(6.0, voltage - 0.01)  # Slow discharge
        packet = struct.pack('!f', voltage + random.gauss(0, 0.02))
        sock.sendto(packet, ('localhost', 10004))
        time.sleep(1.0)  # 1Hz
    
    sock.close()


def simulate_ultrasonic(duration: float = 10.0):
    """Send ultrasonic updates at 20Hz with moving obstacle."""
    import math
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    
    start = time.time()
    while time.time() - start < duration:
        # Simulate obstacle moving back and forth
        t = time.time() - start
        distance = 50 + 40 * math.sin(t * 0.5)  # Oscillates 10-90cm
        
        packet = struct.pack('!f', distance + random.gauss(0, 1))
        sock.sendto(packet, ('localhost', 10003))
        time.sleep(0.05)  # 20Hz
    
    sock.close()


def main():
    import math
    
    print("=" * 70)
    print("DUAL SENSOR FIELD INJECTION TEST")
    print("=" * 70)
    print("\nTesting parallel injection with multiple sensors:")
    print("  • Battery @ 1Hz → field[0,0,0,-1]")
    print("  • Ultrasonic @ 20Hz → field[0:2,0:2,0:2,28:32]")
    print()
    
    # Create mock brain field
    print("Creating field tensor...")
    field = torch.zeros(16, 16, 16, 64)
    initial_mean = field.mean().item()
    print(f"  Field shape: {field.shape}")
    print(f"  Initial mean: {initial_mean:.4f}")
    
    # Create injection manager (with mock brain)
    print("\nInitializing injection manager...")
    class MockBrain:
        def __init__(self, field):
            self.field = field
            self.levels = [type('Level', (), {'field': field})]
    
    mock_brain = MockBrain(field)
    manager = SensorFieldInjectionManager(mock_brain)
    
    # Start both sensor injectors
    print("\nStarting sensor injectors...")
    started = manager.start_all_basic_sensors()
    for sensor in started:
        print(f"  ✓ {sensor} injector started")
    
    if len(started) != 2:
        print("⚠️  Not all sensors started!")
    
    # Start sensor simulators
    print("\nStarting sensor simulators...")
    battery_thread = threading.Thread(
        target=lambda: simulate_battery(8.0), 
        daemon=True
    )
    ultrasonic_thread = threading.Thread(
        target=lambda: simulate_ultrasonic(8.0),
        daemon=True
    )
    
    battery_thread.start()
    ultrasonic_thread.start()
    print("  ✓ Battery simulator @ 1Hz")
    print("  ✓ Ultrasonic simulator @ 20Hz")
    
    # Monitor field changes
    print("\nMonitoring parallel injection...")
    print("-" * 70)
    print("Time | Battery Region | Ultrasonic Region      | Field Stats")
    print("-" * 70)
    
    for i in range(40):  # 8 seconds at 5Hz monitoring
        # Check battery region
        battery_val = field[0, 0, 0, -1].item()
        
        # Check ultrasonic region
        ultrasonic_region = field[0:2, 0:2, 0:2, 28:32]
        ultrasonic_mean = ultrasonic_region.mean().item()
        ultrasonic_max = ultrasonic_region.max().item()
        
        # Overall field stats
        field_mean = field.mean().item()
        field_std = field.std().item()
        
        # Get injection counts
        stats = manager.get_stats()
        battery_count = stats.get('battery', {}).get('injections', 0)
        ultrasonic_count = stats.get('ultrasonic', {}).get('injections', 0)
        
        print(f"{i*0.2:4.1f}s | "
              f"B:{battery_val:6.3f} ({battery_count:2d}) | "
              f"U:{ultrasonic_mean:6.3f}/{ultrasonic_max:6.3f} ({ultrasonic_count:3d}) | "
              f"μ:{field_mean:7.4f} σ:{field_std:7.4f}")
        
        time.sleep(0.2)  # 5Hz monitoring
    
    # Stop everything
    print("\nStopping injectors...")
    manager.stop_all()
    
    # Final analysis
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    
    final_stats = manager.get_stats()
    
    print("\nInjection statistics:")
    for sensor, stats in final_stats.items():
        print(f"  {sensor}:")
        print(f"    Injections: {stats.get('injections', 0)}")
        print(f"    Last value: {stats.get('last_value', 'N/A')}")
    
    # Check field coherence
    print("\nField coherence check:")
    if torch.isnan(field).any():
        print("  ❌ Field has NaN values!")
    elif torch.isinf(field).any():
        print("  ❌ Field has Inf values!")
    else:
        print("  ✓ No NaN or Inf values")
    
    final_mean = field.mean().item()
    final_std = field.std().item()
    print(f"  Final mean: {final_mean:.4f}")
    print(f"  Final std: {final_std:.4f}")
    
    # Check if both sensors modified their regions
    battery_modified = abs(field[0, 0, 0, -1].item()) > 0.001
    ultrasonic_modified = ultrasonic_region.abs().max().item() > 0.001
    
    if battery_modified and ultrasonic_modified:
        print("\n✅ SUCCESS! Dual sensor parallel injection works!")
        print("  • Battery @ 1Hz modified its region")
        print("  • Ultrasonic @ 20Hz modified its region")
        print("  • Different frequencies, different regions")
        print("  • No synchronization needed!")
        print("  • Field stayed coherent")
    else:
        print("\n⚠️  Not all sensors successfully injected")
        if not battery_modified:
            print("  Battery region not modified")
        if not ultrasonic_modified:
            print("  Ultrasonic region not modified")


if __name__ == "__main__":
    main()