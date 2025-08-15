#!/usr/bin/env python3
"""
Simple Brainstem Efficiency Profiler (no async complexity)
"""

import sys
import os
import time
import numpy as np
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.brainstem.brain_client import MessageProtocol
from src.brainstem.sensor_motor_adapter_fixed import PiCarXBrainAdapter
from src.config.brainstem_config import get_config


def profile_brainstem():
    """Profile brainstem components."""
    print("🔬 BRAINSTEM EFFICIENCY PROFILER (Simplified)")
    print("=" * 60)
    
    config = get_config()
    
    # 1. Profile Adapter
    print("\n📊 Adapter Performance:")
    adapter = PiCarXBrainAdapter(config)
    
    raw_sensors = [0.5] * 16
    brain_output = [0.5, 0.0, 0.0, 0.0]
    
    # Warmup
    for _ in range(100):
        adapter.sensors_to_brain_input(raw_sensors)
        adapter.brain_output_to_motors(brain_output)
    
    # Measure
    cycles = 10000
    start = time.perf_counter()
    for _ in range(cycles):
        brain_input = adapter.sensors_to_brain_input(raw_sensors)
    sensor_time = time.perf_counter() - start
    
    start = time.perf_counter()
    for _ in range(cycles):
        motor_commands = adapter.brain_output_to_motors(brain_output)
    motor_time = time.perf_counter() - start
    
    sensor_us = (sensor_time / cycles) * 1_000_000
    motor_us = (motor_time / cycles) * 1_000_000
    total_us = sensor_us + motor_us
    
    print(f"  Sensor conversion: {sensor_us:.1f}μs")
    print(f"  Motor conversion: {motor_us:.1f}μs")
    print(f"  Total: {total_us:.1f}μs ({1_000_000/total_us:.0f} Hz max)")
    
    # 2. Profile Protocol
    print("\n📡 Protocol Performance:")
    protocol = MessageProtocol(config)
    
    sensor_vector = [0.5] * 16
    
    cycles = 10000
    start = time.perf_counter()
    for _ in range(cycles):
        encoded = protocol.encode_sensory_input(sensor_vector)
    encode_time = time.perf_counter() - start
    
    encode_us = (encode_time / cycles) * 1_000_000
    message_size = len(encoded)
    
    print(f"  Encoding: {encode_us:.1f}μs per message")
    print(f"  Message size: {message_size} bytes")
    print(f"  Throughput: {cycles * message_size / (encode_time * 1_000_000):.1f} MB/s")
    
    # 3. Full Pipeline Simulation
    print("\n⚡ Full Pipeline (simulated):")
    
    pipeline_times = []
    
    for _ in range(1000):
        start = time.perf_counter()
        
        # Sensor processing
        raw = [np.random.random() for _ in range(16)]
        brain_in = adapter.sensors_to_brain_input(raw)
        
        # Encoding
        msg = protocol.encode_sensory_input(brain_in)
        
        # Simulate network (0.5ms average)
        time.sleep(0.0005)
        
        # Motor processing
        brain_out = [0.5, 0.0, 0.0, 0.0]
        motors = adapter.brain_output_to_motors(brain_out)
        
        elapsed = (time.perf_counter() - start) * 1000
        pipeline_times.append(elapsed)
    
    avg_ms = np.mean(pipeline_times)
    p99_ms = np.percentile(pipeline_times, 99)
    
    print(f"  Average: {avg_ms:.2f}ms")
    print(f"  P99: {p99_ms:.2f}ms")
    print(f"  Max frequency: {1000/avg_ms:.0f} Hz")
    
    # Summary
    print("\n" + "=" * 60)
    print("EFFICIENCY SUMMARY:")
    print("=" * 60)
    
    brainstem_overhead_us = sensor_us + motor_us + encode_us
    brainstem_overhead_ms = brainstem_overhead_us / 1000
    
    print(f"\n📊 Brainstem overhead: {brainstem_overhead_ms:.3f}ms")
    print(f"   Adapter: {(sensor_us + motor_us)/1000:.3f}ms ({(sensor_us + motor_us)/brainstem_overhead_us*100:.1f}%)")
    print(f"   Protocol: {encode_us/1000:.3f}ms ({encode_us/brainstem_overhead_us*100:.1f}%)")
    
    if brainstem_overhead_ms < 0.1:
        print("\n✅ EXCELLENT: <0.1ms overhead - negligible impact!")
    elif brainstem_overhead_ms < 1.0:
        print("\n✅ GOOD: <1ms overhead - suitable for real-time")
    else:
        print("\n⚠️  Could optimize further, but still acceptable")
    
    print(f"\n🎯 With {avg_ms:.1f}ms total pipeline (including 0.5ms network),")
    print(f"   the system can sustain {1000/avg_ms:.0f} Hz control rate.")
    print(f"   On production hardware (10x faster): ~{10000/avg_ms:.0f} Hz possible!")


if __name__ == "__main__":
    profile_brainstem()