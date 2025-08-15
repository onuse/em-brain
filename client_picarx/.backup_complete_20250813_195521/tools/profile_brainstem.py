#!/usr/bin/env python3
"""
Brainstem/Communication Efficiency Profiler

Measures:
1. Event bus overhead
2. Network latency
3. Sensor→Brain→Motor pipeline timing
4. Thread synchronization costs
"""

import sys
import os
import time
import asyncio
import threading
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
import cProfile
import pstats
from io import StringIO

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.brainstem.event_bus import AsyncEventBus, Event, EventType
from src.brainstem.brain_client import BrainServerClient, BrainServerConfig, MessageProtocol
from src.brainstem.sensor_motor_adapter_fixed import PiCarXBrainAdapter
from src.config.brainstem_config import get_config


class BrainstemProfiler:
    """Profile brainstem communication efficiency."""
    
    def __init__(self):
        self.config = get_config()
        self.results = {}
        
    def profile_event_bus(self, num_events: int = 1000):
        """Profile event bus overhead."""
        print(f"\n📊 Profiling Event Bus ({num_events} events)...")
        
        async def run_test():
            bus = AsyncEventBus()
            await bus.start()
            
            # Track delivery times
            delivery_times = []
            
            # Create a subscriber that measures latency
            async def measure_latency(event: Event):
                latency = time.time() - event.timestamp
                delivery_times.append(latency * 1000)  # Convert to ms
            
            bus.subscribe_async(EventType.SENSOR_DATA, measure_latency)
            
            # Send events and measure
            start = time.perf_counter()
            
            for i in range(num_events):
                await bus.emit(Event(
                    type=EventType.SENSOR_DATA,
                    data={'index': i},
                    source='profiler'
                ))
            
            # Wait for all events to be processed
            await asyncio.sleep(0.1)
            
            elapsed = time.perf_counter() - start
            
            await bus.stop()
            
            return {
                'total_time_ms': elapsed * 1000,
                'events_per_second': num_events / elapsed,
                'avg_latency_ms': np.mean(delivery_times) if delivery_times else 0,
                'max_latency_ms': np.max(delivery_times) if delivery_times else 0,
                'min_latency_ms': np.min(delivery_times) if delivery_times else 0,
            }
        
        results = asyncio.run(run_test())
        
        print(f"  Total time: {results['total_time_ms']:.2f}ms")
        print(f"  Throughput: {results['events_per_second']:.0f} events/sec")
        print(f"  Avg latency: {results['avg_latency_ms']:.3f}ms")
        print(f"  Max latency: {results['max_latency_ms']:.3f}ms")
        
        self.results['event_bus'] = results
        return results
    
    def profile_network_protocol(self, num_messages: int = 100):
        """Profile network protocol encoding/decoding."""
        print(f"\n📊 Profiling Network Protocol ({num_messages} messages)...")
        
        protocol = MessageProtocol(self.config)
        
        # Test data
        sensor_vector = [0.5] * 16  # 16 sensors
        
        # Profile encoding
        start = time.perf_counter()
        for _ in range(num_messages):
            encoded = protocol.encode_sensory_input(sensor_vector)
        encode_time = time.perf_counter() - start
        
        # Profile decoding (simulate)
        import struct
        start = time.perf_counter()
        for _ in range(num_messages):
            # Simulate decoding
            msg_type = struct.unpack('!B', encoded[4:5])[0]
            vector_length = struct.unpack('!I', encoded[5:9])[0]
            vector_data = struct.unpack(f'{vector_length}f', encoded[9:])
        decode_time = time.perf_counter() - start
        
        results = {
            'encode_time_ms': (encode_time / num_messages) * 1000,
            'decode_time_ms': (decode_time / num_messages) * 1000,
            'message_size_bytes': len(encoded),
            'throughput_mbps': (len(encoded) * num_messages * 8) / (encode_time * 1_000_000)
        }
        
        print(f"  Encode time: {results['encode_time_ms']:.3f}ms per message")
        print(f"  Decode time: {results['decode_time_ms']:.3f}ms per message")
        print(f"  Message size: {results['message_size_bytes']} bytes")
        print(f"  Throughput: {results['throughput_mbps']:.1f} Mbps")
        
        self.results['protocol'] = results
        return results
    
    def profile_adapter(self, num_cycles: int = 1000):
        """Profile sensor-motor adapter efficiency."""
        print(f"\n📊 Profiling Adapter ({num_cycles} cycles)...")
        
        adapter = PiCarXBrainAdapter(self.config)
        
        # Test data
        raw_sensors = [0.5] * 16
        brain_output = [0.5, 0.0, 0.0, 0.0]
        
        # Profile sensor conversion
        start = time.perf_counter()
        for _ in range(num_cycles):
            brain_input = adapter.sensors_to_brain_input(raw_sensors)
        sensor_time = time.perf_counter() - start
        
        # Profile motor conversion
        start = time.perf_counter()
        for _ in range(num_cycles):
            motor_commands = adapter.brain_output_to_motors(brain_output)
        motor_time = time.perf_counter() - start
        
        results = {
            'sensor_conversion_us': (sensor_time / num_cycles) * 1_000_000,
            'motor_conversion_us': (motor_time / num_cycles) * 1_000_000,
            'total_conversion_us': ((sensor_time + motor_time) / num_cycles) * 1_000_000,
            'max_frequency_hz': num_cycles / (sensor_time + motor_time)
        }
        
        print(f"  Sensor conversion: {results['sensor_conversion_us']:.1f}μs")
        print(f"  Motor conversion: {results['motor_conversion_us']:.1f}μs")
        print(f"  Total conversion: {results['total_conversion_us']:.1f}μs")
        print(f"  Max frequency: {results['max_frequency_hz']:.0f} Hz")
        
        self.results['adapter'] = results
        return results
    
    def profile_full_pipeline(self, num_cycles: int = 100):
        """Profile the complete sensor→brain→motor pipeline."""
        print(f"\n📊 Profiling Full Pipeline ({num_cycles} cycles)...")
        
        adapter = PiCarXBrainAdapter(self.config)
        protocol = MessageProtocol(self.config)
        
        # Simulate full pipeline
        pipeline_times = []
        
        for _ in range(num_cycles):
            start = time.perf_counter()
            
            # 1. Raw sensors
            raw_sensors = [np.random.random() for _ in range(16)]
            
            # 2. Adapt to brain format
            brain_input = adapter.sensors_to_brain_input(raw_sensors)
            
            # 3. Encode for network
            encoded = protocol.encode_sensory_input(brain_input)
            
            # 4. Simulate network transmission (1ms)
            time.sleep(0.001)
            
            # 5. Decode response (simulated)
            brain_output = [0.5, 0.0, 0.0, 0.0]
            
            # 6. Convert to motor commands
            motor_commands = adapter.brain_output_to_motors(brain_output)
            
            elapsed = time.perf_counter() - start
            pipeline_times.append(elapsed * 1000)  # Convert to ms
        
        results = {
            'avg_pipeline_ms': np.mean(pipeline_times),
            'max_pipeline_ms': np.max(pipeline_times),
            'min_pipeline_ms': np.min(pipeline_times),
            'std_pipeline_ms': np.std(pipeline_times),
            'p95_pipeline_ms': np.percentile(pipeline_times, 95),
            'p99_pipeline_ms': np.percentile(pipeline_times, 99)
        }
        
        print(f"  Average: {results['avg_pipeline_ms']:.2f}ms")
        print(f"  Min/Max: {results['min_pipeline_ms']:.2f}ms / {results['max_pipeline_ms']:.2f}ms")
        print(f"  P95/P99: {results['p95_pipeline_ms']:.2f}ms / {results['p99_pipeline_ms']:.2f}ms")
        
        self.results['pipeline'] = results
        return results
    
    def generate_report(self):
        """Generate efficiency report."""
        print("\n" + "=" * 60)
        print("BRAINSTEM EFFICIENCY REPORT")
        print("=" * 60)
        
        if 'event_bus' in self.results:
            print("\n🚌 Event Bus:")
            print(f"  Overhead: {self.results['event_bus']['avg_latency_ms']:.3f}ms per event")
            print(f"  Throughput: {self.results['event_bus']['events_per_second']:.0f} events/sec")
        
        if 'protocol' in self.results:
            print("\n📡 Network Protocol:")
            total_protocol = self.results['protocol']['encode_time_ms'] + self.results['protocol']['decode_time_ms']
            print(f"  Total overhead: {total_protocol:.3f}ms per message")
            print(f"  Message size: {self.results['protocol']['message_size_bytes']} bytes")
        
        if 'adapter' in self.results:
            print("\n🔄 Adapter:")
            print(f"  Total overhead: {self.results['adapter']['total_conversion_us']:.1f}μs")
            print(f"  Max frequency: {self.results['adapter']['max_frequency_hz']:.0f} Hz")
        
        if 'pipeline' in self.results:
            print("\n⚡ Full Pipeline:")
            print(f"  Average latency: {self.results['pipeline']['avg_pipeline_ms']:.2f}ms")
            print(f"  P99 latency: {self.results['pipeline']['p99_pipeline_ms']:.2f}ms")
            
            # Calculate max sustainable frequency
            max_freq = 1000 / self.results['pipeline']['avg_pipeline_ms']
            print(f"  Max sustainable rate: {max_freq:.0f} Hz")
        
        # Overall assessment
        print("\n" + "=" * 60)
        print("ASSESSMENT:")
        print("=" * 60)
        
        total_overhead = 0
        if 'event_bus' in self.results:
            total_overhead += self.results['event_bus']['avg_latency_ms']
        if 'protocol' in self.results:
            total_overhead += self.results['protocol']['encode_time_ms'] + self.results['protocol']['decode_time_ms']
        if 'adapter' in self.results:
            total_overhead += self.results['adapter']['total_conversion_us'] / 1000
        
        print(f"\n📊 Total brainstem overhead: {total_overhead:.2f}ms")
        
        if total_overhead < 1.0:
            print("✅ EXCELLENT: Sub-millisecond overhead!")
        elif total_overhead < 5.0:
            print("✅ GOOD: Low overhead, suitable for real-time control")
        elif total_overhead < 10.0:
            print("⚠️  ACCEPTABLE: Some optimization possible")
        else:
            print("❌ NEEDS OPTIMIZATION: Too much overhead")
        
        # Bottleneck identification
        print("\n🔍 Bottlenecks:")
        components = []
        if 'event_bus' in self.results:
            components.append(('Event Bus', self.results['event_bus']['avg_latency_ms']))
        if 'protocol' in self.results:
            components.append(('Protocol', self.results['protocol']['encode_time_ms'] + self.results['protocol']['decode_time_ms']))
        if 'adapter' in self.results:
            components.append(('Adapter', self.results['adapter']['total_conversion_us'] / 1000))
        
        components.sort(key=lambda x: x[1], reverse=True)
        
        for name, overhead_ms in components[:3]:
            percentage = (overhead_ms / total_overhead) * 100 if total_overhead > 0 else 0
            print(f"  {name}: {overhead_ms:.3f}ms ({percentage:.1f}% of total)")
        
        return self.results


def main():
    """Run complete profiling suite."""
    print("🔬 BRAINSTEM EFFICIENCY PROFILER")
    print("=" * 60)
    
    profiler = BrainstemProfiler()
    
    # Run all profiles
    profiler.profile_adapter()
    profiler.profile_network_protocol()
    profiler.profile_event_bus()
    profiler.profile_full_pipeline()
    
    # Generate report
    profiler.generate_report()
    
    print("\n✅ Profiling complete!")


if __name__ == "__main__":
    main()