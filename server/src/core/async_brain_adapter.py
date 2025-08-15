#!/usr/bin/env python3
"""
Async Brain Adapter - The brain's sensory buffer system.

Each sensor stream writes to its own buffer. The brain reads whatever's 
there on each cycle. No synchronization, just like biology.

This is how it was always supposed to work!
"""

import time
import numpy as np
import threading
from typing import Dict, Any, Optional, Tuple
import torch


class AsyncBrainAdapter:
    """
    Manages per-sensor buffers that get overwritten with latest values.
    Brain reads whatever's there on each cycle - no waiting, no sync.
    """
    
    def __init__(self, expected_dimensions: Dict[str, int]):
        """
        Initialize sensor buffers.
        
        Args:
            expected_dimensions: Expected size for each sensor type
                e.g. {'vision': 307200, 'ultrasonic': 1, 'battery': 1}
        """
        self.expected_dimensions = expected_dimensions
        
        # Latest value buffers - overwritten on arrival
        self.buffers = {}
        self.timestamps = {}
        self.update_counts = {}
        self.locks = {}
        
        # Initialize buffers with defaults
        for sensor_type, dim in expected_dimensions.items():
            self.buffers[sensor_type] = self._get_default_value(sensor_type, dim)
            self.timestamps[sensor_type] = 0
            self.update_counts[sensor_type] = 0
            self.locks[sensor_type] = threading.Lock()
        
        self.start_time = time.time()
        
    def _get_default_value(self, sensor_type: str, dim: int):
        """Get sensible default for each sensor type."""
        if sensor_type == 'battery':
            return 7.4  # Nominal voltage
        elif sensor_type == 'ultrasonic':
            return 100.0  # No obstacle
        elif sensor_type == 'vision':
            return np.ones(dim) * 0.5  # Gray (no signal)
        elif sensor_type == 'imu':
            return np.zeros(dim)  # No motion
        elif sensor_type == 'motor_feedback':
            return np.zeros(dim)  # Centered
        else:
            return np.zeros(dim) if dim > 1 else 0.0
    
    def update_sensor(self, sensor_type: str, data: Any, timestamp: Optional[float] = None):
        """
        Update sensor buffer with latest value.
        Old value is simply overwritten - no queuing!
        
        This happens async whenever sensor data arrives.
        """
        if sensor_type not in self.buffers:
            # New sensor type - add it dynamically
            if isinstance(data, (int, float)):
                dim = 1
            else:
                dim = len(data) if hasattr(data, '__len__') else 1
            
            self.buffers[sensor_type] = data
            self.timestamps[sensor_type] = timestamp or time.time()
            self.update_counts[sensor_type] = 0
            self.locks[sensor_type] = threading.Lock()
            return
        
        # Overwrite buffer with latest value (with lock for thread safety)
        with self.locks[sensor_type]:
            self.buffers[sensor_type] = data
            self.timestamps[sensor_type] = timestamp or time.time()
            self.update_counts[sensor_type] += 1
    
    def get_sensory_snapshot(self) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Get current state of all buffers for brain processing.
        
        Returns whatever's in the buffers RIGHT NOW.
        No waiting, no synchronization.
        
        Returns:
            sensory_vector: Combined sensory data
            staleness: Age of each sensor's data in seconds
        """
        current_time = time.time()
        sensory_parts = []
        staleness = {}
        
        # Read each buffer (quick locks to avoid mid-write reads)
        for sensor_type in sorted(self.buffers.keys()):
            with self.locks[sensor_type]:
                value = self.buffers[sensor_type]
                age = current_time - self.timestamps[sensor_type]
            
            # Flatten to 1D if needed
            if isinstance(value, np.ndarray):
                sensory_parts.append(value.flatten())
            elif isinstance(value, (list, tuple)):
                sensory_parts.append(np.array(value).flatten())
            else:
                sensory_parts.append(np.array([value]))
            
            staleness[sensor_type] = age
        
        # Combine into single vector
        if sensory_parts:
            sensory_vector = np.concatenate(sensory_parts)
        else:
            sensory_vector = np.array([])
        
        return sensory_vector, staleness
    
    def get_sensor_value(self, sensor_type: str) -> Tuple[Any, float]:
        """Get specific sensor's current value and age."""
        with self.locks.get(sensor_type, threading.Lock()):
            value = self.buffers.get(sensor_type, None)
            age = time.time() - self.timestamps.get(sensor_type, 0)
        return value, age
    
    def get_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics about sensor updates."""
        current_time = time.time()
        runtime = current_time - self.start_time
        
        stats = {}
        for sensor_type in self.buffers:
            with self.locks[sensor_type]:
                count = self.update_counts[sensor_type]
                age = current_time - self.timestamps[sensor_type]
            
            stats[sensor_type] = {
                'updates': count,
                'rate_hz': count / runtime if runtime > 0 else 0,
                'age_seconds': age,
                'is_stale': age > 1.0,  # Arbitrary staleness threshold
            }
        
        return stats


class BrainLoopIntegration:
    """
    Shows how the async adapter integrates with the brain loop.
    """
    
    def __init__(self, brain, adapter: AsyncBrainAdapter):
        self.brain = brain
        self.adapter = adapter
        self.running = False
    
    def brain_loop(self, cycle_rate: float = 50.0):
        """
        Main brain loop - reads buffers on each cycle.
        
        No synchronization! Just process whatever's there.
        """
        cycle_time = 1.0 / cycle_rate
        
        while self.running:
            start = time.time()
            
            # Get whatever's in the buffers RIGHT NOW
            sensory_vector, staleness = self.adapter.get_sensory_snapshot()
            
            # Check for critically stale data (optional)
            if staleness.get('ultrasonic', float('inf')) > 0.5:
                # Maybe trigger reflex stop if obstacle sensor is too old
                pass
            
            # Brain processes with whatever it has
            motor_commands = self.brain.process(
                motor_feedback=self.adapter.buffers.get('motor_feedback', np.zeros(3)),
                sensory_input=sensory_vector
            )
            
            # Send motor commands (this stays synchronous for safety)
            self.send_motor_commands(motor_commands)
            
            # Sleep to maintain cycle rate
            elapsed = time.time() - start
            if elapsed < cycle_time:
                time.sleep(cycle_time - elapsed)
    
    def send_motor_commands(self, commands):
        """Send motor commands - this stays on TCP for reliability."""
        pass  # Actual implementation sends to robot


# Example of how it all connects
if __name__ == "__main__":
    print("Async Brain Adapter - Biological Sensor Buffering")
    print("=" * 50)
    
    # Define expected sensors
    expected = {
        'battery': 1,
        'ultrasonic': 1,
        'vision': 307200,
        'imu': 6,
        'motor_feedback': 3
    }
    
    adapter = AsyncBrainAdapter(expected)
    
    # Simulate async sensor updates
    import random
    
    print("\nSimulating async sensor arrivals...")
    
    # Battery at 1Hz
    adapter.update_sensor('battery', 7.35)
    time.sleep(0.05)
    
    # Ultrasonic at 20Hz
    for _ in range(3):
        adapter.update_sensor('ultrasonic', random.uniform(10, 100))
        time.sleep(0.05)
    
    # Vision at 15Hz  
    adapter.update_sensor('vision', np.random.rand(307200) * 0.5)
    
    # Get snapshot for brain
    sensory, staleness = adapter.get_sensory_snapshot()
    
    print(f"\nSensory vector size: {len(sensory)}")
    print("\nStaleness (seconds):")
    for sensor, age in staleness.items():
        print(f"  {sensor}: {age:.3f}s")
    
    # Show stats
    print("\nUpdate statistics:")
    for sensor, stats in adapter.get_stats().items():
        print(f"  {sensor}: {stats['updates']} updates, "
              f"{stats['rate_hz']:.1f} Hz, "
              f"{'STALE' if stats['is_stale'] else 'fresh'}")
    
    print("\nThis is how the brain was always meant to work!")
    print("No synchronization, just read whatever's there.")