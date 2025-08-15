#!/usr/bin/env python3
"""
Async Brain Loop - How the brain SHOULD work with streams.

The brain checks buffers on each cycle. No synchronization.
Sensors write whenever they have data. Brain reads whatever's there.

This is the biologically-inspired approach!
"""

import time
import numpy as np
from typing import Optional
from core.async_brain_adapter import AsyncBrainAdapter
from streams.sensor_listeners import StreamManager


class AsyncBrainLoop:
    """
    Brain loop that reads from async sensor buffers.
    
    This is how it was always supposed to work:
    - Sensors write to buffers whenever they have data
    - Brain reads buffers on each cycle
    - No waiting, no synchronization
    """
    
    def __init__(self, brain, cycle_rate: float = 50.0):
        self.brain = brain
        self.cycle_rate = cycle_rate
        self.cycle_time = 1.0 / cycle_rate
        
        # Setup async adapter with expected sensors
        self.adapter = AsyncBrainAdapter({
            'battery': 1,
            'ultrasonic': 1, 
            'vision': 307200,  # Or whatever resolution
            'imu': 6,
            'motor_feedback': 3
        })
        
        # Setup stream listeners that write to adapter
        self.stream_manager = StreamManager(adapter=self.adapter)
        
        self.running = False
        self.cycle_count = 0
        self.motor_commands = np.zeros(3)  # Last motor commands
        
    def start(self):
        """Start stream listeners and brain loop."""
        print("🧠 Starting async brain loop")
        print(f"   Cycle rate: {self.cycle_rate} Hz")
        print(f"   Cycle time: {self.cycle_time*1000:.1f} ms")
        
        # Start sensor stream listeners
        self.stream_manager.start_all()
        print("   ✓ Stream listeners started")
        
        # Start brain loop
        self.running = True
        self._run_loop()
    
    def stop(self):
        """Stop everything."""
        self.running = False
        self.stream_manager.stop_all()
        print("🧠 Brain loop stopped")
    
    def _run_loop(self):
        """
        Main brain loop - the heart of the system.
        
        On each cycle:
        1. Read whatever's in sensor buffers
        2. Process with brain
        3. Send motor commands
        4. Sleep to maintain rate
        """
        print("\n🔄 Brain loop running...")
        print("   Sensors update async → Brain reads buffers → Motors out")
        
        last_stats_time = time.time()
        
        while self.running:
            cycle_start = time.time()
            
            # 1. Get sensory snapshot (whatever's in buffers NOW)
            sensory_vector, staleness = self.adapter.get_sensory_snapshot()
            
            # 2. Check for critical staleness (optional safety)
            if staleness.get('ultrasonic', float('inf')) > 0.5:
                # Ultrasonic too old? Maybe slow down
                self.motor_commands *= 0.9
            
            # 3. Update motor feedback in buffer
            self.adapter.update_sensor('motor_feedback', self.motor_commands)
            
            # 4. Brain processes with whatever it has
            try:
                # Brain expects (motor_feedback, sensory_input)
                motor_feedback = self.adapter.get_sensor_value('motor_feedback')[0]
                
                # Process cycle
                self.motor_commands = self.brain.process(
                    motor_feedback=motor_feedback,
                    sensory_input=sensory_vector
                )
                
                self.cycle_count += 1
                
            except Exception as e:
                print(f"⚠️  Brain error: {e}")
                # Safe fallback
                self.motor_commands = np.zeros(3)
            
            # 5. Send motor commands (TCP - needs reliability)
            self._send_motor_commands(self.motor_commands)
            
            # 6. Print stats every 5 seconds
            if time.time() - last_stats_time > 5.0:
                self._print_stats(staleness)
                last_stats_time = time.time()
            
            # 7. Sleep to maintain cycle rate
            elapsed = time.time() - cycle_start
            if elapsed < self.cycle_time:
                time.sleep(self.cycle_time - elapsed)
            elif elapsed > self.cycle_time * 2:
                print(f"⚠️  Cycle took {elapsed*1000:.1f}ms (target: {self.cycle_time*1000:.1f}ms)")
    
    def _send_motor_commands(self, commands):
        """Send motor commands to robot (placeholder)."""
        # In real system, this sends via TCP to robot
        pass
    
    def _print_stats(self, staleness):
        """Print statistics about sensor updates."""
        print(f"\n📊 Cycle {self.cycle_count} @ {self.cycle_rate:.1f} Hz")
        
        # Show sensor staleness
        print("   Sensor staleness:")
        for sensor, age in sorted(staleness.items()):
            if age < 1.0:
                status = "✓ fresh"
            elif age < 5.0:
                status = "⚠ stale"
            else:
                status = "✗ dead"
            print(f"     {sensor:12s}: {age:6.3f}s {status}")
        
        # Show update rates
        stats = self.adapter.get_stats()
        print("   Update rates:")
        for sensor, info in sorted(stats.items()):
            rate = info['rate_hz']
            print(f"     {sensor:12s}: {rate:6.1f} Hz")


def main():
    """Demo the async brain loop."""
    print("=" * 60)
    print("ASYNC BRAIN LOOP DEMO")
    print("=" * 60)
    print("\nThis shows how the brain was MEANT to work:")
    print("• Sensors write to buffers at their natural rates")
    print("• Brain reads buffers on each cycle")
    print("• No synchronization, no waiting")
    print("• Just like biological nervous systems!")
    print()
    
    # Create mock brain
    class MockBrain:
        def process(self, motor_feedback, sensory_input):
            # Just return some motor commands
            return np.array([0.1, 0.0, 0.0])
    
    brain = MockBrain()
    loop = AsyncBrainLoop(brain, cycle_rate=50.0)
    
    # Simulate some sensor updates in background
    import threading
    
    def simulate_battery():
        while loop.running:
            loop.adapter.update_sensor('battery', 7.4 + np.random.randn() * 0.1)
            time.sleep(1.0)  # 1 Hz
    
    def simulate_ultrasonic():
        while loop.running:
            loop.adapter.update_sensor('ultrasonic', 50 + np.random.randn() * 10)
            time.sleep(0.05)  # 20 Hz
    
    # Start sensor simulators
    threading.Thread(target=simulate_battery, daemon=True).start()
    threading.Thread(target=simulate_ultrasonic, daemon=True).start()
    
    try:
        loop.start()
    except KeyboardInterrupt:
        print("\n\nStopping...")
        loop.stop()


if __name__ == "__main__":
    main()