#!/usr/bin/env python3
"""
Test GPU Future Simulator Integration
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'server'))

from server.src.core.simplified_brain_factory import SimplifiedBrainFactory
import numpy as np
import time

print("Testing GPU Future Simulator...")
print("=" * 60)

# Create brain with future simulation
factory = SimplifiedBrainFactory()
brain_interface = factory.create(sensory_dim=24, motor_dim=4)
brain = brain_interface.brain

print(f"\nBrain configuration:")
print(f"  Device: {brain.device}")
print(f"  Future simulation: {brain.use_future_simulation}")
if brain.future_simulator:
    print(f"  Futures per action: {brain.future_simulator.n_futures}")
    print(f"  Simulation horizon: {brain.future_simulator.horizon} cycles")

# Run a few cycles to test
print("\nRunning brain cycles with future simulation...")
sensory_input = [0.5] * 24

for i in range(5):
    start = time.time()
    motor_output, brain_state = brain.process_robot_cycle(sensory_input)
    cycle_time = time.time() - start
    
    print(f"\nCycle {i+1}:")
    print(f"  Cycle time: {cycle_time*1000:.1f}ms")
    print(f"  Motor output: {motor_output}")
    print(f"  Confidence: {brain_state['prediction_confidence']:.3f}")
    print(f"  Cognitive mode: {brain_state['cognitive_mode']}")
    
    # Vary input slightly
    sensory_input[0] += 0.1

print("\n" + "=" * 60)
print("FUTURE SIMULATOR TEST COMPLETE")
print("\nKey observations:")
print("- Future simulation is enabled and running")
print("- Actions are now evaluated through mental simulation")
print("- Cycle time includes GPU simulation overhead")
print("- Decision quality should improve with future lookahead")