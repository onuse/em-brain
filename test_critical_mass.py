#!/usr/bin/env python3
"""
Test script for Critical Mass Brain integration
"""

import sys
import time
sys.path.append('server/src')

print("Testing Critical Mass Brain Integration")
print("=" * 60)

# Import with error handling
try:
    from brains.field.critical_mass_field_brain import CriticalMassFieldBrain, EmergenceConfig
    print("✓ Import successful")
except ImportError as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# Create lighter config for testing
test_config = EmergenceConfig(
    field_size=(16, 16, 16, 32),  # Much smaller for testing
    swarm_size=100,  # Reduced from 1000
    superposition_branches=10,  # Reduced from 100
)

print(f"\nTest Configuration:")
print(f"  Field size: {test_config.field_size}")
print(f"  Swarm size: {test_config.swarm_size}")
print(f"  Superposition branches: {test_config.superposition_branches}")

# Initialize brain
print("\nInitializing brain...")
start = time.time()
brain = CriticalMassFieldBrain(config=test_config)
init_time = time.time() - start
print(f"✓ Brain initialized in {init_time:.2f}s")

# Test processing cycle
print("\nTesting processing cycle...")
sensor_data = {
    'ultrasonic': 50.0,
    'vision_detected': 1.0,
    'audio_level': 0.3,
    'battery': 0.8,
    'temperature': 25.0
}

start = time.time()
motor_commands = brain.process(sensor_data)
process_time = time.time() - start
print(f"✓ Process completed in {process_time:.3f}s")

# Show motor commands
print("\nMotor Commands:")
for key, value in motor_commands.items():
    print(f"  {key}: {value:.3f}")

# Show telemetry
telemetry = brain.get_telemetry()
print("\nTelemetry:")
print(f"  Device: {telemetry['device']}")
print(f"  Cycles: {telemetry['cycles']}")
print(f"  Concepts formed: {telemetry['concepts_formed']}")
print(f"  Emergence score: {telemetry['emergence_score']:.1%}")

# Run a few more cycles to see emergence
print("\nRunning 10 cycles to observe emergence...")
for i in range(10):
    # Vary input slightly
    sensor_data['ultrasonic'] = 50.0 + i * 5
    sensor_data['audio_level'] = 0.3 + i * 0.05
    
    start = time.time()
    motor_commands = brain.process(sensor_data)
    cycle_time = time.time() - start
    
    print(f"  Cycle {i+2}: {cycle_time:.3f}s, emergence score: {brain.get_telemetry()['emergence_score']:.1%}")

# Final telemetry
final_telemetry = brain.get_telemetry()
print("\nFinal State:")
print(f"  Total cycles: {final_telemetry['cycles']}")
print(f"  Concepts formed: {final_telemetry['concepts_formed']}")
print(f"  Bindings active: {final_telemetry['bindings_active']}")
print(f"  Decision confidence: {final_telemetry['decision_confidence']:.3f}")
print(f"  Preference stability: {final_telemetry['preference_stability']:.3f}")
print(f"  Memory accuracy: {final_telemetry['memory_accuracy']:.3f}")
print(f"  Emergence score: {final_telemetry['emergence_score']:.1%}")

print("\n" + "=" * 60)
print("INTEGRATION TEST COMPLETE")

if final_telemetry['emergence_score'] > 0:
    print("✓ Emergence detected - brain is working!")
else:
    print("⚠ No emergence yet - may need more cycles")

print("\nTo integrate with robot:")
print("1. Copy critical_mass_field_brain.py to server/src/brains/field/")
print("2. Update brain factory to include CriticalMassFieldBrain")
print("3. Start server with --brain critical_mass")
print("4. Monitor emergence through telemetry")