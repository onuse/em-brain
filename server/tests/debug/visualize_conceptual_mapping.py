#!/usr/bin/env python3
"""
Visualize how PiCar-X sensors map to conceptual space.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.core.dynamic_brain_factory import DynamicBrainFactory
from src.core.dynamic_dimension_calculator import DynamicDimensionCalculator
import numpy as np


def visualize_conceptual_mapping():
    """Show how sensors map to conceptual dimensions."""
    
    print("🧠 Conceptual Space Mapping Visualization")
    print("=" * 50)
    
    # Create dimension calculator
    calc = DynamicDimensionCalculator(complexity_factor=6.0)
    
    # Calculate dimensions for PiCar-X
    picarx_sensors = 17  # 16 + reward
    picarx_motors = 4
    
    dimensions = calc.calculate_dimensions(picarx_sensors, picarx_motors)
    
    print(f"\n📊 PiCar-X Conceptual Space Structure:")
    print(f"   Input: {picarx_sensors} sensors")
    print(f"   Output: {len(dimensions)} conceptual dimensions")
    print(f"   Compression: {len(dimensions)}/{picarx_sensors} = {len(dimensions)/picarx_sensors:.1f}x")
    
    # Show family distribution
    print(f"\n🌐 Dimension Families:")
    from collections import defaultdict
    families = defaultdict(list)
    
    for i, dim in enumerate(dimensions):
        families[dim.family].append((i, dim.name))
    
    for family, dims in families.items():
        print(f"\n{family.value.upper()} ({len(dims)} dimensions):")
        for idx, name in dims:
            print(f"  [{idx:2d}] {name}")
    
    # Simulate sensor mapping
    print(f"\n🔄 Example Sensor → Conceptual Mapping:")
    
    # Create a brain to see actual mapping
    factory = DynamicBrainFactory({
        'use_dynamic_brain': True,
        'use_full_features': True,
        'quiet_mode': True
    })
    
    brain_wrapper = factory.create(
        field_dimensions=None,
        spatial_resolution=4,
        sensory_dim=picarx_sensors,
        motor_dim=picarx_motors
    )
    brain = brain_wrapper.brain
    
    # Example sensor data
    example_sensors = [
        0.3,   # ultrasonic (close obstacle)
        0.2, 0.8, 0.2,  # line sensors (on line)
        0.3, 0.3,  # motors moving
        0, 0, 0,  # angles
        7.4, 1, 0,  # battery, line detected, no cliff
        45, 0.3, 1000, 0,  # system stats
        0.7  # high reward
    ]
    
    # Get the field experience
    experience = brain._robot_sensors_to_field_experience(example_sensors)
    
    print(f"\nSensor values → Conceptual coordinates:")
    print(f"{'Sensor':<20} {'Value':>8} → {'Conceptual Influence'}")
    print("-" * 60)
    
    # Show key sensor mappings
    key_sensors = [
        (0, "Ultrasonic", "→ Spatial dims 0-2 (position)"),
        (1, "Grayscale R", "→ Spatial dim 3"),
        (2, "Grayscale C", "→ Spatial dim 4"), 
        (3, "Grayscale L", "→ Spatial dim 5"),
        (4, "Motor L", "→ Flow dims (velocity)"),
        (5, "Motor R", "→ Flow dims (velocity)"),
        (16, "Reward", "→ Field intensity (all dims)")
    ]
    
    for idx, name, mapping in key_sensors:
        if idx < len(example_sensors):
            value = example_sensors[idx]
            print(f"{name:<20} {value:8.2f} {mapping}")
    
    # Show resulting field coordinates
    coords = experience.field_coordinates.cpu().numpy()
    
    print(f"\n📍 Resulting Conceptual Position (first 10 dims):")
    for i in range(min(10, len(coords))):
        bar = "█" * int(abs(coords[i]) * 20)
        print(f"  Dim {i:2d} [{dimensions[i].name[:12]:>12}]: {coords[i]:+.3f} {bar}")
    
    # Show how this creates gradients
    print(f"\n🌊 Field Dynamics:")
    print(f"  Field intensity: {experience.field_intensity:.3f}")
    print(f"  This creates an activation 'bubble' in the {len(dimensions)}D space")
    print(f"  The bubble spreads via diffusion")
    print(f"  Gradients form naturally, pointing toward/away from activation")
    print(f"  Motor commands follow these gradients")
    
    # Show tensor mapping
    tensor_indices = brain._conceptual_to_tensor_indices(experience.field_coordinates)
    print(f"\n🎯 Tensor Mapping:")
    print(f"  26D conceptual → 11D tensor")
    print(f"  Tensor indices: {tensor_indices}")
    print(f"  Multiple conceptual dims map to each tensor dim")
    print(f"  Preserves neighborhood relationships")


if __name__ == "__main__":
    visualize_conceptual_mapping()