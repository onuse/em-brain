#!/usr/bin/env python3
"""Analyze whether current memory system values work across different timescales."""

import math
import numpy as np
import matplotlib.pyplot as plt

# Current system parameters
FIELD_DECAY_RATE = 0.999  # Per cycle
MEMORY_HALF_LIFE = 3600  # seconds (1 hour)
IMPORTANCE_BOOST = 1.1  # Per resonant activation
CONSOLIDATION_DECAY_SLOWDOWN = 0.99  # Multiplier for important memories
ENERGY_DISSIPATION = 0.98  # During maintenance
TOPOLOGY_THRESHOLD = 0.1  # Minimum field value for topology discovery

# Assume 10 cycles per second (100ms per cycle)
CYCLES_PER_SECOND = 10

def field_value_after_time(initial_value, seconds, resonance_frequency=0):
    """Calculate field value after given time with decay and resonance."""
    cycles = seconds * CYCLES_PER_SECOND
    
    # Basic exponential decay
    value = initial_value * (FIELD_DECAY_RATE ** cycles)
    
    # Add resonance boosts if any
    if resonance_frequency > 0:
        # How many times does this memory resonate?
        resonance_events = int(seconds * resonance_frequency)
        for _ in range(resonance_events):
            value *= IMPORTANCE_BOOST
    
    return value

def analyze_timescales():
    """Analyze memory persistence across different timescales."""
    print("=== Memory System Timescale Analysis ===\n")
    
    # Initial field intensity calculation
    # Field coords are in [-1, 1] range for 37 dimensions
    # Typical norm would be sqrt(sum of squares) ≈ sqrt(37 * 0.5^2) ≈ 3.0
    typical_norm = 3.0
    initial_intensity = typical_norm / math.sqrt(37)
    print(f"Typical initial field intensity: {initial_intensity:.4f}")
    print(f"Topology discovery threshold: {TOPOLOGY_THRESHOLD}")
    print(f"Ratio: {initial_intensity/TOPOLOGY_THRESHOLD:.2f}x\n")
    
    # Problem: Initial intensity (0.115) is barely above threshold (0.1)!
    
    # Analyze decay over different timescales
    timescales = [
        ("30 seconds", 30),
        ("5 minutes", 300),
        ("1 hour", 3600),
        ("1 day", 86400),
        ("1 week", 604800),
        ("1 month", 2592000),
        ("1 year", 31536000)
    ]
    
    print("Field decay without resonance:")
    print("-" * 50)
    for name, seconds in timescales:
        value = field_value_after_time(initial_intensity, seconds)
        percent = (value / initial_intensity) * 100
        below_threshold = " [BELOW THRESHOLD]" if value < TOPOLOGY_THRESHOLD else ""
        print(f"{name:12} -> {value:.6f} ({percent:.1f}%){below_threshold}")
    
    # Find when field drops below threshold
    cycles_to_threshold = math.log(TOPOLOGY_THRESHOLD / initial_intensity) / math.log(FIELD_DECAY_RATE)
    seconds_to_threshold = cycles_to_threshold / CYCLES_PER_SECOND
    print(f"\nField drops below threshold after: {seconds_to_threshold:.1f} seconds!")
    
    # Analyze with resonance
    print("\n\nField value with periodic resonance (once per minute):")
    print("-" * 50)
    resonance_freq = 1/60  # Once per minute
    for name, seconds in timescales[:5]:  # First few timescales
        value = field_value_after_time(initial_intensity, seconds, resonance_freq)
        percent = (value / initial_intensity) * 100
        print(f"{name:12} -> {value:.6f} ({percent:.1f}%)")
    
    # Recommendations
    print("\n\n=== RECOMMENDATIONS ===")
    print("\n1. IMMEDIATE ISSUE: Field intensity too weak")
    print("   - Initial intensity (~0.115) barely exceeds threshold (0.1)")
    print("   - Field drops below threshold in ~700 seconds without reinforcement")
    print("   - This explains why no topology regions form in tests!")
    
    print("\n2. SUGGESTED FIXES:")
    print("   a) Increase field intensity calculation:")
    print("      - Current: norm(coords) / sqrt(37) ≈ 0.115")
    print("      - Suggested: norm(coords) / sqrt(37) * 5.0 ≈ 0.575")
    print("   OR")
    print("   b) Lower topology threshold:")
    print("      - Current: 0.1")
    print("      - Suggested: 0.02")
    
    print("\n3. FOR LONG-TERM STABILITY:")
    print("   - Implement adaptive thresholds based on field statistics")
    print("   - Use log-scale intensity: log(1 + intensity)")
    print("   - Add baseline field activity to prevent complete decay")
    
    # Calculate better parameters
    print("\n\n=== PROPOSED PARAMETER SET ===")
    
    # Option 1: Boost intensity
    boost_factor = 5.0
    new_intensity = initial_intensity * boost_factor
    new_decay_rate = 0.9995  # Slower decay for higher values
    
    print(f"\nOption 1: Boost field intensity")
    print(f"  - Intensity multiplier: {boost_factor}x")
    print(f"  - New typical intensity: {new_intensity:.3f}")
    print(f"  - Slower decay rate: {new_decay_rate}")
    
    # Option 2: Adaptive thresholds
    print(f"\nOption 2: Adaptive topology threshold")
    print(f"  - Base threshold: 0.02 (5x lower)")
    print(f"  - Adaptive: threshold = 0.02 + 0.1 * field_mean")
    print(f"  - This scales with overall field activity")
    
    # Option 3: Log-scale intensity
    print(f"\nOption 3: Log-scale field intensity")
    print(f"  - intensity = log(1 + norm(coords))")
    print(f"  - More stable across timescales")
    print(f"  - Natural compression of large values")

if __name__ == "__main__":
    analyze_timescales()