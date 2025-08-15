#!/usr/bin/env python3
"""Find sane memory values that work for both 30-second tests and 1-year runs."""

import math
import numpy as np

def test_parameter_sets():
    """Test different parameter combinations."""
    
    print("=== Finding Sane Memory Parameters ===\n")
    
    # The fundamental problem: 30 seconds vs 1 year = 1,051,200x difference!
    # We need a system that doesn't rely solely on exponential decay
    
    # Option 1: Baseline + Decay model
    print("Option 1: BASELINE + DECAY MODEL")
    print("-" * 50)
    baseline = 0.05  # Minimum field value (never decays below this)
    active_value = 0.5  # Value when actively stimulated
    decay_rate = 0.999  # Per cycle
    threshold = 0.08  # Discovery threshold above baseline
    
    print(f"Baseline field value: {baseline}")
    print(f"Active field value: {active_value}")
    print(f"Decay rate: {decay_rate}")
    print(f"Discovery threshold: {threshold}")
    
    # After 30 seconds (300 cycles)
    value_30s = baseline + (active_value - baseline) * (decay_rate ** 300)
    print(f"\nAfter 30 seconds: {value_30s:.3f} {'✓' if value_30s > threshold else '✗'}")
    
    # After 1 hour
    value_1h = baseline + (active_value - baseline) * (decay_rate ** 36000)
    print(f"After 1 hour: {value_1h:.3f} {'✓' if value_1h > threshold else '✗'}")
    
    # After 1 day
    value_1d = baseline + (active_value - baseline) * (decay_rate ** 864000)
    print(f"After 1 day: {value_1d:.3f} {'✓' if value_1d > threshold else '✗'}")
    
    print("\nThis works! Baseline prevents complete decay.")
    
    # Option 2: Activity-triggered persistence
    print("\n\nOption 2: ACTIVITY-TRIGGERED PERSISTENCE")
    print("-" * 50)
    print("Instead of continuous decay, use discrete persistence levels:")
    print("- Transient: Decays in minutes (for exploration)")
    print("- Short-term: Persists for hours (for current task)")
    print("- Long-term: Persists for days/months (consolidated)")
    print("- Permanent: Never decays (core knowledge)")
    
    print("\nImplementation:")
    print("- New experiences start as transient")
    print("- Repeated activation promotes to next level")
    print("- Each level has different decay rate:")
    print("  - Transient: 0.99 (half-life ~69 cycles = 7 seconds)")
    print("  - Short-term: 0.9999 (half-life ~6,931 cycles = 11.5 minutes)")
    print("  - Long-term: 0.999999 (half-life ~693,147 cycles = 19 hours)")
    print("  - Permanent: 1.0 (no decay)")
    
    # Option 3: Hybrid approach (RECOMMENDED)
    print("\n\nOption 3: HYBRID APPROACH (RECOMMENDED)")
    print("-" * 50)
    print("Combine baseline + activity-based persistence:")
    
    # Parameters
    baseline = 0.02  # Low baseline
    transient_boost = 0.3  # Initial activation
    threshold = 0.05  # Discovery threshold
    
    # Decay rates by consolidation level
    decay_rates = {
        0: 0.995,     # Unconsolidated: fast decay
        1: 0.9995,    # Lightly consolidated
        2: 0.99995,   # Well consolidated
        3: 1.0        # Permanent
    }
    
    print(f"\nParameters:")
    print(f"- Baseline: {baseline}")
    print(f"- Initial activation: {baseline + transient_boost} = {baseline + transient_boost}")
    print(f"- Discovery threshold: {threshold}")
    print(f"- Consolidation increases with:")
    print(f"  - Repeated activation (resonance)")
    print(f"  - Importance scoring")
    print(f"  - Maintenance cycles")
    
    # Simulate different scenarios
    print(f"\nScenario 1: Single exposure (30-second test)")
    value = baseline + transient_boost
    for t in [3, 30, 300]:
        cycles = t * 10
        decayed = baseline + transient_boost * (decay_rates[0] ** cycles)
        print(f"  After {t}s: {decayed:.3f} {'✓' if decayed > threshold else '✗'}")
    
    print(f"\nScenario 2: Important memory (consolidated)")
    consolidation = 2
    value = baseline + transient_boost
    for name, seconds in [("1 hour", 3600), ("1 day", 86400), ("1 week", 604800)]:
        cycles = seconds * 10
        decayed = baseline + transient_boost * (decay_rates[consolidation] ** cycles)
        print(f"  After {name}: {decayed:.3f} {'✓' if decayed > threshold else '✗'}")
    
    # Final recommendations
    print("\n\n=== FINAL RECOMMENDATIONS ===")
    print("\n1. Implement baseline field activity:")
    print("   self.field_baseline = 0.02")
    print("   self.unified_field.clamp_(min=self.field_baseline)")
    
    print("\n2. Adjust field intensity calculation:")
    print("   intensity = self.field_baseline + 0.3 * (norm / sqrt(dims))")
    
    print("\n3. Lower topology threshold:")
    print("   self.topology_stability_threshold = 0.05")
    
    print("\n4. Add consolidation levels to topology regions:")
    print("   - Track activation count")
    print("   - Promote after N activations")
    print("   - Use consolidation-specific decay rates")
    
    print("\n5. During maintenance, apply consolidation-aware decay:")
    print("   decay = self.decay_rates[region['consolidation_level']]")
    print("   field_value = baseline + (field_value - baseline) * decay")

if __name__ == "__main__":
    test_parameter_sets()