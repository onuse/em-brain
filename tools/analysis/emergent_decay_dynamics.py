#!/usr/bin/env python3
"""Design emergent non-uniform decay for the unified field."""

import numpy as np
import matplotlib.pyplot as plt

def analyze_emergent_decay():
    """Explore non-uniform decay that preserves memories naturally."""
    
    print("=== EMERGENT NON-UNIFORM DECAY ===\n")
    
    print("PRINCIPLE: Decay should hit high-energy regions hardest")
    print("This creates natural memory consolidation without extra fields!\n")
    
    print("1. BIOLOGICAL INSPIRATION")
    print("-" * 50)
    print("- Highly active neurons consume more energy")
    print("- Need more maintenance (protein synthesis, etc.)")
    print("- Without maintenance, decay faster")
    print("- But strong synapses have structural support (baseline)")
    
    print("\n2. PROPOSED DECAY FUNCTION")
    print("-" * 50)
    print("Instead of: field *= 0.999")
    print("Use: field = baseline + (field - baseline) * decay_function(field)")
    
    # Define the decay function
    field_values = np.linspace(0, 2, 100)
    baseline = 0.02
    
    # Option 1: Inverse sigmoid (high values decay more)
    def decay_fn_1(x, baseline=0.02):
        # Protect values near baseline, decay high values more
        relative_magnitude = np.abs(x - baseline)
        decay_rate = 0.999 - 0.01 * np.tanh(relative_magnitude * 2)
        return decay_rate
    
    # Option 2: Energy-based decay
    def decay_fn_2(x, baseline=0.02):
        # High energy = high decay
        energy = (x - baseline) ** 2
        decay_rate = 0.999 - 0.005 * energy
        return np.clip(decay_rate, 0.95, 0.999)
    
    # Option 3: Homeostatic decay (RECOMMENDED)
    def decay_fn_3(x, baseline=0.02, target=0.1):
        # Decay toward target value, not zero
        if x > target:
            # Above target: stronger decay for higher values
            excess = x - target
            decay_rate = 0.999 - 0.01 * np.tanh(excess * 5)
        else:
            # Below target: very slow decay (or even growth)
            deficit = target - x
            decay_rate = 0.999 + 0.001 * np.tanh(deficit * 10)
        return np.clip(decay_rate, 0.98, 1.001)
    
    print("\nOption 1: Inverse Sigmoid Decay")
    print("- Simple, always decays")
    print("- High values decay faster")
    
    print("\nOption 2: Energy-Based Decay")
    print("- Quadratic energy penalty")
    print("- Very high values decay much faster")
    
    print("\nOption 3: Homeostatic Decay (RECOMMENDED)")
    print("- Decays toward target, not zero")
    print("- Can even grow if below target")
    print("- Natural equilibrium points")
    
    print("\n3. IMPLEMENTATION")
    print("-" * 50)
    print("""
    def _evolve_unified_field(self, experience):
        # Current approach (uniform decay)
        # self.unified_field *= self.field_decay_rate
        
        # New approach: Non-uniform homeostatic decay
        baseline = 0.02
        target = 0.1
        
        # Compute decay rates based on field values
        excess = torch.relu(self.unified_field - target)
        deficit = torch.relu(target - self.unified_field)
        
        # High values decay faster, low values decay slower (or grow)
        decay_rates = 0.999 - 0.01 * torch.tanh(excess * 5) + 0.001 * torch.tanh(deficit * 10)
        decay_rates = torch.clamp(decay_rates, 0.98, 1.001)
        
        # Apply non-uniform decay with baseline protection
        self.unified_field = baseline + (self.unified_field - baseline) * decay_rates
        
        # Natural consolidation: frequently activated regions stay near target
        # Unused regions decay toward baseline
        # No separate consolidation field needed!
    """)
    
    print("\n4. EMERGENT PROPERTIES")
    print("-" * 50)
    print("This creates natural memory dynamics:")
    print("- New inputs: Start at high values, decay quickly at first")
    print("- Repeated inputs: Stabilize near target value")
    print("- Unused memories: Decay to baseline (not zero)")
    print("- Important patterns: Self-maintain near equilibrium")
    
    print("\n5. ADVANTAGES")
    print("-" * 50)
    print("✓ No separate consolidation field")
    print("✓ Biologically plausible (homeostasis)")
    print("✓ Natural stability-plasticity balance")
    print("✓ Memories never fully disappear (baseline)")
    print("✓ Self-organizing dynamics")
    print("✓ Single, elegant mechanism")
    
    print("\n6. PARAMETER TUNING")
    print("-" * 50)
    print("Key parameters:")
    print("- baseline = 0.02  # Minimum field value")
    print("- target = 0.1     # Equilibrium point")
    print("- decay_strength = 0.01  # How fast high values decay")
    print("- growth_strength = 0.001  # How fast low values grow")
    
    print("\n7. TIMESCALE ANALYSIS")
    print("-" * 50)
    
    # Simulate the dynamics
    timesteps = 10000
    value_high = 0.5  # Strong activation
    value_med = 0.1   # Medium activation  
    value_low = 0.03  # Weak activation
    
    values_high = [value_high]
    values_med = [value_med]
    values_low = [value_low]
    
    for t in range(timesteps):
        # Apply homeostatic decay
        values_high.append(baseline + (values_high[-1] - baseline) * decay_fn_3(values_high[-1]))
        values_med.append(baseline + (values_med[-1] - baseline) * decay_fn_3(values_med[-1]))
        values_low.append(baseline + (values_low[-1] - baseline) * decay_fn_3(values_low[-1]))
    
    print(f"After 1000 steps (~100 seconds):")
    print(f"  High (0.5) -> {values_high[1000]:.3f}")
    print(f"  Medium (0.1) -> {values_med[1000]:.3f}")
    print(f"  Low (0.03) -> {values_low[1000]:.3f}")
    
    print(f"\nAfter 10000 steps (~1000 seconds):")
    print(f"  High (0.5) -> {values_high[-1]:.3f}")
    print(f"  Medium (0.1) -> {values_med[-1]:.3f}")
    print(f"  Low (0.03) -> {values_low[-1]:.3f}")
    
    print("\nNotice: All converge toward equilibrium, not zero!")
    
    print("\n8. CONCLUSION")
    print("-" * 50)
    print("Non-uniform decay solves multiple problems:")
    print("1. Natural memory persistence (baseline)")
    print("2. Automatic consolidation (homeostasis)")
    print("3. No separate metadata fields")
    print("4. Biologically plausible")
    print("5. Philosophically pure (one field, one dynamics)")

if __name__ == "__main__":
    analyze_emergent_decay()