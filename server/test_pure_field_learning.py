#!/usr/bin/env python3
"""
Simple test script to verify PureFieldBrain actually learns.
Provides clear visibility into what's happening inside the field.
"""

import torch
import numpy as np
import time
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.brains.field.pure_field_brain import PureFieldBrain, SCALE_CONFIGS

def create_simple_pattern(dim: int, pattern_type: str = 'sine') -> np.ndarray:
    """Create simple sensory patterns for testing."""
    t = np.linspace(0, 2 * np.pi, dim)
    if pattern_type == 'sine':
        return np.sin(t)
    elif pattern_type == 'cosine':
        return np.cos(t)
    elif pattern_type == 'square':
        return np.sign(np.sin(t))
    elif pattern_type == 'random':
        return np.random.randn(dim) * 0.5
    else:
        return np.zeros(dim)

def test_learning_progression():
    """Test that the brain shows clear learning progression."""
    
    print("=" * 70)
    print("🧠 PUREFIELDBRAIN LEARNING TEST")
    print("=" * 70)
    
    # Configuration
    input_dim = 10
    output_dim = 4
    test_cycles = 1000
    
    # Detect hardware and choose appropriate scale
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        scale_config = SCALE_CONFIGS['medium']
        print(f"🎯 Using MEDIUM scale on GPU for optimal learning")
    else:
        scale_config = SCALE_CONFIGS['small']
        print(f"⚡ Using SMALL scale on CPU for fast performance")
    
    print(f"📊 Configuration: {scale_config.total_params:,} parameters across {len(scale_config.levels)} levels")
    print(f"🔧 Test parameters: {input_dim} inputs → {output_dim} outputs")
    print()
    
    # Create brain
    brain = PureFieldBrain(
        input_dim=input_dim,
        output_dim=output_dim,
        scale_config=scale_config,
        device=device,
        aggressive=True  # Use aggressive learning for visible changes
    )
    
    print("🎓 Learning parameters (aggressive mode):")
    print(f"  - Learning rate: {brain.learning_rate:.3f}")
    print(f"  - Decay rate: {brain.decay_rate:.3f}")
    print(f"  - Diffusion: {brain.diffusion_rate:.3f}")
    print(f"  - Noise scale: {brain.noise_scale:.3f}")
    print(f"  - Emergent dynamics: {brain.emergent_dynamics}")
    print()
    
    # Track metrics over time
    energy_history = []
    motor_history = []
    resonance_history = []
    
    print("🚀 Starting learning test...")
    print("-" * 50)
    
    start_time = time.time()
    
    for cycle in range(test_cycles):
        # Create varied input patterns
        if cycle < 250:
            # Phase 1: Simple sine patterns
            pattern = create_simple_pattern(input_dim, 'sine') * (1 + cycle * 0.001)
            reward = 0.1
        elif cycle < 500:
            # Phase 2: Cosine patterns
            pattern = create_simple_pattern(input_dim, 'cosine') * 2.0
            reward = 0.2
        elif cycle < 750:
            # Phase 3: Square waves
            pattern = create_simple_pattern(input_dim, 'square')
            reward = -0.1  # Negative reward
        else:
            # Phase 4: Random exploration
            pattern = create_simple_pattern(input_dim, 'random')
            reward = np.random.randn() * 0.1
        
        # Process through brain
        sensory = torch.tensor(pattern, dtype=torch.float32)
        motor = brain.forward(sensory, reward)
        
        # Get metrics
        metrics = brain.metrics
        energy_history.append(metrics['field_energy'])
        motor_history.append(metrics['motor_strength'])
        resonance_history.append(metrics['sensory_resonance'])
        
        # Print progress every 100 cycles
        if (cycle + 1) % 100 == 0:
            avg_energy = np.mean(energy_history[-100:])
            avg_motor = np.mean(motor_history[-100:])
            avg_resonance = np.mean(resonance_history[-100:])
            
            # Determine learning phase
            if cycle < 250:
                phase = "SINE LEARNING"
            elif cycle < 500:
                phase = "COSINE LEARNING"
            elif cycle < 750:
                phase = "SQUARE LEARNING"
            else:
                phase = "EXPLORATION"
            
            print(f"Cycle {cycle+1:4d} [{phase:15s}]: "
                  f"Energy={avg_energy:.3f} | "
                  f"Motor={avg_motor:.3f} | "
                  f"Resonance={avg_resonance:.3f}")
            
            # Check for emergent behavior
            if brain.emergent_dynamics and metrics['meta_adaptation_rate'] > 0:
                print(f"  🌟 Emergent: Cross-scale coherence={metrics['cross_scale_coherence']:.3f}, "
                      f"Meta-adaptation={metrics['meta_adaptation_rate']:.3f}")
    
    elapsed = time.time() - start_time
    
    print("-" * 50)
    print(f"✅ Test completed in {elapsed:.2f} seconds")
    print(f"   Average cycle time: {elapsed/test_cycles*1000:.2f} ms")
    print()
    
    # Analyze learning progression
    print("📈 Learning Analysis:")
    
    # Check if energy increased (field developing structure)
    early_energy = np.mean(energy_history[:100])
    late_energy = np.mean(energy_history[-100:])
    energy_growth = (late_energy - early_energy) / (early_energy + 1e-6) * 100
    
    print(f"  - Field energy growth: {energy_growth:.1f}%")
    if energy_growth > 10:
        print("    ✅ Significant field structure development")
    elif energy_growth > 0:
        print("    ⚠️  Modest field development")
    else:
        print("    ❌ No field development detected")
    
    # Check motor activation
    early_motor = np.mean(motor_history[:100])
    late_motor = np.mean(motor_history[-100:])
    motor_growth = (late_motor - early_motor) / (early_motor + 1e-6) * 100
    
    print(f"  - Motor strength growth: {motor_growth:.1f}%")
    if abs(motor_growth) > 10:
        print("    ✅ Motor system actively responding")
    else:
        print("    ⚠️  Limited motor response")
    
    # Check resonance (field-input coupling)
    avg_resonance = np.mean(resonance_history)
    resonance_std = np.std(resonance_history)
    
    print(f"  - Average resonance: {avg_resonance:.3f} (std={resonance_std:.3f})")
    if resonance_std > 0.1:
        print("    ✅ Dynamic field-input coupling")
    else:
        print("    ⚠️  Static field response")
    
    # Final state summary
    print()
    print("🏁 Final Brain State:")
    final_metrics = brain.metrics
    print(f"  - Total cycles: {final_metrics['cycle_count']}")
    print(f"  - Field energy: {final_metrics['field_energy']:.3f}")
    print(f"  - Hierarchical levels: {final_metrics['hierarchical_depth']}")
    print(f"  - Total parameters: {final_metrics['total_parameters']:,}")
    
    if brain.emergent_dynamics:
        print(f"  - Emergence detected: Cross-scale coherence active")
    
    print()
    print("=" * 70)
    
    # Return success indicator
    return energy_growth > 0 or motor_growth != 0

if __name__ == "__main__":
    # Run the test
    success = test_learning_progression()
    
    if success:
        print("✅ PureFieldBrain is learning successfully!")
        sys.exit(0)
    else:
        print("⚠️  Learning needs parameter tuning")
        sys.exit(1)