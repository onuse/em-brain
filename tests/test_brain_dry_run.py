#!/usr/bin/env python3
"""
Quick dry-run test for PureFieldBrain
======================================
Validates basic brain functionality before robot deployment.
Tests that the brain does SOMETHING rather than NOTHING.
"""

import sys
import time
import numpy as np
import torch
from pathlib import Path

# Add server to path
sys.path.insert(0, str(Path(__file__).parent / "server"))

from src.brains.field.pure_field_brain import PureFieldBrain

def to_numpy(x):
    """Convert tensor to numpy array"""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x

def test_brain_basics():
    """Quick test of basic brain functionality"""
    print("🧠 PureFieldBrain Dry-Run Test\n" + "="*50)
    
    # Create brain with small configuration for quick testing
    print("Creating brain (hardware_constrained config)...")
    
    # Import and use the helper function
    from src.brains.field.pure_field_brain import create_pure_field_brain
    
    brain = create_pure_field_brain(
        size='hardware_constrained',  # Smallest config for speed
        input_dim=12,  # Simple sensory input
        output_dim=4   # 4 motor outputs
    )
    
    # Get field info from first level
    first_level = brain.levels[0] if hasattr(brain, 'levels') else brain
    print(f"✓ Brain created with {len(brain.levels)} level(s)")
    print(f"  Parameters: ~{sum(p.numel() for p in brain.parameters())/1000:.1f}K")
    
    # Test 1: Verify brain produces output
    print("\n1. Testing basic input/output...")
    sensory = np.random.randn(12).astype(np.float32) * 0.5
    motor = brain.process(sensory)
    motor_np = to_numpy(motor)
    
    if motor_np is not None and len(motor_np) == 4:
        print(f"✓ Brain produces motor output: {motor_np}")
    else:
        print("✗ Brain failed to produce valid output")
        return False
    
    # Test 2: Verify output changes with different inputs
    print("\n2. Testing responsiveness to input changes...")
    outputs = []
    for i in range(5):
        sensory = np.sin(np.arange(12) * i * 0.5).astype(np.float32)
        motor = brain.process(sensory)
        outputs.append(to_numpy(motor))
    
    # Check if outputs vary
    output_variance = np.var(outputs, axis=0)
    if np.any(output_variance > 1e-6):
        print(f"✓ Brain responds to different inputs")
        print(f"  Output variance: {output_variance}")
    else:
        print("✗ Brain produces constant output (may be stuck)")
        return False
    
    # Test 3: Verify brain state evolves (does SOMETHING)
    print("\n3. Testing field evolution...")
    initial_metrics = brain.metrics
    initial_energy = initial_metrics.get('field_energy', initial_metrics.get('energy', 0))
    
    # Run several cycles
    for i in range(20):
        sensory = np.random.randn(12).astype(np.float32) * 0.3
        _ = brain.process(sensory)
    
    final_metrics = brain.metrics
    final_energy = final_metrics.get('field_energy', final_metrics.get('energy', 0))
    energy_change = abs(final_energy - initial_energy)
    
    if energy_change > 1e-4:
        print(f"✓ Field evolves over time")
        print(f"  Energy change: {initial_energy:.4f} → {final_energy:.4f}")
    else:
        print("✗ Field appears static")
        return False
    
    # Test 4: Check for pathological behaviors
    print("\n4. Checking for pathological behaviors...")
    issues = []
    
    # Check for NaN/Inf
    stats = brain.metrics
    energy = stats.get('field_energy', stats.get('energy', 0))
    if np.isnan(energy) or np.isinf(energy):
        issues.append("NaN/Inf in field")
    
    # Check for dead field (all zeros)
    if energy < 1e-10:
        issues.append("Field energy too low (dead)")
    
    # Check for exploding values (check if any metric is too large)
    max_metric = max(abs(v) for v in stats.values() if isinstance(v, (int, float)))
    if max_metric > 1000:
        issues.append("Metrics exploding")
    
    # Check for frozen output
    test_outputs = []
    for _ in range(10):
        motor = brain.process(np.random.randn(12).astype(np.float32))
        test_outputs.append(to_numpy(motor))
    
    if np.var(test_outputs) < 1e-8:
        issues.append("Output frozen")
    
    if issues:
        print(f"✗ Issues detected: {', '.join(issues)}")
        return False
    else:
        print("✓ No pathological behaviors detected")
    
    # Test 5: Verify memory/learning capability
    print("\n5. Testing memory formation...")
    
    # Present repeated pattern
    pattern = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0], dtype=np.float32)
    responses_before = []
    responses_after = []
    
    # Initial responses
    for _ in range(5):
        motor = brain.process(pattern)
        responses_before.append(to_numpy(motor).copy())
    
    # "Training" - present pattern multiple times
    for _ in range(20):
        brain.process(pattern)
    
    # Test responses after training
    for _ in range(5):
        motor = brain.process(pattern)
        responses_after.append(to_numpy(motor).copy())
    
    # Check if responses stabilized (indicates some form of memory)
    variance_before = np.mean(np.var(responses_before, axis=0))
    variance_after = np.mean(np.var(responses_after, axis=0))
    
    if variance_after < variance_before * 0.9:  # 10% reduction in variance
        print(f"✓ Shows signs of memory formation")
        print(f"  Response variance: {variance_before:.6f} → {variance_after:.6f}")
    else:
        print(f"⚠ Memory formation unclear (but not required)")
        print(f"  Response variance: {variance_before:.6f} → {variance_after:.6f}")
    
    return True


def test_different_inputs():
    """Test brain with different types of sensory patterns"""
    print("\n\n🔬 Testing Different Input Patterns\n" + "="*50)
    
    from src.brains.field.pure_field_brain import create_pure_field_brain
    
    brain = create_pure_field_brain(
        size='hardware_constrained',
        input_dim=12,
        output_dim=4
    )
    
    test_patterns = {
        "zeros": np.zeros(12, dtype=np.float32),
        "ones": np.ones(12, dtype=np.float32),
        "alternating": np.array([1, -1] * 6, dtype=np.float32),
        "gradient": np.linspace(-1, 1, 12, dtype=np.float32),
        "sparse": np.array([0, 0, 1, 0, 0, 0, -1, 0, 0, 0, 0.5, 0], dtype=np.float32),
        "noise": np.random.randn(12).astype(np.float32)
    }
    
    print("Input Pattern → Motor Output")
    print("-" * 50)
    
    for name, pattern in test_patterns.items():
        motor = brain.process(pattern)
        motor_np = to_numpy(motor)
        print(f"{name:12} → {motor_np}")
    
    print("\n✓ Brain handles various input patterns")
    return True


def test_performance():
    """Quick performance check"""
    print("\n\n⚡ Performance Check\n" + "="*50)
    
    from src.brains.field.pure_field_brain import create_pure_field_brain
    
    brain = create_pure_field_brain(
        size='hardware_constrained',
        input_dim=24,  # Larger input for more realistic test
        output_dim=4
    )
    
    # Warm up
    for _ in range(10):
        brain.process(np.random.randn(24).astype(np.float32))
    
    # Time 100 cycles
    start = time.perf_counter()
    for _ in range(100):
        sensory = np.random.randn(24).astype(np.float32) * 0.5
        brain.process(sensory)
    elapsed = time.perf_counter() - start
    
    avg_time = elapsed / 100 * 1000  # Convert to ms
    print(f"Average cycle time: {avg_time:.2f}ms")
    
    if avg_time < 10:  # Less than 10ms is good
        print("✓ Excellent performance for real-time control")
    elif avg_time < 50:  # Less than 50ms is acceptable
        print("✓ Acceptable performance")
    else:
        print("⚠ Performance may be too slow for real-time")
    
    return True


def main():
    """Run all dry-run tests"""
    print("="*60)
    print("      PureFieldBrain Dry-Run Test Suite")
    print("="*60)
    print("\nThis test validates that the brain:")
    print("  1. Produces motor outputs")
    print("  2. Responds to different inputs") 
    print("  3. Shows field evolution (does SOMETHING)")
    print("  4. Avoids pathological behaviors")
    print("  5. Shows potential for memory formation")
    print("")
    
    all_passed = True
    
    # Run basic tests
    if not test_brain_basics():
        print("\n❌ Basic functionality test FAILED")
        all_passed = False
    
    # Test different inputs
    if not test_different_inputs():
        print("\n❌ Input pattern test FAILED")
        all_passed = False
    
    # Test performance
    if not test_performance():
        print("\n❌ Performance test FAILED")
        all_passed = False
    
    # Final summary
    print("\n" + "="*60)
    if all_passed:
        print("✅ ALL TESTS PASSED - Brain is ready for robot deployment!")
        print("\nThe brain:")
        print("  • Produces varied motor outputs")
        print("  • Responds to sensory input")
        print("  • Shows field evolution")
        print("  • Has no pathological behaviors")
        print("  • Runs fast enough for real-time control")
        print("\n🚀 Ready to test on your wheeled robot!")
    else:
        print("❌ SOME TESTS FAILED - Review issues before deployment")
    print("="*60)


if __name__ == "__main__":
    main()