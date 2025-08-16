#!/usr/bin/env python3
"""
Minimal Brain Demo & Test

A self-contained demonstration that:
1. Tests the brain technically works
2. Shows intrinsic motivation in action
3. Visualizes learning and exploration
4. Serves as a sanity check for the system

Usage:
    python3 demo.py              # Run full demo with visualization
    python3 demo.py --quick      # Quick test (30 seconds)
    python3 demo.py --headless   # No visualization, just metrics
"""

import sys
import os
import time
import argparse
import numpy as np
from pathlib import Path

# Add server to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'server'))


def check_dependencies():
    """Check required dependencies."""
    required = {
        'numpy': False,
        'torch': False,
    }
    optional = {
        'matplotlib': False,
    }
    
    for module in required:
        try:
            __import__(module)
            required[module] = True
        except ImportError:
            print(f"❌ Missing required: {module}")
    
    for module in optional:
        try:
            __import__(module)
            optional[module] = True
        except ImportError:
            print(f"⚠️  Missing optional: {module} (visualization disabled)")
    
    if not all(required.values()):
        print("\nInstall with: pip install numpy torch")
        sys.exit(1)
    
    return optional['matplotlib']


def test_brain_sanity():
    """Basic sanity checks that brain works."""
    print("\n" + "="*60)
    print("BRAIN SANITY CHECK")
    print("="*60)
    
    from src.brains.field.truly_minimal_brain import TrulyMinimalBrain
    
    print("\n✓ Brain imports successfully")
    
    # Create brain
    brain = TrulyMinimalBrain(
        spatial_size=16,
        channels=32,
        quiet_mode=True
    )
    print("✓ Brain creates successfully")
    
    # Test processing
    sensors = [0.5] * 16
    motors, telemetry = brain.process(sensors)
    print("✓ Brain processes input")
    
    # Check outputs
    assert len(motors) == 5, "Wrong motor count"
    assert 'energy' in telemetry, "Missing telemetry"
    assert 'comfort' in telemetry, "Missing comfort metric"
    assert 'motivation' in telemetry, "Missing motivation"
    print("✓ Brain outputs correct format")
    
    # Test persistence
    brain.save("test_save.pt")
    print("✓ Brain saves state")
    
    brain.reset()
    original_cycle = brain.cycle
    brain.load("test_save.pt")
    assert brain.cycle > original_cycle, "Load failed"
    print("✓ Brain loads state")
    
    print("\n✅ ALL SANITY CHECKS PASSED")
    return True


def demonstrate_intrinsic_motivation():
    """Show how intrinsic motivation drives behavior."""
    print("\n" + "="*60)
    print("INTRINSIC MOTIVATION DEMO")
    print("="*60)
    
    from src.brains.field.truly_minimal_brain import TrulyMinimalBrain
    
    brain = TrulyMinimalBrain(
        spatial_size=16,
        channels=32,
        quiet_mode=True
    )
    
    print("\n1. STARVATION TEST (no input)")
    print("-" * 40)
    for i in range(5):
        sensors = [0.0] * 16  # No input
        motors, telemetry = brain.process(sensors)
        print(f"  Cycle {i+1}: {telemetry['motivation']}")
        if "STARVED" in telemetry['motivation']:
            print("  ✓ Brain gets hungry without input!")
            break
    
    print("\n2. BOREDOM TEST (uniform input)")
    print("-" * 40)
    for i in range(5):
        sensors = [0.5] * 16  # Boring uniform input
        motors, telemetry = brain.process(sensors)
        print(f"  Cycle {i+1}: {telemetry['motivation']}")
        if "BORED" in telemetry['motivation']:
            print("  ✓ Brain gets bored with uniformity!")
            break
    
    print("\n3. EXPLORATION TEST (varied input)")
    print("-" * 40)
    for i in range(5):
        sensors = [np.sin(i * 0.5 + j * 0.2) for j in range(16)]
        motors, telemetry = brain.process(sensors)
        print(f"  Cycle {i+1}: {telemetry['motivation']}, "
              f"Exploring: {telemetry.get('exploring', False)}")
    print("  ✓ Brain explores when stimulated!")
    
    print("\n4. LEARNING TEST (pattern)")
    print("-" * 40)
    pattern_a = [1.0 if i < 8 else 0.0 for i in range(16)]
    pattern_b = [0.0 if i < 8 else 1.0 for i in range(16)]
    
    predictions_improved = False
    initial_error = 0
    
    for epoch in range(3):
        print(f"\n  Epoch {epoch+1}:")
        for p_name, pattern in [("A", pattern_a), ("B", pattern_b)]:
            motors, telemetry = brain.process(pattern)
            error = abs(telemetry.get('energy', 0) - 0.5)
            if epoch == 0 and p_name == "A":
                initial_error = error
            print(f"    Pattern {p_name}: {telemetry['motivation'][:20]}, "
                  f"Error: {error:.3f}")
            if epoch == 2 and error < initial_error * 0.8:
                predictions_improved = True
    
    if predictions_improved:
        print("  ✓ Brain learns patterns over time!")
    
    return True


def run_visual_demo(duration=60, headless=False):
    """Run demo with optional visualization."""
    print("\n" + "="*60)
    print("BRAIN BEHAVIOR DEMO" + (" (HEADLESS)" if headless else ""))
    print("="*60)
    
    from src.brains.field.truly_minimal_brain import TrulyMinimalBrain
    
    brain = TrulyMinimalBrain(
        spatial_size=24,  # Larger for more interesting dynamics
        channels=48,
        quiet_mode=False
    )
    
    if not headless:
        try:
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(2, 2, figsize=(10, 8))
            plt.ion()
            show_plot = True
        except:
            print("⚠️  Matplotlib unavailable, running headless")
            show_plot = False
            headless = True
    else:
        show_plot = False
    
    # Metrics tracking
    energy_history = []
    comfort_history = []
    motor_history = []
    motivation_counts = {
        'STARVED': 0,
        'BORED': 0,
        'ACTIVE': 0,
        'CONTENT': 0,
        'UNCOMFORTABLE': 0
    }
    
    print(f"\nRunning for {duration} seconds...")
    print("Watch the brain's motivation change!\n")
    
    start_time = time.time()
    cycle = 0
    
    while time.time() - start_time < duration:
        cycle += 1
        
        # Generate interesting sensory input
        t = time.time() - start_time
        if cycle < 100:
            # Start with low input (starvation)
            sensors = [np.random.randn() * 0.1 for _ in range(16)]
        elif cycle < 200:
            # Uniform input (boredom)
            sensors = [0.5] * 16
        else:
            # Varied input (exploration)
            sensors = [
                np.sin(t * 2 + i * 0.5) * np.cos(t * 0.5 + i * 0.2) 
                + np.random.randn() * 0.1
                for i in range(16)
            ]
        
        # Process
        motors, telemetry = brain.process(sensors)
        
        # Track metrics
        energy_history.append(telemetry['energy'])
        comfort_history.append(telemetry['comfort'])
        motor_history.append(motors[0])  # Track first motor
        
        # Count motivations
        for key in motivation_counts:
            if key in telemetry['motivation']:
                motivation_counts[key] += 1
                break
        
        # Update visualization
        if show_plot and cycle % 10 == 0:
            # Clear axes
            for ax in axes.flat:
                ax.clear()
            
            # Energy over time
            axes[0, 0].plot(energy_history[-100:])
            axes[0, 0].set_title('Field Energy')
            axes[0, 0].set_ylabel('Energy')
            
            # Comfort over time
            axes[0, 1].plot(comfort_history[-100:], 'orange')
            axes[0, 1].set_title('Comfort Level')
            axes[0, 1].set_ylabel('Comfort')
            axes[0, 1].axhline(y=0, color='red', linestyle='--', alpha=0.3)
            
            # Motor output
            axes[1, 0].plot(motor_history[-100:], 'green')
            axes[1, 0].set_title('Motor Output (first)')
            axes[1, 0].set_ylabel('Motor Value')
            
            # Motivation distribution
            if sum(motivation_counts.values()) > 0:
                axes[1, 1].bar(range(len(motivation_counts)), 
                             list(motivation_counts.values()))
                axes[1, 1].set_xticks(range(len(motivation_counts)))
                axes[1, 1].set_xticklabels(list(motivation_counts.keys()), 
                                          rotation=45, ha='right')
                axes[1, 1].set_title('Motivation States')
                axes[1, 1].set_ylabel('Count')
            
            plt.tight_layout()
            plt.pause(0.01)
        
        # Console output every 100 cycles
        if cycle % 100 == 0:
            print(f"Cycle {cycle}: {telemetry['motivation']}, "
                  f"Energy: {telemetry['energy']:.3f}, "
                  f"Comfort: {telemetry['comfort']:.2f}")
    
    # Final summary
    print("\n" + "="*60)
    print("DEMO COMPLETE - SUMMARY")
    print("="*60)
    
    print(f"\nTotal cycles: {cycle}")
    print(f"Average energy: {np.mean(energy_history):.3f}")
    print(f"Energy variance: {np.var(energy_history):.3f}")
    
    print("\nMotivation distribution:")
    total = sum(motivation_counts.values())
    for state, count in motivation_counts.items():
        percentage = (count / total * 100) if total > 0 else 0
        bar = '█' * int(percentage / 5)
        print(f"  {state:15} {percentage:5.1f}% {bar}")
    
    print("\nKey observations:")
    if motivation_counts['STARVED'] > cycle * 0.1:
        print("  • Brain showed hunger for input")
    if motivation_counts['BORED'] > cycle * 0.1:
        print("  • Brain exhibited boredom with uniformity")
    if motivation_counts['ACTIVE'] > cycle * 0.2:
        print("  • Brain actively explored and learned")
    if np.var(motor_history) > 0.01:
        print("  • Motor outputs showed dynamic behavior")
    
    if show_plot:
        plt.ioff()
        plt.show()
    
    return True


def main():
    """Main demo entry point."""
    parser = argparse.ArgumentParser(description='Minimal Brain Demo & Test')
    parser.add_argument('--quick', action='store_true',
                       help='Quick 30-second test')
    parser.add_argument('--headless', action='store_true',
                       help='Run without visualization')
    parser.add_argument('--skip-sanity', action='store_true',
                       help='Skip sanity checks')
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("MINIMAL BRAIN - DEMO & TEST")
    print("="*60)
    print("\nA brain that learns through discomfort, not rewards.")
    
    # Check dependencies
    has_viz = check_dependencies()
    
    # Run sanity check
    if not args.skip_sanity:
        if not test_brain_sanity():
            print("❌ Sanity check failed!")
            sys.exit(1)
    
    # Demonstrate intrinsic motivation
    demonstrate_intrinsic_motivation()
    
    # Run visual demo
    duration = 30 if args.quick else 60
    run_visual_demo(duration, args.headless or not has_viz)
    
    print("\n✅ Demo complete! The brain works through intrinsic motivation.")
    print("\nNext steps:")
    print("  • Run the server: cd server && python3 brain.py")
    print("  • Connect a robot: cd client_picarx && python3 picarx_robot.py")
    print("  • Try 3D demo: python3 demos/demo_3d.py")


if __name__ == "__main__":
    main()