#!/usr/bin/env python3
"""
GPU Performance Profiler for Brain
Identifies the exact bottleneck causing 9+ second cycles at 96³×192
"""

import torch
import time
import sys
import os

# Add server to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'server', 'src'))

from brains.field.truly_minimal_brain import TrulyMinimalBrain
from brains.field.intrinsic_tensions import IntrinsicTensions
from brains.field.simple_field_dynamics import SimpleFieldDynamics
from brains.field.simple_motor import SimpleMotorExtraction

def profile_component(name, func, *args, warmup=3, iterations=10):
    """Profile a single component."""
    # Warmup
    for _ in range(warmup):
        func(*args)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    
    # Time
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        result = func(*args)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)
    
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    
    print(f"{name:40} avg: {avg_time:8.2f}ms  min: {min_time:8.2f}ms  max: {max_time:8.2f}ms")
    return result

def profile_brain_operations(spatial_size=96, channels=192):
    """Profile all brain operations individually."""
    print(f"\n{'='*80}")
    print(f"PROFILING BRAIN COMPONENTS: {spatial_size}³×{channels}")
    print(f"Total parameters: {spatial_size**3 * channels:,}")
    print(f"Memory size: {spatial_size**3 * channels * 4 / 1024**3:.2f} GB")
    print(f"{'='*80}\n")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print()
    
    # Create field tensor
    print("Creating tensors...")
    field = profile_component(
        "Create field tensor",
        lambda: torch.randn(spatial_size, spatial_size, spatial_size, channels, device=device) * 0.01
    )
    
    field_momentum = torch.zeros_like(field)
    
    # Initialize components
    print("\nInitializing components...")
    dynamics = SimpleFieldDynamics()
    tensions = IntrinsicTensions(field.shape, device)
    motor = SimpleMotorExtraction(5, device, spatial_size)
    
    # Profile individual operations
    print("\n" + "-"*80)
    print("BASIC TENSOR OPERATIONS")
    print("-"*80)
    
    profile_component("field.mean()", lambda: field.mean())
    profile_component("field.var()", lambda: field.var())
    profile_component("torch.abs(field).mean()", lambda: torch.abs(field).mean())
    profile_component("field * 0.995 (decay)", lambda: field * 0.995)
    profile_component("field + field * 0.1 (add)", lambda: field + field * 0.1)
    profile_component("torch.randn_like(field)", lambda: torch.randn_like(field))
    
    print("\n" + "-"*80)
    print("GRADIENT OPERATIONS")
    print("-"*80)
    
    profile_component(
        "torch.diff(field, dim=0)",
        lambda: torch.diff(field, dim=0, prepend=field[:1])
    )
    
    def compute_all_gradients():
        dx = torch.diff(field, dim=0, prepend=field[:1])
        dy = torch.diff(field, dim=1, prepend=field[:, :1])
        dz = torch.diff(field, dim=2, prepend=field[:, :, :1])
        return dx, dy, dz
    
    profile_component("Compute all 3 gradients", compute_all_gradients)
    
    dx, dy, dz = compute_all_gradients()
    profile_component(
        "Gradient magnitude",
        lambda: torch.sqrt(dx**2 + dy**2 + dz**2)
    )
    
    print("\n" + "-"*80)
    print("COMPONENT METHODS")
    print("-"*80)
    
    # Profile IntrinsicTensions methods
    profile_component(
        "tensions._compute_local_variance()",
        tensions._compute_local_variance, field
    )
    
    profile_component(
        "tensions._compute_gradients()",
        tensions._compute_gradients, field
    )
    
    profile_component(
        "tensions.apply_tensions()",
        tensions.apply_tensions, field, 0.1
    )
    
    # Profile SimpleFieldDynamics methods
    profile_component(
        "dynamics._apply_diffusion()",
        dynamics._apply_diffusion, field
    )
    
    profile_component(
        "dynamics.evolve()",
        dynamics.evolve, field
    )
    
    # Profile motor extraction
    profile_component(
        "motor.extract_motors()",
        motor.extract_motors, field
    )
    
    print("\n" + "-"*80)
    print("FULL BRAIN CYCLE")
    print("-"*80)
    
    # Create brain and profile full cycle
    brain = TrulyMinimalBrain(
        sensory_dim=16,
        motor_dim=5,
        spatial_size=spatial_size,
        channels=channels,
        device=device,
        quiet_mode=True
    )
    
    sensory_input = [0.5] * 16
    
    def brain_cycle():
        return brain.process(sensory_input)
    
    profile_component("brain.process() FULL CYCLE", brain_cycle)
    
    # Now profile individual parts of the process method
    print("\n" + "-"*80)
    print("DETAILED BRAIN.PROCESS() BREAKDOWN")
    print("-"*80)
    
    # Manually trace through the process method
    sensors = torch.tensor(sensory_input[:16], dtype=torch.float32, device=device)
    
    # Sensory injection
    if not hasattr(brain, 'sensor_spots'):
        brain.sensor_spots = torch.randint(0, spatial_size, (16, 3), device=device)
    
    def inject_sensors():
        for i, value in enumerate(sensors):
            if i >= 16:
                break
            x, y, z = brain.sensor_spots[i]
            brain.field[x, y, z, i % 8] += value * 0.3
    
    profile_component("Sensory injection loop", inject_sensors)
    
    # Field momentum update
    def update_momentum():
        brain.field_momentum = 0.9 * brain.field_momentum + 0.1 * brain.field
        brain.field = brain.field + brain.field_momentum * 0.05
    
    profile_component("Field momentum update", update_momentum)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)

def compare_sizes():
    """Compare different field sizes to find the performance cliff."""
    print("\n" + "="*80)
    print("COMPARING DIFFERENT FIELD SIZES")
    print("="*80)
    
    sizes = [
        (32, 64),   # 2M params
        (48, 96),   # 10M params  
        (64, 128),  # 33M params
        (80, 160),  # 82M params
        (96, 192),  # 170M params
    ]
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    for spatial_size, channels in sizes:
        params = spatial_size**3 * channels
        memory_gb = params * 4 / 1024**3
        
        print(f"\nSize: {spatial_size}³×{channels} = {params:,} params ({memory_gb:.2f} GB)")
        
        brain = TrulyMinimalBrain(
            sensory_dim=16,
            motor_dim=5,
            spatial_size=spatial_size,
            channels=channels,
            device=device,
            quiet_mode=True
        )
        
        sensory_input = [0.5] * 16
        
        # Warmup
        for _ in range(3):
            brain.process(sensory_input)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        
        # Time
        times = []
        for _ in range(10):
            start = time.perf_counter()
            brain.process(sensory_input)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000)
        
        avg_time = sum(times) / len(times)
        print(f"  Average cycle time: {avg_time:.2f} ms")
        print(f"  Theoretical Hz: {1000/avg_time:.1f}")
        
        # Cleanup
        del brain
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    # First compare sizes to confirm the issue
    compare_sizes()
    
    # Then profile the problematic size in detail
    print("\n" + "="*80)
    print("DETAILED PROFILING OF PROBLEMATIC SIZE")
    print("="*80)
    profile_brain_operations(96, 192)
    
    # Also profile the working size for comparison
    print("\n" + "="*80)
    print("PROFILING WORKING SIZE FOR COMPARISON")
    print("="*80)
    profile_brain_operations(64, 128)