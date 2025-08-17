"""
GPU Performance Profiler for Field Brain

Identifies actual bottlenecks in the field brain system.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import time
import cProfile
import pstats
from io import StringIO
from truly_minimal_brain import TrulyMinimalBrain

def profile_brain_gpu():
    """Profile the brain to identify GPU bottlenecks."""
    
    # Create brain with actual production size
    brain = TrulyMinimalBrain(
        sensory_dim=16,
        motor_dim=5,
        spatial_size=96,  # Production size: 96³×192
        channels=192,
        quiet_mode=False
    )
    
    print("\n" + "="*80)
    print("GPU PERFORMANCE PROFILING")
    print("="*80)
    
    # Warm up GPU
    print("\n🔥 Warming up GPU...")
    for _ in range(5):
        sensory_input = [0.5] * 16
        brain.process(sensory_input)
    
    # Test individual components
    print("\n📊 Component-level timing (average of 10 cycles):")
    
    component_times = {
        'sensory_injection': [],
        'learning': [],
        'tensions': [],
        'momentum': [],
        'dynamics': [],
        'motor': [],
        'prediction': [],
        'telemetry': []
    }
    
    for cycle in range(10):
        sensory_input = [0.5 + 0.1 * cycle] * 16
        
        # Time each component
        start_total = time.perf_counter()
        
        # Convert input
        sensors = torch.tensor(sensory_input[:brain.sensory_dim], 
                              dtype=torch.float32, device=brain.device)
        
        # 1. Sensory injection
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        
        if not hasattr(brain, 'sensor_spots'):
            brain.sensor_spots = torch.randint(0, brain.spatial_size, 
                                              (brain.sensory_dim, 3), 
                                              device=brain.device)
        
        for i, value in enumerate(sensors):
            if i >= brain.sensory_dim:
                break
            x, y, z = brain.sensor_spots[i]
            brain.field[x, y, z, i % 8] += value * 0.3
        
        torch.cuda.synchronize()
        component_times['sensory_injection'].append(time.perf_counter() - t0)
        
        # 2. Learning from prediction
        t0 = time.perf_counter()
        if brain.last_prediction is not None:
            error = brain.prediction.compute_error(brain.last_prediction, sensors)
            tension = brain.learning.error_to_field_tension(error, brain.field)
            brain.field = brain.field + tension
            brain.prediction.learn_from_error(error, brain.field)
            error_magnitude = torch.abs(error).mean().item()
        else:
            error_magnitude = 0.0
        torch.cuda.synchronize()
        component_times['learning'].append(time.perf_counter() - t0)
        
        # 3. Intrinsic tensions
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        brain.field = brain.tensions.apply_tensions(brain.field, error_magnitude)
        torch.cuda.synchronize()
        component_times['tensions'].append(time.perf_counter() - t0)
        
        # 4. Field momentum
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        brain.field_momentum = 0.9 * brain.field_momentum + 0.1 * brain.field
        brain.field = brain.field + brain.field_momentum * 0.05
        torch.cuda.synchronize()
        component_times['momentum'].append(time.perf_counter() - t0)
        
        # 5. Field dynamics
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        exploration = brain.learning.should_explore()
        if exploration:
            noise = torch.randn_like(brain.field) * 0.02
        else:
            noise = None
        brain.field = brain.dynamics.evolve(brain.field, noise)
        torch.cuda.synchronize()
        component_times['dynamics'].append(time.perf_counter() - t0)
        
        # 6. Motor extraction
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        motor_output = brain.motor.extract_motors(brain.field)
        torch.cuda.synchronize()
        component_times['motor'].append(time.perf_counter() - t0)
        
        # 7. Prediction
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        brain.last_prediction = brain.prediction.predict_next_sensors(brain.field)
        brain.prediction.update_history(sensors)
        torch.cuda.synchronize()
        component_times['prediction'].append(time.perf_counter() - t0)
        
        # 8. Telemetry
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        comfort = brain.tensions.get_comfort_metrics(brain.field)
        torch.cuda.synchronize()
        component_times['telemetry'].append(time.perf_counter() - t0)
        
        total_time = (time.perf_counter() - start_total) * 1000
        print(f"  Cycle {cycle+1}: {total_time:.1f}ms")
    
    # Print results
    print("\n📈 Performance Analysis:")
    print("-" * 60)
    
    total_avg = 0
    for component, times in component_times.items():
        avg_time = sum(times) / len(times) * 1000  # Convert to ms
        total_avg += avg_time
        percentage = (avg_time / total_avg * 100) if total_avg > 0 else 0
        print(f"  {component:20s}: {avg_time:8.2f}ms ({percentage:5.1f}%)")
    
    print(f"  {'TOTAL':20s}: {total_avg:8.2f}ms")
    
    # Identify bottlenecks
    print("\n🎯 Identified Bottlenecks:")
    sorted_components = sorted(component_times.items(), 
                              key=lambda x: sum(x[1])/len(x[1]), 
                              reverse=True)
    
    for component, times in sorted_components[:3]:
        avg_time = sum(times) / len(times) * 1000
        if avg_time > 100:  # More than 100ms
            print(f"  ⚠️  {component}: {avg_time:.1f}ms - CRITICAL")
        elif avg_time > 50:  # More than 50ms  
            print(f"  ⚡ {component}: {avg_time:.1f}ms - needs optimization")
        else:
            print(f"  ✓ {component}: {avg_time:.1f}ms - acceptable")
    
    # Memory analysis
    print("\n💾 Memory Usage:")
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"  Allocated: {allocated:.2f} GB")
        print(f"  Reserved:  {reserved:.2f} GB")
    
    # Check for CPU-GPU transfers
    print("\n🔄 Checking for CPU-GPU transfers...")
    check_cpu_gpu_transfers(brain)
    
    return brain

def check_cpu_gpu_transfers(brain):
    """Check for inadvertent CPU-GPU transfers."""
    
    issues = []
    
    # Check tensor operations
    if hasattr(brain, 'field'):
        if brain.field.device.type != 'cuda':
            issues.append("Field tensor not on GPU!")
    
    # Check for .item() calls (forces sync)
    import inspect
    
    # Check intrinsic tensions for .item() calls
    tensions_source = inspect.getsource(brain.tensions.apply_tensions)
    item_count = tensions_source.count('.item()')
    if item_count > 0:
        issues.append(f"Found {item_count} .item() calls in apply_tensions (forces GPU sync)")
    
    # Check for Python min/max instead of torch operations
    if 'min(' in tensions_source or 'max(' in tensions_source:
        issues.append("Using Python min/max instead of torch operations")
    
    # Check motor extraction
    motor_source = inspect.getsource(brain.motor.extract_motors)
    if '.cpu()' in motor_source:
        issues.append("CPU transfer in motor extraction (expected at end)")
    
    if issues:
        print("  Issues found:")
        for issue in issues:
            print(f"    ⚠️  {issue}")
    else:
        print("  ✅ No major CPU-GPU transfer issues detected")

if __name__ == "__main__":
    brain = profile_brain_gpu()
    
    print("\n" + "="*80)
    print("PROFILING COMPLETE")
    print("="*80)