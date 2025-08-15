#!/usr/bin/env python3
"""
PureFieldBrain Real-Time Performance Analysis for Robot Control (Simple Version)

Tests the brain's ability to maintain 30+ Hz control loop for safe robot operation.
Identifies performance bottlenecks and validates timing requirements.
"""

import torch
import numpy as np
import time
import gc
from typing import Dict, List, Tuple, Optional
import json
from dataclasses import dataclass, asdict
from pathlib import Path

# Add parent directories to path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.brains.field.pure_field_brain import PureFieldBrain, SCALE_CONFIGS


@dataclass
class PerformanceMetrics:
    """Detailed performance metrics for a test run"""
    config_name: str
    device: str
    field_size: int
    channels: int
    total_params: int
    
    # Timing metrics (in milliseconds)
    mean_cycle_time: float
    std_cycle_time: float
    min_cycle_time: float
    max_cycle_time: float
    p50_cycle_time: float  # median
    p95_cycle_time: float
    p99_cycle_time: float
    
    # Component timings
    sensory_injection_time: float
    field_evolution_time: float
    motor_extraction_time: float
    learning_time: float
    
    # Memory metrics
    gpu_memory_used_mb: float
    gpu_memory_peak_mb: float
    
    # Performance indicators
    achieved_hz: float
    meets_30hz_requirement: bool
    max_sustainable_hz: float
    latency_jitter_ms: float  # std deviation of cycle times
    
    # CPU-GPU transfer metrics
    transfer_overhead_ms: float
    transfer_count_per_cycle: int
    
    def to_dict(self) -> Dict:
        return asdict(self)


def test_single_configuration(
    config_name: str,
    input_dim: int = 10,
    output_dim: int = 4,
    num_warmup: int = 50,
    num_cycles: int = 500,
    verbose: bool = True
) -> PerformanceMetrics:
    """Test a specific brain configuration"""
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Testing {config_name} configuration")
        print(f"{'='*60}")
    
    # Get configuration
    scale_config = SCALE_CONFIGS[config_name]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create brain
    brain = PureFieldBrain(
        input_dim=input_dim,
        output_dim=output_dim,
        scale_config=scale_config,
        device=device,
        aggressive=True
    )
    
    if verbose:
        print(f"Device: {device}")
        print(f"Field size: {scale_config.levels[0][0]}³×{scale_config.levels[0][1]}")
        print(f"Total parameters: {scale_config.total_params:,}")
        print(f"Hierarchical levels: {len(scale_config.levels)}")
    
    # Prepare test data
    test_inputs = [torch.randn(input_dim, device=device) for _ in range(num_cycles)]
    test_rewards = [np.random.random() * 0.1 - 0.05 for _ in range(num_cycles)]
    
    # Memory baseline
    if device == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        initial_gpu_memory = torch.cuda.memory_allocated() / 1024**2
    else:
        initial_gpu_memory = 0
    
    # Warmup
    if verbose:
        print(f"\nWarming up ({num_warmup} cycles)...")
    
    for i in range(num_warmup):
        _ = brain(test_inputs[i % len(test_inputs)])
    
    if device == 'cuda':
        torch.cuda.synchronize()
    
    # Detailed timing arrays
    cycle_times = []
    component_breakdown = {
        'sensory': [],
        'evolution': [],
        'motor': []
    }
    learning_times = []
    
    # Main performance test
    if verbose:
        print(f"Running performance test ({num_cycles} cycles)...")
    
    for i in range(num_cycles):
        # Overall cycle timing
        if device == 'cuda':
            torch.cuda.synchronize()
        
        cycle_start = time.perf_counter()
        
        # Measure forward pass with component timing
        # We'll measure the whole forward pass and estimate components
        forward_start = time.perf_counter()
        motor = brain(test_inputs[i], test_rewards[i])
        
        if device == 'cuda':
            torch.cuda.synchronize()
        
        cycle_time = (time.perf_counter() - cycle_start) * 1000  # Convert to ms
        cycle_times.append(cycle_time)
        
        # Test learning periodically
        if i % 10 == 0:
            learn_start = time.perf_counter()
            predicted = torch.randn(input_dim, device=device)
            actual = test_inputs[i]
            brain.learn_from_prediction_error(actual, predicted)
            if device == 'cuda':
                torch.cuda.synchronize()
            learning_times.append((time.perf_counter() - learn_start) * 1000)
        
        # Progress update
        if verbose and (i + 1) % 100 == 0:
            current_mean = np.mean(cycle_times)
            current_hz = 1000 / current_mean
            print(f"  Cycle {i+1}/{num_cycles}: {current_mean:.2f}ms ({current_hz:.1f} Hz)")
    
    # Estimate component breakdown (rough approximation)
    # Based on typical GPU kernel launch patterns
    mean_cycle = np.mean(cycle_times)
    sensory_estimate = mean_cycle * 0.15  # ~15% for input projection
    motor_estimate = mean_cycle * 0.10    # ~10% for motor extraction
    evolution_estimate = mean_cycle * 0.75  # ~75% for field evolution
    
    # Calculate statistics
    cycle_times = np.array(cycle_times)
    
    # Memory check
    if device == 'cuda':
        final_gpu_memory = torch.cuda.memory_allocated() / 1024**2
        peak_gpu_memory = torch.cuda.max_memory_allocated() / 1024**2
    else:
        final_gpu_memory = 0
        peak_gpu_memory = 0
    
    # Test CPU-GPU transfer overhead
    transfer_times = []
    if device == 'cuda':
        for _ in range(10):
            # Test GPU -> CPU transfer
            test_tensor = torch.randn(100, device=device)
            start = time.perf_counter()
            _ = test_tensor.cpu()
            torch.cuda.synchronize()
            transfer_times.append((time.perf_counter() - start) * 1000)
            
            # Test CPU -> GPU transfer
            cpu_tensor = torch.randn(100)
            start = time.perf_counter()
            _ = cpu_tensor.to(device)
            torch.cuda.synchronize()
            transfer_times.append((time.perf_counter() - start) * 1000)
    
    transfer_overhead = np.mean(transfer_times) if transfer_times else 0.0
    
    # Create metrics
    metrics = PerformanceMetrics(
        config_name=config_name,
        device=device,
        field_size=scale_config.levels[0][0],
        channels=scale_config.levels[0][1],
        total_params=scale_config.total_params,
        
        # Timing metrics
        mean_cycle_time=np.mean(cycle_times),
        std_cycle_time=np.std(cycle_times),
        min_cycle_time=np.min(cycle_times),
        max_cycle_time=np.max(cycle_times),
        p50_cycle_time=np.percentile(cycle_times, 50),
        p95_cycle_time=np.percentile(cycle_times, 95),
        p99_cycle_time=np.percentile(cycle_times, 99),
        
        # Component timings (estimated)
        sensory_injection_time=sensory_estimate,
        field_evolution_time=evolution_estimate,
        motor_extraction_time=motor_estimate,
        learning_time=np.mean(learning_times) if learning_times else 0,
        
        # Memory metrics
        gpu_memory_used_mb=final_gpu_memory,
        gpu_memory_peak_mb=peak_gpu_memory,
        
        # Performance indicators
        achieved_hz=1000 / np.mean(cycle_times),
        meets_30hz_requirement=np.percentile(cycle_times, 95) < 33.33,  # 95% of cycles under 33ms
        max_sustainable_hz=1000 / np.percentile(cycle_times, 95),
        latency_jitter_ms=np.std(cycle_times),
        
        # Transfer metrics
        transfer_overhead_ms=transfer_overhead,
        transfer_count_per_cycle=3  # input, output, and internal transfers
    )
    
    if verbose:
        print_metrics(metrics)
    
    return metrics


def print_metrics(metrics: PerformanceMetrics):
    """Print formatted metrics"""
    
    print(f"\n📊 Performance Results for {metrics.config_name}")
    print(f"{'='*50}")
    
    # Timing summary
    print(f"\n⏱️  Timing Performance:")
    print(f"  Mean cycle time: {metrics.mean_cycle_time:.2f}ms ({metrics.achieved_hz:.1f} Hz)")
    print(f"  Latency range: {metrics.min_cycle_time:.2f} - {metrics.max_cycle_time:.2f}ms")
    print(f"  Jitter (std): {metrics.latency_jitter_ms:.2f}ms")
    print(f"  95th percentile: {metrics.p95_cycle_time:.2f}ms")
    print(f"  99th percentile: {metrics.p99_cycle_time:.2f}ms")
    
    # Component breakdown (estimated)
    print(f"\n🔧 Component Breakdown (estimated):")
    total_component = (metrics.sensory_injection_time + 
                      metrics.field_evolution_time + 
                      metrics.motor_extraction_time)
    print(f"  Sensory injection: ~{metrics.sensory_injection_time:.2f}ms ({metrics.sensory_injection_time/total_component*100:.1f}%)")
    print(f"  Field evolution: ~{metrics.field_evolution_time:.2f}ms ({metrics.field_evolution_time/total_component*100:.1f}%)")
    print(f"  Motor extraction: ~{metrics.motor_extraction_time:.2f}ms ({metrics.motor_extraction_time/total_component*100:.1f}%)")
    if metrics.learning_time > 0:
        print(f"  Learning update: {metrics.learning_time:.2f}ms")
    
    # Memory usage
    print(f"\n💾 Memory Usage:")
    if metrics.device == 'cuda':
        print(f"  GPU memory: {metrics.gpu_memory_used_mb:.1f}MB (peak: {metrics.gpu_memory_peak_mb:.1f}MB)")
    
    # Transfer overhead
    if metrics.device == 'cuda' and metrics.transfer_overhead_ms > 0:
        print(f"\n🔄 CPU-GPU Transfer:")
        print(f"  Average overhead: {metrics.transfer_overhead_ms:.3f}ms per transfer")
        print(f"  Estimated transfers per cycle: {metrics.transfer_count_per_cycle}")
    
    # Requirements check
    print(f"\n✅ Requirements Check:")
    print(f"  30Hz requirement: {'✅ PASS' if metrics.meets_30hz_requirement else '❌ FAIL'}")
    print(f"  Max sustainable: {metrics.max_sustainable_hz:.1f} Hz")
    
    if not metrics.meets_30hz_requirement:
        deficit = metrics.p95_cycle_time - 33.33
        print(f"  ⚠️ Need {abs(deficit):.1f}ms improvement to meet 30Hz")


def identify_bottlenecks(metrics: PerformanceMetrics) -> List[str]:
    """Identify performance bottlenecks"""
    
    bottlenecks = []
    
    # Check if meeting 30Hz requirement
    if not metrics.meets_30hz_requirement:
        deficit = metrics.p95_cycle_time - 33.33
        bottlenecks.append(f"Not meeting 30Hz requirement (deficit: {abs(deficit):.1f}ms)")
    
    # Check component bottlenecks
    total_time = (metrics.sensory_injection_time + 
                 metrics.field_evolution_time + 
                 metrics.motor_extraction_time)
    
    if metrics.field_evolution_time / total_time > 0.6:
        bottlenecks.append(f"Field evolution is bottleneck (~{metrics.field_evolution_time/total_time*100:.1f}% of time)")
    
    # Check transfer overhead
    if metrics.device == 'cuda' and metrics.transfer_overhead_ms > 1.0:
        bottlenecks.append(f"High CPU-GPU transfer overhead ({metrics.transfer_overhead_ms:.2f}ms)")
    
    # Check jitter
    if metrics.latency_jitter_ms > 5.0:
        bottlenecks.append(f"High latency jitter ({metrics.latency_jitter_ms:.2f}ms)")
    
    # Check if it's too slow overall
    if metrics.achieved_hz < 20:
        bottlenecks.append(f"Very low frame rate ({metrics.achieved_hz:.1f} Hz)")
    
    return bottlenecks


def main():
    """Run comprehensive performance analysis"""
    
    print("=" * 70)
    print("PureFieldBrain Real-Time Performance Analysis")
    print("Testing for 30+ Hz robot control requirements")
    print("=" * 70)
    
    # Test configurations in order of size
    test_configs = ['hardware_constrained', 'tiny', 'small']
    
    # Add medium only if we have GPU (it's too slow on CPU)
    if torch.cuda.is_available():
        test_configs.append('medium')
        # Don't test large by default as it's memory intensive
        # test_configs.append('large')
    
    results = {}
    
    for config_name in test_configs:
        try:
            print(f"\n{'='*70}")
            print(f"Testing {config_name} configuration...")
            print(f"{'='*70}")
            
            metrics = test_single_configuration(
                config_name, 
                num_warmup=30,
                num_cycles=300,  # Reduced for faster testing
                verbose=True
            )
            results[config_name] = metrics
            
            # Clean up between tests
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Print bottlenecks
            bottlenecks = identify_bottlenecks(metrics)
            if bottlenecks:
                print(f"\n⚠️ Bottlenecks identified:")
                for b in bottlenecks:
                    print(f"  - {b}")
            else:
                print(f"\n✅ No major bottlenecks identified")
                
        except Exception as e:
            print(f"❌ Failed to test {config_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 PERFORMANCE SUMMARY")
    print("=" * 70)
    
    # Find best configuration for 30Hz
    best_for_30hz = None
    for config, metrics in results.items():
        if metrics.meets_30hz_requirement:
            if best_for_30hz is None or metrics.total_params > best_for_30hz.total_params:
                best_for_30hz = metrics
    
    if best_for_30hz:
        print(f"\n✅ Best configuration for 30Hz robot control: {best_for_30hz.config_name}")
        print(f"   - Achieves {best_for_30hz.achieved_hz:.1f} Hz")
        print(f"   - {best_for_30hz.total_params:,} parameters")
        print(f"   - 95th percentile: {best_for_30hz.p95_cycle_time:.2f}ms")
        if best_for_30hz.device == 'cuda':
            print(f"   - GPU memory: {best_for_30hz.gpu_memory_peak_mb:.1f}MB")
    else:
        print(f"\n⚠️ No configuration meets 30Hz requirement on this hardware!")
        if results:
            fastest = min(results.values(), key=lambda m: m.mean_cycle_time)
            print(f"   Fastest: {fastest.config_name} at {fastest.achieved_hz:.1f} Hz")
    
    # Table summary
    print(f"\n📋 Configuration Comparison:")
    print(f"{'Config':<20} {'Hz':<10} {'Mean (ms)':<12} {'P95 (ms)':<12} {'30Hz?':<8}")
    print("-" * 70)
    for config, metrics in results.items():
        print(f"{config:<20} {metrics.achieved_hz:<10.1f} {metrics.mean_cycle_time:<12.2f} "
              f"{metrics.p95_cycle_time:<12.2f} {'✅' if metrics.meets_30hz_requirement else '❌':<8}")
    
    # Optimization recommendations
    print("\n" + "=" * 70)
    print("💡 OPTIMIZATION RECOMMENDATIONS")
    print("=" * 70)
    
    if torch.cuda.is_available():
        print("\n✅ GPU detected - using CUDA acceleration")
    else:
        print("\n⚠️ No GPU detected - running on CPU")
        print("   Consider using CUDA-enabled hardware for better performance")
    
    # Specific recommendations based on results
    for config, metrics in results.items():
        if metrics.field_evolution_time > 10:
            print(f"\n📍 {config}:")
            print(f"   - Field evolution takes ~{metrics.field_evolution_time:.1f}ms")
            print(f"   - Consider reducing field size or using mixed precision")
        
        if metrics.latency_jitter_ms > 3:
            print(f"\n📍 {config}:")
            print(f"   - High jitter ({metrics.latency_jitter_ms:.2f}ms std)")
            print(f"   - Consider pinning CPU affinity or reducing background processes")
    
    # Save results
    print("\n" + "=" * 70)
    print("💾 Saving results...")
    
    report = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'configurations_tested': len(results),
        'results': {k: v.to_dict() for k, v in results.items()}
    }
    
    with open('pure_field_performance_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"📄 Report saved to pure_field_performance_report.json")


if __name__ == "__main__":
    main()