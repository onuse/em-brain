#!/usr/bin/env python3
"""
PureFieldBrain Real-Time Performance Analysis for Robot Control

Tests the brain's ability to maintain 30+ Hz control loop for safe robot operation.
Identifies performance bottlenecks and validates timing requirements.
"""

import torch
import numpy as np
import time
import psutil
import gc
from typing import Dict, List, Tuple, Optional
import json
from dataclasses import dataclass, asdict
import matplotlib.pyplot as plt
import seaborn as sns
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
    cpu_memory_used_mb: float
    memory_leak_detected: bool
    
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


class PureFieldPerformanceTester:
    """Comprehensive performance tester for PureFieldBrain"""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.results: List[PerformanceMetrics] = []
        
    def test_configuration(
        self,
        config_name: str,
        input_dim: int = 10,
        output_dim: int = 4,
        num_warmup: int = 50,
        num_cycles: int = 1000,
        test_learning: bool = True
    ) -> PerformanceMetrics:
        """Test a specific brain configuration"""
        
        if self.verbose:
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
        
        if self.verbose:
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
        
        process = psutil.Process()
        initial_cpu_memory = process.memory_info().rss / 1024**2
        
        # Warmup
        if self.verbose:
            print(f"\nWarming up ({num_warmup} cycles)...")
        
        for i in range(num_warmup):
            _ = brain(test_inputs[i % len(test_inputs)])
        
        if device == 'cuda':
            torch.cuda.synchronize()
        
        # Detailed timing arrays
        cycle_times = []
        sensory_times = []
        evolution_times = []
        motor_times = []
        learning_times = []
        
        # Main performance test
        if self.verbose:
            print(f"Running performance test ({num_cycles} cycles)...")
        
        for i in range(num_cycles):
            # Overall cycle timing
            if device == 'cuda':
                torch.cuda.synchronize()
            
            cycle_start = time.perf_counter()
            
            # Component timing using monkey patching
            component_times = {}
            
            # Patch methods to measure timing
            original_inject = brain._inject_sensory_hierarchical
            original_evolve = brain._evolve_hierarchical
            original_extract = brain._extract_motor_hierarchical
            
            def timed_inject(sensory_input):
                start = time.perf_counter()
                result = original_inject(sensory_input)
                if device == 'cuda':
                    torch.cuda.synchronize()
                component_times['sensory'] = (time.perf_counter() - start) * 1000
                return result
            
            def timed_evolve(reward):
                start = time.perf_counter()
                result = original_evolve(reward)
                if device == 'cuda':
                    torch.cuda.synchronize()
                component_times['evolution'] = (time.perf_counter() - start) * 1000
                return result
            
            def timed_extract():
                start = time.perf_counter()
                result = original_extract()
                if device == 'cuda':
                    torch.cuda.synchronize()
                component_times['motor'] = (time.perf_counter() - start) * 1000
                return result
            
            # Temporarily replace methods
            brain._inject_sensory_hierarchical = timed_inject
            brain._evolve_hierarchical = timed_evolve
            brain._extract_motor_hierarchical = timed_extract
            
            # Run forward pass
            motor = brain(test_inputs[i], test_rewards[i])
            
            # Restore original methods
            brain._inject_sensory_hierarchical = original_inject
            brain._evolve_hierarchical = original_evolve
            brain._extract_motor_hierarchical = original_extract
            
            if device == 'cuda':
                torch.cuda.synchronize()
            
            cycle_time = (time.perf_counter() - cycle_start) * 1000  # Convert to ms
            cycle_times.append(cycle_time)
            
            # Store component times
            sensory_times.append(component_times.get('sensory', 0))
            evolution_times.append(component_times.get('evolution', 0))
            motor_times.append(component_times.get('motor', 0))
            
            # Test learning if requested
            if test_learning and i % 10 == 0:
                learn_start = time.perf_counter()
                predicted = torch.randn(input_dim, device=device)
                actual = test_inputs[i]
                brain.learn_from_prediction_error(actual, predicted)
                if device == 'cuda':
                    torch.cuda.synchronize()
                learning_times.append((time.perf_counter() - learn_start) * 1000)
            
            # Progress update
            if self.verbose and (i + 1) % 100 == 0:
                current_mean = np.mean(cycle_times)
                current_hz = 1000 / current_mean
                print(f"  Cycle {i+1}/{num_cycles}: {current_mean:.2f}ms ({current_hz:.1f} Hz)")
        
        # Calculate statistics
        cycle_times = np.array(cycle_times)
        
        # Memory check
        if device == 'cuda':
            final_gpu_memory = torch.cuda.memory_allocated() / 1024**2
            peak_gpu_memory = torch.cuda.max_memory_allocated() / 1024**2
            gpu_memory_leak = (final_gpu_memory - initial_gpu_memory) > 10  # >10MB growth
        else:
            final_gpu_memory = 0
            peak_gpu_memory = 0
            gpu_memory_leak = False
        
        final_cpu_memory = process.memory_info().rss / 1024**2
        cpu_memory_growth = final_cpu_memory - initial_cpu_memory
        
        # Test CPU-GPU transfer overhead
        transfer_overhead = self._measure_transfer_overhead(brain, device, test_inputs[:10])
        
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
            
            # Component timings
            sensory_injection_time=np.mean(sensory_times) if sensory_times else 0,
            field_evolution_time=np.mean(evolution_times) if evolution_times else 0,
            motor_extraction_time=np.mean(motor_times) if motor_times else 0,
            learning_time=np.mean(learning_times) if learning_times else 0,
            
            # Memory metrics
            gpu_memory_used_mb=final_gpu_memory,
            gpu_memory_peak_mb=peak_gpu_memory,
            cpu_memory_used_mb=cpu_memory_growth,
            memory_leak_detected=gpu_memory_leak,
            
            # Performance indicators
            achieved_hz=1000 / np.mean(cycle_times),
            meets_30hz_requirement=np.percentile(cycle_times, 95) < 33.33,  # 95% of cycles under 33ms
            max_sustainable_hz=1000 / np.percentile(cycle_times, 95),
            latency_jitter_ms=np.std(cycle_times),
            
            # Transfer metrics
            transfer_overhead_ms=transfer_overhead,
            transfer_count_per_cycle=3  # input, output, and internal transfers
        )
        
        self.results.append(metrics)
        
        if self.verbose:
            self._print_metrics(metrics)
        
        return metrics
    
    def _measure_transfer_overhead(
        self,
        brain: PureFieldBrain,
        device: str,
        test_inputs: List[torch.Tensor]
    ) -> float:
        """Measure CPU-GPU transfer overhead"""
        
        if device == 'cpu':
            return 0.0
        
        transfer_times = []
        
        for input_tensor in test_inputs:
            # Test GPU -> CPU transfer (motor output)
            start = time.perf_counter()
            dummy = input_tensor.cpu()
            torch.cuda.synchronize()
            transfer_times.append((time.perf_counter() - start) * 1000)
            
            # Test CPU -> GPU transfer (sensory input)
            cpu_tensor = torch.randn(input_tensor.shape[0])
            start = time.perf_counter()
            gpu_tensor = cpu_tensor.to(device)
            torch.cuda.synchronize()
            transfer_times.append((time.perf_counter() - start) * 1000)
        
        return np.mean(transfer_times)
    
    def _print_metrics(self, metrics: PerformanceMetrics):
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
        
        # Component breakdown
        print(f"\n🔧 Component Breakdown:")
        total_component = (metrics.sensory_injection_time + 
                          metrics.field_evolution_time + 
                          metrics.motor_extraction_time)
        print(f"  Sensory injection: {metrics.sensory_injection_time:.2f}ms ({metrics.sensory_injection_time/total_component*100:.1f}%)")
        print(f"  Field evolution: {metrics.field_evolution_time:.2f}ms ({metrics.field_evolution_time/total_component*100:.1f}%)")
        print(f"  Motor extraction: {metrics.motor_extraction_time:.2f}ms ({metrics.motor_extraction_time/total_component*100:.1f}%)")
        if metrics.learning_time > 0:
            print(f"  Learning update: {metrics.learning_time:.2f}ms")
        
        # Memory usage
        print(f"\n💾 Memory Usage:")
        if metrics.device == 'cuda':
            print(f"  GPU memory: {metrics.gpu_memory_used_mb:.1f}MB (peak: {metrics.gpu_memory_peak_mb:.1f}MB)")
            print(f"  Memory leak: {'⚠️ YES' if metrics.memory_leak_detected else '✅ NO'}")
        print(f"  CPU memory growth: {metrics.cpu_memory_used_mb:.1f}MB")
        
        # Transfer overhead
        if metrics.device == 'cuda':
            print(f"\n🔄 CPU-GPU Transfer:")
            print(f"  Average overhead: {metrics.transfer_overhead_ms:.2f}ms per transfer")
            print(f"  Transfers per cycle: {metrics.transfer_count_per_cycle}")
        
        # Requirements check
        print(f"\n✅ Requirements Check:")
        print(f"  30Hz requirement: {'✅ PASS' if metrics.meets_30hz_requirement else '❌ FAIL'}")
        print(f"  Max sustainable: {metrics.max_sustainable_hz:.1f} Hz")
        
        if not metrics.meets_30hz_requirement:
            deficit = 33.33 - metrics.p95_cycle_time
            print(f"  ⚠️ Need {abs(deficit):.1f}ms improvement to meet 30Hz")
    
    def test_all_configurations(self) -> Dict[str, PerformanceMetrics]:
        """Test all available configurations"""
        
        results = {}
        
        # Test configurations in order of size
        test_configs = ['hardware_constrained', 'tiny', 'small', 'medium']
        
        # Add large configs only if we have GPU
        if torch.cuda.is_available():
            test_configs.append('large')
        
        for config_name in test_configs:
            try:
                metrics = self.test_configuration(config_name)
                results[config_name] = metrics
                
                # Clean up between tests
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"❌ Failed to test {config_name}: {e}")
        
        return results
    
    def generate_report(self, output_file: str = "pure_field_performance_report.json"):
        """Generate comprehensive performance report"""
        
        if not self.results:
            print("No results to report")
            return
        
        # Create report dictionary
        report = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'configurations_tested': len(self.results),
            'results': [m.to_dict() for m in self.results],
            'summary': self._generate_summary()
        }
        
        # Save to file
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n📄 Report saved to {output_file}")
        
        # Generate plots
        self._generate_plots()
    
    def _generate_summary(self) -> Dict:
        """Generate summary statistics"""
        
        # Find best configuration for 30Hz
        best_for_30hz = None
        for metrics in self.results:
            if metrics.meets_30hz_requirement:
                if best_for_30hz is None or metrics.total_params > best_for_30hz.total_params:
                    best_for_30hz = metrics
        
        # Find fastest configuration
        fastest = min(self.results, key=lambda m: m.mean_cycle_time)
        
        return {
            'best_for_30hz': best_for_30hz.config_name if best_for_30hz else 'None',
            'best_30hz_params': best_for_30hz.total_params if best_for_30hz else 0,
            'fastest_config': fastest.config_name,
            'fastest_hz': fastest.achieved_hz,
            'all_configs_meet_30hz': all(m.meets_30hz_requirement for m in self.results)
        }
    
    def _generate_plots(self):
        """Generate performance visualization plots"""
        
        if not self.results:
            return
        
        # Set style
        sns.set_style("whitegrid")
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot 1: Cycle time by configuration
        ax = axes[0, 0]
        configs = [m.config_name for m in self.results]
        mean_times = [m.mean_cycle_time for m in self.results]
        p95_times = [m.p95_cycle_time for m in self.results]
        
        x = np.arange(len(configs))
        width = 0.35
        
        ax.bar(x - width/2, mean_times, width, label='Mean', color='skyblue')
        ax.bar(x + width/2, p95_times, width, label='95th %ile', color='salmon')
        ax.axhline(y=33.33, color='red', linestyle='--', label='30Hz threshold')
        ax.set_xlabel('Configuration')
        ax.set_ylabel('Cycle Time (ms)')
        ax.set_title('Cycle Time by Configuration')
        ax.set_xticks(x)
        ax.set_xticklabels(configs, rotation=45)
        ax.legend()
        
        # Plot 2: Achieved Hz vs Parameters
        ax = axes[0, 1]
        params = [m.total_params / 1e6 for m in self.results]  # Convert to millions
        hz = [m.achieved_hz for m in self.results]
        
        ax.scatter(params, hz, s=100, alpha=0.7)
        ax.axhline(y=30, color='red', linestyle='--', label='30Hz requirement')
        for i, config in enumerate(configs):
            ax.annotate(config, (params[i], hz[i]), fontsize=8)
        ax.set_xlabel('Parameters (millions)')
        ax.set_ylabel('Achieved Hz')
        ax.set_title('Performance vs Model Size')
        ax.legend()
        
        # Plot 3: Component breakdown
        ax = axes[1, 0]
        components = ['Sensory', 'Evolution', 'Motor']
        for i, metrics in enumerate(self.results):
            times = [metrics.sensory_injection_time, 
                    metrics.field_evolution_time,
                    metrics.motor_extraction_time]
            ax.bar(x + i*0.15, times, 0.15, label=metrics.config_name)
        
        ax.set_xlabel('Component')
        ax.set_ylabel('Time (ms)')
        ax.set_title('Component Timing Breakdown')
        ax.set_xticks(x + 0.3)
        ax.set_xticklabels(components)
        ax.legend(fontsize=8)
        
        # Plot 4: Memory usage
        ax = axes[1, 1]
        if any(m.device == 'cuda' for m in self.results):
            gpu_configs = [m for m in self.results if m.device == 'cuda']
            configs = [m.config_name for m in gpu_configs]
            memory = [m.gpu_memory_peak_mb for m in gpu_configs]
            
            ax.bar(configs, memory, color='green', alpha=0.7)
            ax.set_xlabel('Configuration')
            ax.set_ylabel('GPU Memory (MB)')
            ax.set_title('GPU Memory Usage')
            ax.set_xticklabels(configs, rotation=45)
        else:
            ax.text(0.5, 0.5, 'No GPU tests', ha='center', va='center')
        
        plt.tight_layout()
        plt.savefig('pure_field_performance.png', dpi=150)
        print(f"\n📊 Performance plots saved to pure_field_performance.png")
        plt.show()
    
    def identify_bottlenecks(self) -> Dict[str, List[str]]:
        """Identify performance bottlenecks for each configuration"""
        
        bottlenecks = {}
        
        for metrics in self.results:
            config_bottlenecks = []
            
            # Check if meeting 30Hz requirement
            if not metrics.meets_30hz_requirement:
                deficit = metrics.p95_cycle_time - 33.33
                config_bottlenecks.append(f"Not meeting 30Hz requirement (deficit: {abs(deficit):.1f}ms)")
            
            # Check component bottlenecks
            total_time = (metrics.sensory_injection_time + 
                         metrics.field_evolution_time + 
                         metrics.motor_extraction_time)
            
            if metrics.field_evolution_time / total_time > 0.6:
                config_bottlenecks.append(f"Field evolution is bottleneck ({metrics.field_evolution_time/total_time*100:.1f}% of time)")
            
            # Check memory issues
            if metrics.memory_leak_detected:
                config_bottlenecks.append("Memory leak detected")
            
            # Check transfer overhead
            if metrics.device == 'cuda' and metrics.transfer_overhead_ms > 1.0:
                config_bottlenecks.append(f"High CPU-GPU transfer overhead ({metrics.transfer_overhead_ms:.2f}ms)")
            
            # Check jitter
            if metrics.latency_jitter_ms > 5.0:
                config_bottlenecks.append(f"High latency jitter ({metrics.latency_jitter_ms:.2f}ms)")
            
            bottlenecks[metrics.config_name] = config_bottlenecks
        
        return bottlenecks


def main():
    """Run comprehensive performance analysis"""
    
    print("=" * 70)
    print("PureFieldBrain Real-Time Performance Analysis")
    print("Testing for 30+ Hz robot control requirements")
    print("=" * 70)
    
    # Create tester
    tester = PureFieldPerformanceTester(verbose=True)
    
    # Test all configurations
    results = tester.test_all_configurations()
    
    # Generate report
    tester.generate_report()
    
    # Print bottleneck analysis
    print("\n" + "=" * 70)
    print("🔍 BOTTLENECK ANALYSIS")
    print("=" * 70)
    
    bottlenecks = tester.identify_bottlenecks()
    for config, issues in bottlenecks.items():
        print(f"\n{config}:")
        if issues:
            for issue in issues:
                print(f"  ⚠️ {issue}")
        else:
            print(f"  ✅ No bottlenecks identified")
    
    # Print recommendations
    print("\n" + "=" * 70)
    print("💡 RECOMMENDATIONS")
    print("=" * 70)
    
    # Find best configuration
    best = None
    for config, metrics in results.items():
        if metrics.meets_30hz_requirement:
            if best is None or metrics.total_params > best.total_params:
                best = metrics
    
    if best:
        print(f"\n✅ Recommended configuration: {best.config_name}")
        print(f"   - Achieves {best.achieved_hz:.1f} Hz")
        print(f"   - {best.total_params:,} parameters")
        print(f"   - 95th percentile: {best.p95_cycle_time:.2f}ms")
        print(f"   - GPU memory: {best.gpu_memory_peak_mb:.1f}MB")
    else:
        print("\n⚠️ No configuration meets 30Hz requirement!")
        fastest = min(results.values(), key=lambda m: m.mean_cycle_time)
        print(f"   Fastest: {fastest.config_name} at {fastest.achieved_hz:.1f} Hz")
        print(f"   Consider optimization or hardware upgrade")
    
    # Additional optimization suggestions
    print("\n🔧 Optimization Opportunities:")
    
    for config, metrics in results.items():
        if metrics.field_evolution_time > 10:
            print(f"  - {config}: Consider optimizing field evolution (currently {metrics.field_evolution_time:.1f}ms)")
        if metrics.transfer_overhead_ms > 0.5:
            print(f"  - {config}: Minimize CPU-GPU transfers (overhead: {metrics.transfer_overhead_ms:.2f}ms)")
        if metrics.latency_jitter_ms > 3:
            print(f"  - {config}: Reduce jitter for more predictable timing (std: {metrics.latency_jitter_ms:.2f}ms)")


if __name__ == "__main__":
    main()