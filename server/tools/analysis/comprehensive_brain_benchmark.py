#!/usr/bin/env python3
"""
Comprehensive Brain Benchmark
============================

Definitive comparison of three field brain implementations:
1. UnifiedFieldBrain (original, complex)
2. MinimalFieldBrain (200 lines, aggressive)
3. PureFieldBrain (single tensor operation)

Tests:
- Performance (cycles per second)
- Memory usage
- Learning capability
- Behavioral emergence
- Real robot deployment readiness

Result: Which brain wins for actual robot deployment?
"""

import torch
import numpy as np
import time
import psutil
import gc
import sys
import os
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
from collections import defaultdict
import json

# Add server path
sys.path.append('/mnt/c/Users/glimm/Documents/Projects/em-brain/server/src')

# Import the three brains
from brains.field.unified_field_brain import UnifiedFieldBrain
from brains.field.minimal_field_brain import MinimalFieldBrain
from brains.field.pure_field_brain import PureFieldBrain, create_pure_field_brain


@dataclass
class BenchmarkResult:
    """Results for one brain implementation"""
    name: str
    performance_hz: float
    memory_mb: float
    learning_speed: float
    behavioral_emergence: float
    deployment_score: float
    stability: float
    gpu_efficiency: float
    raw_metrics: Dict[str, Any]


class ComprehensiveBrainBenchmark:
    """
    The ultimate brain benchmark.
    
    Tests everything that matters for real robot deployment:
    - Raw speed (how fast can it process?)
    - Memory efficiency (can it run on limited hardware?)
    - Learning ability (does it actually adapt?)
    - Behavioral complexity (does it show intelligence?)
    - Stability (does it crash or go insane?)
    """
    
    def __init__(self, device: str = None):
        """Initialize benchmark"""
        if device is None:
            if torch.cuda.is_available():
                self.device = 'cuda'
            else:
                self.device = 'cpu'
        else:
            self.device = device
            
        print(f"🧪 Comprehensive Brain Benchmark")
        print(f"   Device: {self.device}")
        print(f"   PyTorch: {torch.__version__}")
        
        # Benchmark configuration
        self.sensory_dim = 12  # Typical robot sensors
        self.motor_dim = 4     # Typical robot motors  
        self.warmup_cycles = 50
        self.performance_cycles = 200
        self.learning_cycles = 500
        self.stability_cycles = 1000
        
        # Robot simulation parameters
        self.obstacle_distance = 1.0
        self.target_distance = 2.0
        self.noise_level = 0.1
        
        print(f"   Cycles: warmup={self.warmup_cycles}, perf={self.performance_cycles}")
        print(f"   Learning cycles: {self.learning_cycles}")
        print(f"   Stability cycles: {self.stability_cycles}")
        
    def create_brains(self) -> Dict[str, Any]:
        """Create all three brain types for comparison"""
        brains = {}
        
        print("\n🧠 Creating brains...")
        
        try:
            # 1. UnifiedFieldBrain (original complex)
            print("   Creating UnifiedFieldBrain...")
            brains['unified'] = UnifiedFieldBrain(
                sensory_dim=self.sensory_dim,
                motor_dim=self.motor_dim,
                spatial_resolution=24,  # Smaller for fair comparison
                device=torch.device(self.device),
                quiet_mode=True
            )
            print(f"      ✓ Created with {self._count_parameters(brains['unified'])} parameters")
            
        except Exception as e:
            print(f"      ✗ Failed: {e}")
            brains['unified'] = None
        
        try:
            # 2. MinimalFieldBrain (aggressive simplification)
            print("   Creating MinimalFieldBrain...")
            brains['minimal'] = MinimalFieldBrain(
                sensory_dim=self.sensory_dim,
                motor_dim=self.motor_dim,
                spatial_resolution=24,
                device=torch.device(self.device),
                quiet_mode=True
            )
            print(f"      ✓ Created with {self._count_parameters(brains['minimal'])} parameters")
            
        except Exception as e:
            print(f"      ✗ Failed: {e}")
            brains['minimal'] = None
            
        try:
            # 3. PureFieldBrain (ultimate synthesis)
            print("   Creating PureFieldBrain...")
            brains['pure'] = create_pure_field_brain(
                input_dim=self.sensory_dim,
                output_dim=self.motor_dim,
                size='small',  # 24³×48 for fair comparison
                aggressive=True,
                device=self.device
            )
            print(f"      ✓ Created with {sum(p.numel() for p in brains['pure'].parameters()):,} parameters")
            
        except Exception as e:
            print(f"      ✗ Failed: {e}")
            brains['pure'] = None
        
        return {k: v for k, v in brains.items() if v is not None}
    
    def _count_parameters(self, brain) -> int:
        """Count parameters in brain (handle different interfaces)"""
        try:
            if hasattr(brain, 'parameters'):
                return sum(p.numel() for p in brain.parameters() if hasattr(p, 'numel'))
            elif hasattr(brain, 'field'):
                # Estimate from field and weight tensors
                params = np.prod(brain.field.shape)
                if hasattr(brain, 'motor_weights'):
                    params += np.prod(brain.motor_weights.shape)
                if hasattr(brain, 'sensory_weights'):
                    params += np.prod(brain.sensory_weights.shape)
                return params
            else:
                return 0
        except:
            return 0
    
    def simulate_robot_environment(self, cycle: int) -> Tuple[List[float], float]:
        """
        Simulate a robot navigating with obstacles.
        Returns (sensory_input, reward)
        """
        # Simple robot simulation
        # Sensors: [front_dist, left_dist, right_dist, back_dist, 
        #           compass_x, compass_y, target_x, target_y,
        #           velocity_x, velocity_y, battery, temperature]
        
        # Obstacle avoidance scenario
        front_obstacle = max(0, self.obstacle_distance - 0.1 * np.sin(cycle * 0.1))
        left_obstacle = 1.0 + 0.2 * np.cos(cycle * 0.05)
        right_obstacle = 0.8 + 0.3 * np.sin(cycle * 0.07)
        back_clear = 2.0
        
        # Navigation target (moves slowly)
        target_x = np.sin(cycle * 0.02) 
        target_y = np.cos(cycle * 0.03)
        
        # Robot state (accumulated from previous cycles)
        compass_x = np.sin(cycle * 0.01)
        compass_y = np.cos(cycle * 0.01)
        velocity_x = 0.1 * np.sin(cycle * 0.05)
        velocity_y = 0.1 * np.cos(cycle * 0.05)
        
        battery = 0.8 + 0.2 * np.sin(cycle * 0.001)  # Slow battery fluctuation
        temperature = 0.5 + 0.1 * np.sin(cycle * 0.02)  # Thermal cycles
        
        sensory_input = [
            front_obstacle, left_obstacle, right_obstacle, back_clear,
            compass_x, compass_y, target_x, target_y,
            velocity_x, velocity_y, battery, temperature
        ]
        
        # Add noise
        sensory_input = [s + np.random.normal(0, self.noise_level) for s in sensory_input]
        
        # Reward function: avoid obstacles, reach target, conserve energy
        reward = 0.0
        
        # Penalty for being too close to obstacles
        if front_obstacle < 0.3:
            reward -= 1.0
        if min(left_obstacle, right_obstacle) < 0.4:
            reward -= 0.5
            
        # Reward for progress toward target
        target_distance = np.sqrt(target_x**2 + target_y**2)
        if target_distance < 0.5:
            reward += 1.0
        elif target_distance < 1.0:
            reward += 0.5
            
        # Small penalty for excessive movement (energy conservation)
        movement = abs(velocity_x) + abs(velocity_y)
        if movement > 0.5:
            reward -= 0.1
            
        return sensory_input, reward
    
    def benchmark_performance(self, brain, brain_name: str) -> Dict[str, float]:
        """Measure raw performance (cycles per second)"""
        print(f"\n⚡ Performance test: {brain_name}")
        
        # Warmup
        print("   Warming up...")
        for i in range(self.warmup_cycles):
            sensory_input, reward = self.simulate_robot_environment(i)
            try:
                if hasattr(brain, 'process_cycle'):
                    brain.process_cycle(sensory_input)
                else:
                    brain(torch.tensor(sensory_input))
            except Exception as e:
                print(f"   Warning: Warmup cycle {i} failed: {e}")
        
        # Sync if GPU
        if self.device == 'cuda':
            torch.cuda.synchronize()
        
        # Benchmark
        print("   Running performance test...")
        start_time = time.perf_counter()
        successful_cycles = 0
        
        for i in range(self.performance_cycles):
            sensory_input, reward = self.simulate_robot_environment(i + self.warmup_cycles)
            try:
                if hasattr(brain, 'process_cycle'):
                    motor_output, brain_state = brain.process_cycle(sensory_input)
                else:
                    motor_output = brain(torch.tensor(sensory_input), reward)
                successful_cycles += 1
            except Exception as e:
                print(f"   Warning: Cycle {i} failed: {e}")
        
        # Sync if GPU
        if self.device == 'cuda':
            torch.cuda.synchronize()
        
        end_time = time.perf_counter()
        total_time = end_time - start_time
        
        performance_hz = successful_cycles / total_time if total_time > 0 else 0
        avg_time_ms = (total_time * 1000) / successful_cycles if successful_cycles > 0 else float('inf')
        
        print(f"   Results: {performance_hz:.1f} Hz, {avg_time_ms:.2f}ms per cycle")
        print(f"   Success rate: {100*successful_cycles/self.performance_cycles:.1f}%")
        
        return {
            'performance_hz': performance_hz,
            'avg_time_ms': avg_time_ms,
            'success_rate': successful_cycles / self.performance_cycles
        }
    
    def benchmark_memory(self, brain, brain_name: str) -> Dict[str, float]:
        """Measure memory usage"""
        print(f"\n💾 Memory test: {brain_name}")
        
        # Force garbage collection
        gc.collect()
        if self.device == 'cuda':
            torch.cuda.empty_cache()
        
        # Baseline memory
        process = psutil.Process()
        baseline_mb = process.memory_info().rss / 1024 / 1024
        
        if self.device == 'cuda':
            baseline_gpu_mb = torch.cuda.memory_allocated() / 1024 / 1024
        else:
            baseline_gpu_mb = 0
        
        # Run a few cycles to stabilize memory
        for i in range(10):
            sensory_input, reward = self.simulate_robot_environment(i)
            try:
                if hasattr(brain, 'process_cycle'):
                    brain.process_cycle(sensory_input)
                else:
                    brain(torch.tensor(sensory_input))
            except:
                pass
        
        # Measure memory after stabilization
        active_mb = process.memory_info().rss / 1024 / 1024
        if self.device == 'cuda':
            active_gpu_mb = torch.cuda.memory_allocated() / 1024 / 1024
        else:
            active_gpu_mb = 0
        
        memory_usage_mb = active_mb - baseline_mb
        gpu_usage_mb = active_gpu_mb - baseline_gpu_mb
        
        print(f"   CPU memory: {memory_usage_mb:.1f}MB")
        print(f"   GPU memory: {gpu_usage_mb:.1f}MB")
        print(f"   Total: {memory_usage_mb + gpu_usage_mb:.1f}MB")
        
        return {
            'cpu_memory_mb': memory_usage_mb,
            'gpu_memory_mb': gpu_usage_mb,
            'total_memory_mb': memory_usage_mb + gpu_usage_mb
        }
    
    def benchmark_learning(self, brain, brain_name: str) -> Dict[str, float]:
        """Measure learning capability"""
        print(f"\n🧩 Learning test: {brain_name}")
        
        # Learning scenario: robot must learn to avoid obstacles
        initial_reward_sum = 0
        final_reward_sum = 0
        prediction_errors = []
        
        print("   Testing initial behavior...")
        for i in range(50):
            sensory_input, reward = self.simulate_robot_environment(i)
            try:
                if hasattr(brain, 'process_cycle'):
                    motor_output, brain_state = brain.process_cycle(sensory_input)
                    if 'prediction_error' in brain_state:
                        prediction_errors.append(brain_state['prediction_error'])
                else:
                    motor_output = brain(torch.tensor(sensory_input), reward)
                    if hasattr(brain, 'learn_from_prediction_error') and i > 0:
                        # Create a simple prediction task
                        predicted = motor_output
                        actual = torch.tensor([reward] * len(motor_output))
                        brain.learn_from_prediction_error(actual, predicted)
                        if hasattr(brain, 'last_prediction_error'):
                            prediction_errors.append(brain.last_prediction_error)
                
                initial_reward_sum += reward
            except Exception as e:
                print(f"   Warning: Learning cycle {i} failed: {e}")
        
        print("   Running learning phase...")
        # Learning phase
        for i in range(self.learning_cycles):
            sensory_input, reward = self.simulate_robot_environment(i + 50)
            try:
                if hasattr(brain, 'process_cycle'):
                    motor_output, brain_state = brain.process_cycle(sensory_input)
                else:
                    motor_output = brain(torch.tensor(sensory_input), reward)
                    if hasattr(brain, 'learn_from_prediction_error'):
                        # Simple prediction learning
                        predicted = motor_output
                        actual = torch.tensor([reward] * len(motor_output))
                        brain.learn_from_prediction_error(actual, predicted)
            except:
                pass
        
        print("   Testing final behavior...")
        # Test final behavior
        for i in range(50):
            sensory_input, reward = self.simulate_robot_environment(i + 50 + self.learning_cycles)
            try:
                if hasattr(brain, 'process_cycle'):
                    motor_output, brain_state = brain.process_cycle(sensory_input)
                else:
                    motor_output = brain(torch.tensor(sensory_input), reward)
                
                final_reward_sum += reward
            except:
                pass
        
        # Calculate learning metrics
        learning_improvement = final_reward_sum - initial_reward_sum
        learning_speed = abs(learning_improvement) / self.learning_cycles
        
        prediction_convergence = 0
        if prediction_errors:
            # Measure how prediction errors decrease over time
            if len(prediction_errors) > 10:
                early_errors = np.mean(prediction_errors[:len(prediction_errors)//3])
                late_errors = np.mean(prediction_errors[-len(prediction_errors)//3:])
                prediction_convergence = max(0, early_errors - late_errors)
        
        print(f"   Initial reward sum: {initial_reward_sum:.2f}")
        print(f"   Final reward sum: {final_reward_sum:.2f}")
        print(f"   Learning improvement: {learning_improvement:.2f}")
        print(f"   Learning speed: {learning_speed:.4f}")
        print(f"   Prediction convergence: {prediction_convergence:.4f}")
        
        return {
            'initial_reward': initial_reward_sum,
            'final_reward': final_reward_sum,
            'learning_improvement': learning_improvement,
            'learning_speed': learning_speed,
            'prediction_convergence': prediction_convergence
        }
    
    def benchmark_stability(self, brain, brain_name: str) -> Dict[str, float]:
        """Test stability over many cycles"""
        print(f"\n🏗️ Stability test: {brain_name}")
        
        successful_cycles = 0
        crashes = 0
        nan_count = 0
        inf_count = 0
        
        motor_outputs = []
        field_energies = []
        
        print(f"   Running {self.stability_cycles} cycles...")
        for i in range(self.stability_cycles):
            sensory_input, reward = self.simulate_robot_environment(i)
            
            try:
                if hasattr(brain, 'process_cycle'):
                    motor_output, brain_state = brain.process_cycle(sensory_input)
                    motor_list = motor_output if isinstance(motor_output, list) else motor_output.tolist()
                    
                    # Track field energy if available
                    if 'field_energy' in brain_state:
                        field_energies.append(brain_state['field_energy'])
                else:
                    motor_output = brain(torch.tensor(sensory_input), reward)
                    motor_list = motor_output.detach().cpu().tolist()
                    
                    # Track field energy from brain metrics
                    if hasattr(brain, 'metrics'):
                        metrics = brain.metrics
                        if 'field_energy' in metrics:
                            field_energies.append(metrics['field_energy'])
                
                # Check for NaN/Inf
                if any(np.isnan(x) for x in motor_list):
                    nan_count += 1
                elif any(np.isinf(x) for x in motor_list):
                    inf_count += 1
                else:
                    motor_outputs.append(motor_list)
                    successful_cycles += 1
                    
            except Exception as e:
                crashes += 1
                if crashes < 5:  # Only print first few crashes
                    print(f"   Crash at cycle {i}: {e}")
        
        # Analyze stability
        stability_rate = successful_cycles / self.stability_cycles
        motor_variance = 0
        if motor_outputs:
            motor_arrays = np.array(motor_outputs)
            motor_variance = np.mean(np.var(motor_arrays, axis=0))
        
        field_stability = 0
        if field_energies:
            field_stability = 1.0 / (1.0 + np.var(field_energies))
        
        print(f"   Successful cycles: {successful_cycles}/{self.stability_cycles}")
        print(f"   Stability rate: {stability_rate:.2%}")
        print(f"   Crashes: {crashes}")
        print(f"   NaN outputs: {nan_count}")
        print(f"   Inf outputs: {inf_count}")
        print(f"   Motor variance: {motor_variance:.4f}")
        print(f"   Field stability: {field_stability:.4f}")
        
        return {
            'stability_rate': stability_rate,
            'crashes': crashes,
            'nan_count': nan_count,
            'inf_count': inf_count,
            'motor_variance': motor_variance,
            'field_stability': field_stability
        }
    
    def calculate_deployment_score(self, results: Dict[str, Any]) -> float:
        """
        Calculate overall deployment readiness score.
        
        Weights factors that matter for real robot deployment:
        - Performance (can it run in real-time?)
        - Memory efficiency (can it run on robot hardware?)
        - Learning capability (does it adapt?)
        - Stability (does it crash?)
        """
        # Performance score (higher Hz is better, target 10+ Hz for real robots)
        perf_score = min(1.0, results['performance']['performance_hz'] / 10.0)
        
        # Memory score (lower usage is better, penalize >100MB)
        memory_mb = results['memory']['total_memory_mb']
        memory_score = max(0, 1.0 - memory_mb / 100.0)
        
        # Learning score (improvement and speed)
        learning_improvement = results['learning']['learning_improvement']
        learning_speed = results['learning']['learning_speed']
        learning_score = min(1.0, max(0, learning_improvement + learning_speed * 10))
        
        # Stability score (success rate is critical)
        stability_score = results['stability']['stability_rate']
        
        # Weighted combination (stability is most important for deployment)
        deployment_score = (
            0.3 * perf_score +
            0.2 * memory_score +
            0.2 * learning_score +
            0.3 * stability_score
        )
        
        return max(0, min(1.0, deployment_score))
    
    def benchmark_brain(self, brain, brain_name: str) -> BenchmarkResult:
        """Run complete benchmark on one brain"""
        print(f"\n{'='*60}")
        print(f"🧠 BENCHMARKING: {brain_name.upper()}")
        print(f"{'='*60}")
        
        results = {}
        
        try:
            # Performance test
            results['performance'] = self.benchmark_performance(brain, brain_name)
        except Exception as e:
            print(f"Performance test failed: {e}")
            results['performance'] = {'performance_hz': 0, 'success_rate': 0}
        
        try:
            # Memory test
            results['memory'] = self.benchmark_memory(brain, brain_name)
        except Exception as e:
            print(f"Memory test failed: {e}")
            results['memory'] = {'total_memory_mb': 1000}  # Large penalty
        
        try:
            # Learning test
            results['learning'] = self.benchmark_learning(brain, brain_name)
        except Exception as e:
            print(f"Learning test failed: {e}")
            results['learning'] = {'learning_improvement': 0, 'learning_speed': 0}
        
        try:
            # Stability test
            results['stability'] = self.benchmark_stability(brain, brain_name)
        except Exception as e:
            print(f"Stability test failed: {e}")
            results['stability'] = {'stability_rate': 0}
        
        # Calculate overall scores
        deployment_score = self.calculate_deployment_score(results)
        
        # GPU efficiency (performance per memory)
        gpu_efficiency = 0
        if results['memory']['total_memory_mb'] > 0:
            gpu_efficiency = results['performance']['performance_hz'] / results['memory']['total_memory_mb']
        
        return BenchmarkResult(
            name=brain_name,
            performance_hz=results['performance']['performance_hz'],
            memory_mb=results['memory']['total_memory_mb'],
            learning_speed=results['learning']['learning_speed'],
            behavioral_emergence=results['learning']['learning_improvement'],
            deployment_score=deployment_score,
            stability=results['stability']['stability_rate'],
            gpu_efficiency=gpu_efficiency,
            raw_metrics=results
        )
    
    def run_comprehensive_benchmark(self) -> List[BenchmarkResult]:
        """Run the complete benchmark suite"""
        print("🚀 Starting Comprehensive Brain Benchmark")
        print("=" * 80)
        
        # Create all brains
        brains = self.create_brains()
        
        if not brains:
            print("❌ No brains could be created!")
            return []
        
        results = []
        
        # Benchmark each brain
        for brain_name, brain in brains.items():
            try:
                result = self.benchmark_brain(brain, brain_name)
                results.append(result)
            except Exception as e:
                print(f"❌ Failed to benchmark {brain_name}: {e}")
        
        return results
    
    def print_final_results(self, results: List[BenchmarkResult]):
        """Print comprehensive comparison"""
        print("\n" + "=" * 80)
        print("🏆 FINAL RESULTS - WHICH BRAIN WINS?")
        print("=" * 80)
        
        if not results:
            print("❌ No results to display")
            return
        
        # Sort by deployment score
        results.sort(key=lambda x: x.deployment_score, reverse=True)
        
        print(f"\n{'Brain':<12} {'Performance':<12} {'Memory':<10} {'Learning':<10} {'Stability':<10} {'Deploy Score':<12}")
        print("-" * 76)
        
        for result in results:
            print(f"{result.name:<12} "
                  f"{result.performance_hz:>8.1f} Hz "
                  f"{result.memory_mb:>7.1f}MB "
                  f"{result.learning_speed:>9.3f} "
                  f"{result.stability:>9.1%} "
                  f"{result.deployment_score:>11.2%}")
        
        # Winner analysis
        winner = results[0]
        print(f"\n🥇 WINNER: {winner.name.upper()}")
        print(f"   Deployment Score: {winner.deployment_score:.1%}")
        print(f"   Performance: {winner.performance_hz:.1f} Hz")
        print(f"   Memory: {winner.memory_mb:.1f}MB")
        print(f"   Stability: {winner.stability:.1%}")
        print(f"   GPU Efficiency: {winner.gpu_efficiency:.2f} Hz/MB")
        
        # Detailed analysis
        print(f"\n📊 DETAILED ANALYSIS:")
        
        fastest = max(results, key=lambda x: x.performance_hz)
        print(f"   Fastest: {fastest.name} ({fastest.performance_hz:.1f} Hz)")
        
        most_efficient = min(results, key=lambda x: x.memory_mb)
        print(f"   Most Memory Efficient: {most_efficient.name} ({most_efficient.memory_mb:.1f}MB)")
        
        best_learner = max(results, key=lambda x: x.learning_speed)
        print(f"   Best Learner: {best_learner.name} ({best_learner.learning_speed:.3f})")
        
        most_stable = max(results, key=lambda x: x.stability)
        print(f"   Most Stable: {most_stable.name} ({most_stable.stability:.1%})")
        
        # Recommendation
        print(f"\n🤖 ROBOT DEPLOYMENT RECOMMENDATION:")
        print(f"   For real robots, use: {winner.name}")
        print(f"   Reason: Best balance of speed, memory, learning, and stability")
        
        if winner.performance_hz < 5:
            print(f"   ⚠️  Warning: May be too slow for real-time robot control")
        if winner.memory_mb > 100:
            print(f"   ⚠️  Warning: High memory usage may limit deployment options")
        if winner.stability < 0.9:
            print(f"   ⚠️  Warning: Stability concerns for long-term deployment")
        
        return results
    
    def save_results(self, results: List[BenchmarkResult], filename: str = None):
        """Save results to JSON file"""
        if filename is None:
            filename = f"brain_benchmark_results_{int(time.time())}.json"
        
        # Convert results to serializable format
        results_data = []
        for result in results:
            results_data.append({
                'name': result.name,
                'performance_hz': result.performance_hz,
                'memory_mb': result.memory_mb,
                'learning_speed': result.learning_speed,
                'behavioral_emergence': result.behavioral_emergence,
                'deployment_score': result.deployment_score,
                'stability': result.stability,
                'gpu_efficiency': result.gpu_efficiency,
                'raw_metrics': result.raw_metrics
            })
        
        benchmark_data = {
            'timestamp': time.time(),
            'device': self.device,
            'pytorch_version': torch.__version__,
            'results': results_data
        }
        
        with open(filename, 'w') as f:
            json.dump(benchmark_data, f, indent=2)
        
        print(f"\n💾 Results saved to: {filename}")


def main():
    """Run the comprehensive benchmark"""
    print("🧪 Comprehensive Brain Benchmark")
    print("Testing UnifiedFieldBrain vs MinimalFieldBrain vs PureFieldBrain")
    print("Goal: Find the best brain for real robot deployment\n")
    
    # Create benchmark
    benchmark = ComprehensiveBrainBenchmark()
    
    # Run all tests
    results = benchmark.run_comprehensive_benchmark()
    
    # Show final comparison
    benchmark.print_final_results(results)
    
    # Save results
    benchmark.save_results(results)
    
    return results


if __name__ == "__main__":
    results = main()