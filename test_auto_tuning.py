"""
Test Auto-Tuning Energy Traversal System.
Demonstrates the cognitive system learning to optimize its own parameters
to find the goldilocks zone of speed vs quality trade-offs.
"""

import sys
import time
import json
from typing import List, Dict
from statistics import mean

# Add project root to path
sys.path.append('.')

from predictor.auto_tuned_energy_traversal import AutoTunedEnergyTraversal
from simulation.brainstem_sim import GridWorldBrainstem


class AutoTuningExperiment:
    """Experiment to demonstrate auto-tuning finding the goldilocks zone."""
    
    def __init__(self, world_graph, test_contexts: List[List[float]]):
        self.world_graph = world_graph
        self.test_contexts = test_contexts
        
        # Initialize auto-tuned traversal system
        self.auto_tuned_system = AutoTunedEnergyTraversal(
            # Starting parameters (deliberately suboptimal)
            initial_energy=100.0,
            energy_boost_rate=25.0,
            energy_drain_rate=20.0,  # Start with aggressive drain
            interest_threshold=0.7,   # Start with high threshold
            branching_threshold=90.0, # Start with high branching threshold
            minimum_energy=5.0,
            curiosity_momentum=0.2,
            randomness_factor=0.3,
            exploration_rate=0.3,
            # Auto-tuning configuration
            enable_auto_tuning=True,
            tuning_frequency=5,  # Tune every 5 traversals
            goldilocks_target_speed=3.0,    # Target 3x speed improvement
            goldilocks_target_quality=0.95  # Target 95% quality retention
        )
        
        # Tracking
        self.experiment_results = []
        self.parameter_evolution = []
        self.goldilocks_scores = []
    
    def run_experiment(self, num_iterations: int = 100) -> Dict:
        """
        Run auto-tuning experiment to find goldilocks zone.
        
        Args:
            num_iterations: Number of traversals to perform
            
        Returns:
            Comprehensive experiment results
        """
        print("🧠 AUTO-TUNING GOLDILOCKS ZONE EXPERIMENT")
        print("=" * 60)
        print(f"Graph size: {self.world_graph.node_count()} nodes")
        print(f"Test contexts: {len(self.test_contexts)}")
        print(f"Iterations: {num_iterations}")
        print(f"Target: {self.auto_tuned_system.goldilocks_target_speed}x speed, " +
              f"{self.auto_tuned_system.goldilocks_target_quality}x quality")
        print()
        
        # Run experiment iterations
        for iteration in range(num_iterations):
            if iteration % 20 == 0:
                print(f"Iteration {iteration}/{num_iterations}")
            
            # Select test context (cycle through available contexts)
            context = self.test_contexts[iteration % len(self.test_contexts)]
            
            # Add some contextual variation
            iteration_context = {
                'threat_level': 'normal',
                'cognitive_load': 0.3 + (iteration % 10) * 0.05,  # Varies 0.3-0.8
                'time_pressure': 0.2 + (iteration % 5) * 0.1      # Varies 0.2-0.6
            }
            
            # Perform auto-tuned traversal
            start_time = time.time()
            result = self.auto_tuned_system.traverse(
                context, self.world_graph, 
                random_seed=iteration,
                context=iteration_context
            )
            iteration_time = time.time() - start_time
            
            # Track results
            self.experiment_results.append({
                'iteration': iteration,
                'speed_ratio': result.performance_metrics.speed_ratio,
                'quality_ratio': result.performance_metrics.quality_ratio,
                'exploration_efficiency': result.performance_metrics.exploration_efficiency,
                'goldilocks_score': result.goldilocks_score,
                'tuning_triggered': result.tuning_triggered,
                'parameter_adjustments': result.parameter_adjustments,
                'termination_reason': result.base_result.termination_reason,
                'steps_taken': result.base_result.steps_taken,
                'branches_spawned': result.base_result.branches_spawned,
                'iteration_time': iteration_time
            })
            
            # Track parameter evolution
            if iteration % 5 == 0:  # Every 5 iterations
                current_params = self.auto_tuned_system._get_current_parameters()
                self.parameter_evolution.append({
                    'iteration': iteration,
                    'parameters': current_params.copy()
                })
            
            self.goldilocks_scores.append(result.goldilocks_score)
            
            # Check for early goldilocks achievement
            if (result.goldilocks_score > 0.95 and 
                iteration > 50 and 
                self.auto_tuned_system.goldilocks_achieved):
                print(f"🎪 GOLDILOCKS ZONE ACHIEVED at iteration {iteration}!")
                break
        
        # Analyze results
        analysis = self._analyze_experiment_results()
        
        # Print summary
        self._print_experiment_summary(analysis)
        
        return analysis
    
    def _analyze_experiment_results(self) -> Dict:
        """Analyze the complete experiment results."""
        if not self.experiment_results:
            return {"error": "no_results"}
        
        # Performance evolution analysis
        all_speeds = [r['speed_ratio'] for r in self.experiment_results]
        all_qualities = [r['quality_ratio'] for r in self.experiment_results]
        all_scores = [r['goldilocks_score'] for r in self.experiment_results]
        
        # Early vs late performance (first 25% vs last 25%)
        quarter_point = len(self.experiment_results) // 4
        early_results = self.experiment_results[:quarter_point] if quarter_point > 0 else self.experiment_results[:1]
        late_results = self.experiment_results[-quarter_point:] if quarter_point > 0 else self.experiment_results[-1:]
        
        early_avg_speed = mean([r['speed_ratio'] for r in early_results])
        late_avg_speed = mean([r['speed_ratio'] for r in late_results])
        early_avg_quality = mean([r['quality_ratio'] for r in early_results])
        late_avg_quality = mean([r['quality_ratio'] for r in late_results])
        early_avg_score = mean([r['goldilocks_score'] for r in early_results])
        late_avg_score = mean([r['goldilocks_score'] for r in late_results])
        
        # Parameter optimization tracking
        tuning_events = [r for r in self.experiment_results if r['tuning_triggered']]
        
        # Best performance found
        best_result = max(self.experiment_results, key=lambda r: r['goldilocks_score'])
        
        # Convergence analysis
        recent_scores = self.goldilocks_scores[-20:] if len(self.goldilocks_scores) >= 20 else self.goldilocks_scores
        score_variance = self._calculate_variance(recent_scores) if len(recent_scores) > 1 else 1.0
        convergence_achieved = score_variance < 0.01  # Low variance indicates convergence
        
        # Parameter stability analysis
        if len(self.parameter_evolution) >= 2:
            param_stability = self._analyze_parameter_stability()
        else:
            param_stability = {"status": "insufficient_data"}
        
        # Final tuning status
        final_status = self.auto_tuned_system.get_tuning_status()
        goldilocks_analysis = self.auto_tuned_system.get_goldilocks_analysis()
        
        return {
            "experiment_summary": {
                "total_iterations": len(self.experiment_results),
                "tuning_events": len(tuning_events),
                "goldilocks_achieved": self.auto_tuned_system.goldilocks_achieved,
                "convergence_achieved": convergence_achieved
            },
            "performance_evolution": {
                "early_performance": {
                    "avg_speed_ratio": early_avg_speed,
                    "avg_quality_ratio": early_avg_quality,
                    "avg_goldilocks_score": early_avg_score
                },
                "late_performance": {
                    "avg_speed_ratio": late_avg_speed,
                    "avg_quality_ratio": late_avg_quality,
                    "avg_goldilocks_score": late_avg_score
                },
                "improvement": {
                    "speed_improvement": late_avg_speed - early_avg_speed,
                    "quality_improvement": late_avg_quality - early_avg_quality,
                    "score_improvement": late_avg_score - early_avg_score
                }
            },
            "best_performance": {
                "iteration": best_result['iteration'],
                "speed_ratio": best_result['speed_ratio'],
                "quality_ratio": best_result['quality_ratio'],
                "goldilocks_score": best_result['goldilocks_score'],
                "parameters_at_best": self._get_parameters_at_iteration(best_result['iteration'])
            },
            "parameter_optimization": {
                "stability_analysis": param_stability,
                "tuning_frequency": len(tuning_events) / len(self.experiment_results),
                "convergence_variance": score_variance
            },
            "final_status": final_status,
            "goldilocks_analysis": goldilocks_analysis,
            "raw_results": self.experiment_results[-10:]  # Last 10 results for inspection
        }
    
    def _calculate_variance(self, values: List[float]) -> float:
        """Calculate variance of a list of values."""
        if len(values) < 2:
            return 0.0
        
        mean_val = mean(values)
        return sum((x - mean_val) ** 2 for x in values) / len(values)
    
    def _analyze_parameter_stability(self) -> Dict:
        """Analyze how parameters have stabilized over time."""
        if len(self.parameter_evolution) < 2:
            return {"status": "insufficient_data"}
        
        # Look at parameter changes over time
        param_names = list(self.parameter_evolution[0]['parameters'].keys())
        stability_scores = {}
        
        for param_name in param_names:
            values = [pe['parameters'][param_name] for pe in self.parameter_evolution]
            
            # Calculate variance (lower variance = more stable)
            variance = self._calculate_variance(values)
            
            # Calculate trend (is it still changing or has it stabilized?)
            if len(values) >= 3:
                recent_change = abs(values[-1] - values[-2])
                early_change = abs(values[1] - values[0]) if len(values) > 1 else recent_change
                change_ratio = recent_change / max(early_change, 0.001)
            else:
                change_ratio = 1.0
            
            stability_scores[param_name] = {
                "variance": variance,
                "change_ratio": change_ratio,
                "stable": variance < 0.1 and change_ratio < 0.5,
                "final_value": values[-1],
                "initial_value": values[0]
            }
        
        overall_stability = mean([s["variance"] for s in stability_scores.values()])
        stable_params = sum(1 for s in stability_scores.values() if s["stable"])
        
        return {
            "status": "analyzed",
            "overall_stability": overall_stability,
            "stable_parameters": stable_params,
            "total_parameters": len(param_names),
            "stability_percentage": stable_params / len(param_names),
            "parameter_details": stability_scores
        }
    
    def _get_parameters_at_iteration(self, iteration: int) -> Dict:
        """Get parameter values at a specific iteration."""
        # Find closest parameter snapshot
        closest_snapshot = None
        min_distance = float('inf')
        
        for snapshot in self.parameter_evolution:
            distance = abs(snapshot['iteration'] - iteration)
            if distance < min_distance:
                min_distance = distance
                closest_snapshot = snapshot
        
        return closest_snapshot['parameters'] if closest_snapshot else {}
    
    def _print_experiment_summary(self, analysis: Dict):
        """Print comprehensive experiment summary."""
        print("📊 AUTO-TUNING EXPERIMENT RESULTS")
        print("=" * 60)
        
        # Experiment overview
        summary = analysis["experiment_summary"]
        print(f"Total iterations: {summary['total_iterations']}")
        print(f"Tuning events: {summary['tuning_events']}")
        print(f"Goldilocks achieved: {'✅ YES' if summary['goldilocks_achieved'] else '❌ NO'}")
        print(f"Convergence achieved: {'✅ YES' if summary['convergence_achieved'] else '❌ NO'}")
        print()
        
        # Performance evolution
        perf = analysis["performance_evolution"]
        print("📈 PERFORMANCE EVOLUTION:")
        print(f"Early performance:")
        print(f"  Speed ratio: {perf['early_performance']['avg_speed_ratio']:.2f}x")
        print(f"  Quality ratio: {perf['early_performance']['avg_quality_ratio']:.2f}x")
        print(f"  Goldilocks score: {perf['early_performance']['avg_goldilocks_score']:.3f}")
        print()
        print(f"Late performance:")
        print(f"  Speed ratio: {perf['late_performance']['avg_speed_ratio']:.2f}x")
        print(f"  Quality ratio: {perf['late_performance']['avg_quality_ratio']:.2f}x")
        print(f"  Goldilocks score: {perf['late_performance']['avg_goldilocks_score']:.3f}")
        print()
        print(f"Improvement:")
        print(f"  Speed: {perf['improvement']['speed_improvement']:+.2f}x")
        print(f"  Quality: {perf['improvement']['quality_improvement']:+.2f}x")
        print(f"  Score: {perf['improvement']['score_improvement']:+.3f}")
        print()
        
        # Best performance
        best = analysis["best_performance"]
        print(f"🏆 BEST PERFORMANCE (iteration {best['iteration']}):")
        print(f"  Speed ratio: {best['speed_ratio']:.2f}x")
        print(f"  Quality ratio: {best['quality_ratio']:.2f}x")
        print(f"  Goldilocks score: {best['goldilocks_score']:.3f}")
        print()
        
        # Parameter optimization
        param_opt = analysis["parameter_optimization"]
        if param_opt["stability_analysis"]["status"] == "analyzed":
            stability = param_opt["stability_analysis"]
            print(f"🎛️  PARAMETER OPTIMIZATION:")
            print(f"  Tuning frequency: {param_opt['tuning_frequency']:.1%}")
            print(f"  Parameter stability: {stability['stability_percentage']:.1%}")
            print(f"  Stable parameters: {stability['stable_parameters']}/{stability['total_parameters']}")
            print(f"  Convergence variance: {param_opt['convergence_variance']:.4f}")
            print()
        
        # Goldilocks analysis
        goldilocks = analysis["goldilocks_analysis"]
        if goldilocks["status"] != "no_data":
            print(f"🎪 GOLDILOCKS ZONE ANALYSIS:")
            current = goldilocks["current_averages"]
            targets = goldilocks["targets"]
            distances = goldilocks["distances_to_target"]
            
            print(f"  Current vs Target:")
            print(f"    Speed: {current['speed_ratio']:.2f}x (target: {targets['speed_ratio']:.2f}x) - " +
                  f"Distance: {distances['speed_distance']:.2f}")
            print(f"    Quality: {current['quality_ratio']:.2f}x (target: {targets['quality_ratio']:.2f}x) - " +
                  f"Distance: {distances['quality_distance']:.2f}")
            print(f"  Best score achieved: {goldilocks['best_score_achieved']:.3f}")
            print()
        
        # Final recommendations
        print("💡 RECOMMENDATIONS:")
        if summary['goldilocks_achieved']:
            print("  ✅ Goldilocks zone achieved! System has found optimal parameters.")
        elif summary['convergence_achieved']:
            print("  ⚠️  System has converged but not reached goldilocks zone.")
            print("      Consider adjusting target parameters or exploration strategy.")
        else:
            print("  🔄 System is still learning. Continue tuning for better performance.")
            
        if param_opt["stability_analysis"]["status"] == "analyzed":
            stability_pct = param_opt["stability_analysis"]["stability_percentage"]
            if stability_pct > 0.8:
                print("  ✅ Parameters are stable - good convergence.")
            else:
                print("  ⚠️  Parameters still changing - may need more iterations.")


def main():
    """Run the auto-tuning experiment."""
    print("🚀 Loading brain system...")
    
    # Load existing brain system
    brainstem = GridWorldBrainstem(seed=42, use_sockets=False)
    world_graph = brainstem.brain_client.get_world_graph()
    
    if world_graph.node_count() < 100:
        print("❌ Need at least 100 nodes for meaningful evaluation")
        print(f"Current graph has {world_graph.node_count()} nodes")
        return
    
    print(f"✅ Loaded graph with {world_graph.node_count()} nodes")
    
    # Create test contexts
    test_contexts = [
        [0.8, 0.2, 0.1, 0.0, 0.5, 0.3, 0.7, 0.4],  # Wall ahead
        [0.1, 0.9, 0.6, 0.3, 0.8, 0.2, 0.4, 0.7],  # Food nearby
        [0.5, 0.5, 0.8, 0.9, 0.1, 0.6, 0.3, 0.2],  # Danger present
        [0.3, 0.7, 0.4, 0.2, 0.9, 0.1, 0.8, 0.5],  # Exploration
        [0.6, 0.4, 0.2, 0.8, 0.3, 0.9, 0.1, 0.7]   # Mixed scenario
    ]
    
    # Run experiment
    experiment = AutoTuningExperiment(world_graph, test_contexts)
    results = experiment.run_experiment(num_iterations=80)
    
    return results


if __name__ == "__main__":
    results = main()