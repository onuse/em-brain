"""
Evaluation framework for comparing energy-based vs depth-limited traversal.
Tests performance, biological realism, and search effectiveness.
"""

import sys
import time
import random
from typing import List, Dict, Any, Tuple
from statistics import mean, median
import numpy as np

# Add project root to path
sys.path.append('.')

from predictor.single_traversal import SingleTraversal
from predictor.energy_based_traversal import EnergyBasedTraversal
from simulation.brainstem_sim import GridWorldBrainstem
from core.world_graph import WorldGraph


class TraversalEvaluator:
    """Compare energy-based vs depth-limited traversal approaches."""
    
    def __init__(self, world_graph: WorldGraph, test_contexts: List[List[float]]):
        self.world_graph = world_graph
        self.test_contexts = test_contexts
        
        # Initialize both traversal types
        self.depth_traversal = SingleTraversal(
            max_depth=5, 
            randomness_factor=0.3,
            exploration_rate=0.3
        )
        
        self.energy_traversal = EnergyBasedTraversal(
            initial_energy=100.0,
            time_budget=0.1,
            interest_threshold=0.6,
            energy_boost_rate=25.0,
            energy_drain_rate=15.0,
            branching_threshold=80.0,
            minimum_energy=5.0,
            curiosity_momentum=0.2,
            randomness_factor=0.3,
            exploration_rate=0.3
        )
    
    def run_comprehensive_evaluation(self, num_trials: int = 50) -> Dict[str, Any]:
        """Run comprehensive evaluation comparing both approaches."""
        print("🧠 Energy-Based Traversal Evaluation")
        print("=" * 50)
        print(f"Graph size: {self.world_graph.node_count()} nodes")
        print(f"Test contexts: {len(self.test_contexts)}")
        print(f"Trials per context: {num_trials}")
        print()
        
        results = {
            "depth_limited": [],
            "energy_based": []
        }
        
        # Run trials for each test context
        for i, context in enumerate(self.test_contexts):
            print(f"Testing context {i+1}/{len(self.test_contexts)}")
            
            # Test depth-limited traversal
            depth_results = []
            for trial in range(num_trials):
                start_time = time.time()
                result = self.depth_traversal.traverse(context, self.world_graph, random_seed=trial)
                duration = time.time() - start_time
                
                depth_results.append({
                    "duration": duration,
                    "depth_reached": result.depth_reached,
                    "terminal_strength": result.terminal_strength,
                    "prediction_quality": self._evaluate_prediction_quality(result.prediction, context),
                    "path_length": len(result.path)
                })
            
            # Test energy-based traversal  
            energy_results = []
            for trial in range(num_trials):
                start_time = time.time()
                result = self.energy_traversal.traverse(context, self.world_graph, random_seed=trial)
                duration = time.time() - start_time
                
                energy_results.append({
                    "duration": duration,
                    "steps_taken": result.steps_taken,
                    "final_energy": result.final_energy,
                    "branches_spawned": result.branches_spawned,
                    "terminal_strength": result.terminal_node.strength if result.terminal_node else 0.0,
                    "prediction_quality": self._evaluate_prediction_quality(result.prediction, context),
                    "path_length": len(result.path),
                    "termination_reason": result.termination_reason,
                    "exploration_quality": self._evaluate_exploration_quality(result.exploration_results)
                })
            
            results["depth_limited"].extend(depth_results)
            results["energy_based"].extend(energy_results)
        
        # Analyze results
        analysis = self._analyze_results(results)
        
        # Print summary
        self._print_evaluation_summary(analysis)
        
        return analysis
    
    def _evaluate_prediction_quality(self, prediction, context: List[float]) -> float:
        """Evaluate the quality of a prediction (0.0 to 1.0)."""
        if not prediction:
            return 0.0
        
        quality_factors = []
        
        # Factor 1: Confidence level
        confidence = getattr(prediction, 'confidence_level', 0.5)
        quality_factors.append(confidence * 0.4)
        
        # Factor 2: Thinking depth (more thinking = potentially better)
        thinking_depth = getattr(prediction, 'thinking_depth', 1)
        depth_factor = min(1.0, thinking_depth / 10.0)
        quality_factors.append(depth_factor * 0.3)
        
        # Factor 3: Motor action coherence (non-zero actions are better)
        motor_action = prediction.motor_action
        action_magnitude = abs(motor_action.get('forward_motor', 0.0)) + \
                          abs(motor_action.get('turn_motor', 0.0)) + \
                          abs(motor_action.get('brake_motor', 0.0))
        action_factor = min(1.0, action_magnitude)
        quality_factors.append(action_factor * 0.3)
        
        return sum(quality_factors)
    
    def _evaluate_exploration_quality(self, exploration_results) -> float:
        """Evaluate the quality of exploration process."""
        if not exploration_results:
            return 0.0
        
        # Calculate average quality of explored nodes
        avg_quality = mean([r.quality for r in exploration_results])
        
        # Bonus for finding high-quality nodes
        high_quality_count = sum(1 for r in exploration_results if r.quality > 0.8)
        high_quality_bonus = min(0.3, high_quality_count * 0.1)
        
        return avg_quality + high_quality_bonus
    
    def _analyze_results(self, results: Dict[str, List[Dict]]) -> Dict[str, Any]:
        """Analyze and compare results from both traversal methods."""
        analysis = {}
        
        for method_name, method_results in results.items():
            if not method_results:
                continue
                
            # Basic statistics
            durations = [r["duration"] for r in method_results]
            terminal_strengths = [r["terminal_strength"] for r in method_results]
            prediction_qualities = [r["prediction_quality"] for r in method_results]
            path_lengths = [r["path_length"] for r in method_results]
            
            stats = {
                "count": len(method_results),
                "avg_duration": mean(durations),
                "median_duration": median(durations),
                "avg_terminal_strength": mean(terminal_strengths),
                "avg_prediction_quality": mean(prediction_qualities),
                "avg_path_length": mean(path_lengths),
                "searches_per_second": 1.0 / mean(durations) if durations else 0.0
            }
            
            # Method-specific statistics
            if method_name == "depth_limited":
                depths = [r["depth_reached"] for r in method_results]
                stats.update({
                    "avg_depth_reached": mean(depths),
                    "max_depth_reached": max(depths),
                    "depth_completion_rate": mean([1.0 if d >= 5 else d/5.0 for d in depths])
                })
            
            elif method_name == "energy_based":
                steps = [r["steps_taken"] for r in method_results]
                energies = [r["final_energy"] for r in method_results]
                branches = [r["branches_spawned"] for r in method_results]
                exploration_qualities = [r["exploration_quality"] for r in method_results]
                
                # Termination reasons
                termination_counts = {}
                for r in method_results:
                    reason = r["termination_reason"]
                    termination_counts[reason] = termination_counts.get(reason, 0) + 1
                
                stats.update({
                    "avg_steps_taken": mean(steps),
                    "avg_final_energy": mean(energies),
                    "avg_branches_spawned": mean(branches),
                    "avg_exploration_quality": mean(exploration_qualities),
                    "termination_reasons": termination_counts,
                    "energy_depletion_rate": termination_counts.get("energy_depleted", 0) / len(method_results)
                })
            
            analysis[method_name] = stats
        
        # Comparative analysis
        if "depth_limited" in analysis and "energy_based" in analysis:
            depth_stats = analysis["depth_limited"]
            energy_stats = analysis["energy_based"]
            
            analysis["comparison"] = {
                "speed_ratio": energy_stats["searches_per_second"] / depth_stats["searches_per_second"],
                "quality_ratio": energy_stats["avg_prediction_quality"] / depth_stats["avg_prediction_quality"],
                "efficiency_ratio": energy_stats["avg_terminal_strength"] / depth_stats["avg_terminal_strength"],
                "exploration_ratio": energy_stats["avg_steps_taken"] / depth_stats["avg_depth_reached"]
            }
        
        return analysis
    
    def _print_evaluation_summary(self, analysis: Dict[str, Any]):
        """Print a comprehensive summary of the evaluation."""
        print("📊 EVALUATION RESULTS")
        print("=" * 50)
        
        # Print individual method results
        for method_name, stats in analysis.items():
            if method_name == "comparison":
                continue
                
            print(f"\n{method_name.replace('_', ' ').title()} Results:")
            print(f"  Average duration: {stats['avg_duration']:.6f}s")
            print(f"  Searches per second: {stats['searches_per_second']:.1f}")
            print(f"  Average prediction quality: {stats['avg_prediction_quality']:.3f}")
            print(f"  Average terminal strength: {stats['avg_terminal_strength']:.1f}")
            print(f"  Average path length: {stats['avg_path_length']:.1f}")
            
            if method_name == "depth_limited":
                print(f"  Average depth reached: {stats['avg_depth_reached']:.1f}")
                print(f"  Depth completion rate: {stats['depth_completion_rate']:.1%}")
            
            elif method_name == "energy_based":
                print(f"  Average steps taken: {stats['avg_steps_taken']:.1f}")
                print(f"  Average final energy: {stats['avg_final_energy']:.1f}")
                print(f"  Average branches spawned: {stats['avg_branches_spawned']:.1f}")
                print(f"  Average exploration quality: {stats['avg_exploration_quality']:.3f}")
                print(f"  Energy depletion rate: {stats['energy_depletion_rate']:.1%}")
                
                print(f"  Termination reasons:")
                for reason, count in stats['termination_reasons'].items():
                    percentage = count / stats['count'] * 100
                    print(f"    {reason}: {count} ({percentage:.1f}%)")
        
        # Print comparison
        if "comparison" in analysis:
            comp = analysis["comparison"]
            print(f"\n🔍 COMPARATIVE ANALYSIS:")
            print(f"  Speed ratio (energy/depth): {comp['speed_ratio']:.2f}x")
            print(f"  Quality ratio (energy/depth): {comp['quality_ratio']:.2f}x")
            print(f"  Efficiency ratio (energy/depth): {comp['efficiency_ratio']:.2f}x")
            print(f"  Exploration ratio (energy/depth): {comp['exploration_ratio']:.2f}x")
            
            print(f"\n🧠 BIOLOGICAL REALISM ASSESSMENT:")
            if comp['speed_ratio'] > 0.8:
                print("  ✅ Energy-based search maintains good performance")
            else:
                print("  ⚠️ Energy-based search is significantly slower")
            
            if comp['quality_ratio'] > 1.0:
                print("  ✅ Energy-based search produces better predictions")
            else:
                print("  ⚠️ Energy-based search produces lower quality predictions")
            
            if analysis["energy_based"]["energy_depletion_rate"] < 0.5:
                print("  ✅ Energy system provides natural stopping points")
            else:
                print("  ⚠️ Energy system may be too aggressive in termination")
        
        print(f"\n💡 RECOMMENDATIONS:")
        if "energy_based" in analysis:
            energy_stats = analysis["energy_based"]
            
            if energy_stats["avg_final_energy"] < 20:
                print("  • Consider reducing energy drain rate")
            if energy_stats["avg_branches_spawned"] < 0.5:
                print("  • Consider lowering branching threshold")
            if energy_stats["avg_exploration_quality"] < 0.6:
                print("  • Consider adjusting interest threshold")
        
        print()


def main():
    """Run the evaluation with real brain memory."""
    print("🚀 Loading brain system...")
    
    # Load existing brain system
    brainstem = GridWorldBrainstem(seed=42, use_sockets=False)
    world_graph = brainstem.brain_client.get_world_graph()
    
    if world_graph.node_count() < 100:
        print("❌ Need at least 100 nodes for meaningful evaluation")
        print(f"Current graph has {world_graph.node_count()} nodes")
        return
    
    print(f"✅ Loaded graph with {world_graph.node_count()} nodes")
    
    # Create test contexts (realistic robot scenarios)
    test_contexts = [
        [0.8, 0.2, 0.1, 0.0, 0.5, 0.3, 0.7, 0.4],  # Wall ahead scenario
        [0.1, 0.9, 0.6, 0.3, 0.8, 0.2, 0.4, 0.7],  # Food nearby scenario  
        [0.5, 0.5, 0.8, 0.9, 0.1, 0.6, 0.3, 0.2],  # Danger present scenario
        [0.3, 0.7, 0.4, 0.2, 0.9, 0.1, 0.8, 0.5],  # Exploration scenario
        [0.6, 0.4, 0.2, 0.8, 0.3, 0.9, 0.1, 0.7]   # Mixed scenario
    ]
    
    # Run evaluation
    evaluator = TraversalEvaluator(world_graph, test_contexts)
    results = evaluator.run_comprehensive_evaluation(num_trials=20)
    
    return results


if __name__ == "__main__":
    results = main()