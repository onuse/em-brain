#!/usr/bin/env python3
"""
Brain Scaling Analysis Tool

Analyzes performance bottlenecks and scaling characteristics of the brain
architecture. Identifies O(n²) or worse scaling issues and optimization opportunities.
"""

import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.brain import MinimalBrain

class ScalingAnalyzer:
    def __init__(self):
        self.results = {}
        
    def analyze_scaling_characteristics(self, max_experiences: int = 200, step_size: int = 20):
        """Analyze how performance scales with number of experiences."""
        
        print("🔬 Brain Scaling Analysis")
        print("=" * 60)
        
        experience_counts = list(range(step_size, max_experiences + 1, step_size))
        
        results = {
            'experience_counts': experience_counts,
            'cycle_times': [],
            'memory_usage': [],
            'similarity_times': [],
            'activation_times': [],
            'prediction_times': [],
            'storage_times': [],
            'working_memory_sizes': []
        }
        
        for num_exp in experience_counts:
            print(f"\n📊 Testing with {num_exp} experiences...")
            
            # Initialize fresh brain for each test
            brain = MinimalBrain(
                enable_logging=False,
                enable_persistence=False,
                enable_phase2_adaptations=False  # Test base performance
            )
            
            # Create varied experiences to test realistic scenarios
            experiences_data = self._generate_test_experiences(num_exp)
            
            # Store experiences
            storage_start = time.time()
            for i, (sensory, action, outcome) in enumerate(experiences_data):
                brain.store_experience(sensory, action, outcome)
            storage_time = time.time() - storage_start
            
            # Test prediction performance with the brain fully loaded
            test_input = np.random.normal(0, 1, 16).tolist()
            
            # Warm up
            brain.process_sensory_input(test_input)
            
            # Time the critical cycle
            cycle_start = time.time()
            predicted_action, brain_state = brain.process_sensory_input(test_input)
            cycle_time = time.time() - cycle_start
            
            # Get detailed timing breakdown
            timing_breakdown = self._detailed_timing_analysis(brain, test_input)
            
            # Collect results
            results['cycle_times'].append(cycle_time * 1000)  # ms
            results['memory_usage'].append(brain_state.get('total_experiences', 0))
            results['similarity_times'].append(timing_breakdown['similarity_time'] * 1000)
            results['activation_times'].append(timing_breakdown['activation_time'] * 1000)
            results['prediction_times'].append(timing_breakdown['prediction_time'] * 1000)
            results['storage_times'].append(storage_time * 1000)
            results['working_memory_sizes'].append(brain_state.get('working_memory_size', 0))
            
            print(f"   Cycle time: {cycle_time*1000:.1f}ms")
            print(f"   Working memory: {brain_state.get('working_memory_size', 0)} experiences")
        
        self.results = results
        return results
    
    def _generate_test_experiences(self, count: int) -> List[Tuple[List[float], List[float], List[float]]]:
        """Generate realistic test experiences with some patterns."""
        experiences = []
        
        for i in range(count):
            # Create sensory input with some structure
            base_sensors = [
                1.0 + i * 0.01,  # x position
                2.0 + np.sin(i * 0.1),  # y position with pattern
                np.random.uniform(0, 360),  # heading
                np.random.uniform(0, 1),  # ultrasonic
                np.random.uniform(100, 300)  # lidar
            ] + np.random.normal(0, 0.1, 11).tolist()  # Additional sensors
            
            # Action with some correlation to sensors
            action = [
                0.5 * np.sin(base_sensors[2] * np.pi / 180),  # steering based on heading
                0.3 if base_sensors[3] > 0.7 else -0.2,  # speed based on distance
                np.random.uniform(-10, 10),  # servo
                0.0  # aux
            ]
            
            # Outcome with slight noise
            outcome = np.array(base_sensors) + np.random.normal(0, 0.05, 16)
            
            experiences.append((base_sensors, action, outcome.tolist()))
        
        return experiences
    
    def _detailed_timing_analysis(self, brain, test_input: List[float]) -> Dict[str, float]:
        """Analyze timing of individual brain components."""
        
        # Time similarity search
        sim_start = time.time()
        experience_vectors = []
        experience_ids = []
        for exp_id, exp in brain.experience_storage._experiences.items():
            experience_vectors.append(exp.get_context_vector())
            experience_ids.append(exp_id)
        
        if experience_vectors:
            similar_experiences = brain.similarity_engine.find_similar_experiences(
                test_input, experience_vectors, experience_ids, max_results=20, min_similarity=0.4
            )
        similarity_time = time.time() - sim_start
        
        # Time activation dynamics
        activation_start = time.time()
        if brain.use_utility_based_activation:
            brain._activate_by_utility(test_input)
        else:
            brain._activate_similar_experiences(test_input)
        activation_time = time.time() - activation_start
        
        # Time prediction
        prediction_start = time.time()
        if experience_vectors:
            predicted_action, confidence, details = brain.prediction_engine.predict_action(
                test_input, brain.similarity_engine, brain.activation_dynamics,
                brain.experience_storage._experiences, 4
            )
        prediction_time = time.time() - prediction_start
        
        return {
            'similarity_time': similarity_time,
            'activation_time': activation_time,
            'prediction_time': prediction_time
        }
    
    def analyze_complexity_scaling(self):
        """Analyze computational complexity scaling patterns."""
        
        if not self.results:
            print("❌ No results to analyze. Run analyze_scaling_characteristics() first.")
            return
        
        print("\n🔍 Complexity Analysis")
        print("=" * 40)
        
        exp_counts = np.array(self.results['experience_counts'])
        cycle_times = np.array(self.results['cycle_times'])
        
        # Fit different complexity models
        complexities = {
            'O(1)': np.ones_like(exp_counts),
            'O(log n)': np.log(exp_counts),
            'O(n)': exp_counts,
            'O(n log n)': exp_counts * np.log(exp_counts),
            'O(n²)': exp_counts ** 2,
            'O(n³)': exp_counts ** 3
        }
        
        best_fit = None
        best_r_squared = -1
        
        for name, model in complexities.items():
            # Normalize model to fit data range
            model_normalized = model / np.max(model) * np.max(cycle_times)
            
            # Calculate R-squared
            ss_res = np.sum((cycle_times - model_normalized) ** 2)
            ss_tot = np.sum((cycle_times - np.mean(cycle_times)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            print(f"   {name:10} R² = {r_squared:.3f}")
            
            if r_squared > best_r_squared:
                best_r_squared = r_squared
                best_fit = name
        
        print(f"\n🎯 Best fit: {best_fit} (R² = {best_r_squared:.3f})")
        
        # Performance scaling rate
        if len(cycle_times) > 1:
            scaling_rate = (cycle_times[-1] - cycle_times[0]) / (exp_counts[-1] - exp_counts[0])
            print(f"📈 Scaling rate: {scaling_rate:.2f} ms per additional experience")
        
        return best_fit, best_r_squared
    
    def identify_bottlenecks(self):
        """Identify the main performance bottlenecks."""
        
        if not self.results:
            print("❌ No results to analyze. Run analyze_scaling_characteristics() first.")
            return
        
        print("\n🔥 Bottleneck Analysis")
        print("=" * 40)
        
        # Analyze which component takes the most time
        similarity_times = np.array(self.results['similarity_times'])
        activation_times = np.array(self.results['activation_times'])
        prediction_times = np.array(self.results['prediction_times'])
        
        avg_similarity = np.mean(similarity_times)
        avg_activation = np.mean(activation_times)
        avg_prediction = np.mean(prediction_times)
        total_avg = avg_similarity + avg_activation + avg_prediction
        
        print(f"📊 Average time breakdown:")
        print(f"   Similarity search: {avg_similarity:.1f}ms ({avg_similarity/total_avg*100:.1f}%)")
        print(f"   Activation dynamics: {avg_activation:.1f}ms ({avg_activation/total_avg*100:.1f}%)")
        print(f"   Prediction engine: {avg_prediction:.1f}ms ({avg_prediction/total_avg*100:.1f}%)")
        
        # Identify which component scales worst
        exp_counts = np.array(self.results['experience_counts'])
        
        # Calculate scaling rates for each component
        if len(exp_counts) > 1:
            sim_scaling = (similarity_times[-1] - similarity_times[0]) / (exp_counts[-1] - exp_counts[0])
            act_scaling = (activation_times[-1] - activation_times[0]) / (exp_counts[-1] - exp_counts[0])
            pred_scaling = (prediction_times[-1] - prediction_times[0]) / (exp_counts[-1] - exp_counts[0])
            
            print(f"\n📈 Scaling rates (ms per additional experience):")
            print(f"   Similarity search: {sim_scaling:.3f} ms/exp")
            print(f"   Activation dynamics: {act_scaling:.3f} ms/exp")
            print(f"   Prediction engine: {pred_scaling:.3f} ms/exp")
        
        # Memory scaling analysis
        working_memory_sizes = np.array(self.results['working_memory_sizes'])
        if len(working_memory_sizes) > 1:
            wm_growth = (working_memory_sizes[-1] - working_memory_sizes[0]) / (exp_counts[-1] - exp_counts[0])
            print(f"   Working memory growth: {wm_growth:.3f} experiences/exp")
        
        return {
            'similarity_avg': avg_similarity,
            'activation_avg': avg_activation,
            'prediction_avg': avg_prediction,
            'worst_scaling_component': self._identify_worst_scaling_component()
        }
    
    def _identify_worst_scaling_component(self):
        """Identify which component has the worst scaling characteristics."""
        
        exp_counts = np.array(self.results['experience_counts'])
        if len(exp_counts) <= 1:
            return "insufficient_data"
        
        similarity_times = np.array(self.results['similarity_times'])
        activation_times = np.array(self.results['activation_times'])
        prediction_times = np.array(self.results['prediction_times'])
        
        # Calculate relative scaling (how much each component grows proportionally)
        sim_growth = (similarity_times[-1] / similarity_times[0]) if similarity_times[0] > 0 else 1
        act_growth = (activation_times[-1] / activation_times[0]) if activation_times[0] > 0 else 1
        pred_growth = (prediction_times[-1] / prediction_times[0]) if prediction_times[0] > 0 else 1
        
        growth_rates = {
            'similarity': sim_growth,
            'activation': act_growth,
            'prediction': pred_growth
        }
        
        return max(growth_rates, key=growth_rates.get)
    
    def generate_optimization_recommendations(self):
        """Generate specific optimization recommendations."""
        
        print("\n🎯 Optimization Recommendations")
        print("=" * 50)
        
        bottlenecks = self.identify_bottlenecks()
        complexity_fit, r_squared = self.analyze_complexity_scaling()
        
        recommendations = []
        
        # Based on complexity scaling
        if complexity_fit in ['O(n²)', 'O(n³)']:
            recommendations.append("🚨 CRITICAL: Algorithm has polynomial scaling - needs fundamental redesign")
            recommendations.append("   → Consider approximate similarity search (LSH, FAISS)")
            recommendations.append("   → Implement experience pruning/clustering")
            recommendations.append("   → Add GPU batch processing for all operations")
        
        elif complexity_fit in ['O(n log n)', 'O(n)']:
            recommendations.append("⚡ GOOD: Linear scaling detected - optimize constants")
            recommendations.append("   → Focus on GPU utilization")
            recommendations.append("   → Optimize memory access patterns")
        
        # Based on bottleneck analysis
        if bottlenecks['similarity_avg'] > 50:  # >50ms is too slow
            recommendations.append("🔍 Similarity search is slow:")
            recommendations.append("   → Implement FAISS or similar fast similarity search")
            recommendations.append("   → Use batch GPU operations")
            recommendations.append("   → Consider approximate methods (LSH)")
        
        if bottlenecks['activation_avg'] > 30:  # >30ms is too slow
            recommendations.append("🧠 Activation dynamics are slow:")
            recommendations.append("   → Vectorize activation spreading")
            recommendations.append("   → Use sparse matrices for connections")
            recommendations.append("   → Limit working memory size")
        
        if bottlenecks['prediction_avg'] > 20:  # >20ms is too slow
            recommendations.append("🔮 Prediction engine is slow:")
            recommendations.append("   → Cache consensus calculations")
            recommendations.append("   → Limit similar experience search")
            recommendations.append("   → Use pattern prediction for common cases")
        
        # Real-time constraint analysis
        max_cycle_time = max(self.results['cycle_times'])
        if max_cycle_time > 50:
            recommendations.append(f"⏰ REAL-TIME VIOLATION: {max_cycle_time:.1f}ms > 50ms target")
            recommendations.append("   → Must achieve 7-10x speedup for real-time control")
            recommendations.append("   → Consider parallel processing pipeline")
            recommendations.append("   → Implement experience streaming/batching")
        
        # Memory growth analysis
        if len(self.results['working_memory_sizes']) > 1:
            wm_start = self.results['working_memory_sizes'][0]
            wm_end = self.results['working_memory_sizes'][-1]
            if wm_end > wm_start * 2:
                recommendations.append("🧮 Working memory growing too fast:")
                recommendations.append("   → Implement adaptive memory decay")
                recommendations.append("   → Add memory pressure-based pruning")
                recommendations.append("   → Consider hierarchical memory organization")
        
        for rec in recommendations:
            print(rec)
        
        return recommendations

def main():
    """Run comprehensive brain scaling analysis."""
    
    analyzer = ScalingAnalyzer()
    
    print("🔬 Starting Brain Performance Scaling Analysis")
    print("This will test performance with increasing numbers of experiences")
    print("to identify bottlenecks and scaling issues.\n")
    
    # Run scaling analysis
    results = analyzer.analyze_scaling_characteristics(max_experiences=100, step_size=20)
    
    # Analyze results
    complexity_fit, r_squared = analyzer.analyze_complexity_scaling()
    bottlenecks = analyzer.identify_bottlenecks()
    recommendations = analyzer.generate_optimization_recommendations()
    
    # Summary
    print(f"\n{'='*60}")
    print("📋 EXECUTIVE SUMMARY")
    print(f"{'='*60}")
    
    max_cycle_time = max(results['cycle_times'])
    target_time = 50  # 50ms for real-time
    speedup_needed = max_cycle_time / target_time
    
    print(f"Current performance: {max_cycle_time:.1f}ms per cycle (with 100 experiences)")
    print(f"Real-time target: {target_time}ms per cycle")
    print(f"Speedup needed: {speedup_needed:.1f}x")
    print(f"Computational complexity: {complexity_fit}")
    print(f"Primary bottleneck: {bottlenecks.get('worst_scaling_component', 'unknown')}")
    
    if speedup_needed > 5:
        print("\n🚨 VERDICT: Major architectural changes needed")
    elif speedup_needed > 2:
        print("\n⚡ VERDICT: Significant optimizations needed")
    else:
        print("\n✅ VERDICT: Minor optimizations sufficient")

if __name__ == "__main__":
    main()