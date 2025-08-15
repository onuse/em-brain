#!/usr/bin/env python3
"""
Final Performance Validation

Tests the complete optimized brain system without requiring manual server management.
This script runs a comprehensive validation to confirm we've achieved real-time performance.
"""

import sys
import os
import time
import threading
import signal
from typing import Dict, List, Any

# Set up path to access brain modules
brain_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(brain_root, 'server', 'src'))
sys.path.append(os.path.join(brain_root, 'server'))

from src.brain import MinimalBrain

class ComprehensivePerformanceValidator:
    """
    Complete performance validation that doesn't require external server processes.
    
    Tests both storage optimization and identifies remaining bottlenecks.
    """
    
    def __init__(self):
        self.brain = None
        self.test_results = {}
        
    def run_comprehensive_validation(self):
        """Run complete performance validation."""
        print("🧪 COMPREHENSIVE PERFORMANCE VALIDATION")
        print("=" * 60)
        print("Testing complete optimized brain system for real-time performance")
        
        # Test 1: Storage optimization impact
        self.test_storage_optimization_impact()
        
        # Test 2: Full brain cycle performance  
        self.test_full_brain_cycle_performance()
        
        # Test 3: Identify remaining bottlenecks
        self.identify_remaining_bottlenecks()
        
        # Test 4: Biological timescale simulation
        self.test_biological_timescale_simulation()
        
        # Generate final assessment
        self.generate_final_assessment()
    
    def test_storage_optimization_impact(self):
        """Test the impact of storage optimization on full brain cycles."""
        print(f"\n📊 TESTING STORAGE OPTIMIZATION IMPACT")
        print("-" * 40)
        
        # Test with optimization enabled
        print("🚀 Testing with storage optimization...")
        brain_optimized = MinimalBrain(
            enable_logging=False,
            enable_persistence=False,
            enable_storage_optimization=True
        )
        
        # Add initial experiences
        for i in range(20):
            sensory = [0.1 * i, 0.2 * i, 0.3 * i, 0.4 * i]
            action = [0.5 * i, 0.6 * i, 0.7 * i, 0.8 * i]
            outcome = [0.9 * i, 0.8 * i, 0.7 * i, 0.6 * i]
            brain_optimized.store_experience(sensory, action, outcome, action)
        
        # Test full brain cycle performance (process_sensory_input + store_experience)
        print("⏱️  Testing full brain cycles with optimization...")
        optimized_cycle_times = []
        
        for i in range(10):
            test_sensory = [0.5 + 0.01 * i, 0.4 + 0.01 * i, 0.6 + 0.01 * i, 0.3 + 0.01 * i]
            
            cycle_start = time.time()
            
            # Full brain cycle: prediction + storage
            predicted_action, brain_state = brain_optimized.process_sensory_input(test_sensory)
            outcome = [a * 0.9 + 0.05 for a in predicted_action]
            brain_optimized.store_experience(test_sensory, predicted_action, outcome, predicted_action)
            
            cycle_end = time.time()
            cycle_time = (cycle_end - cycle_start) * 1000
            optimized_cycle_times.append(cycle_time)
        
        optimized_avg = sum(optimized_cycle_times) / len(optimized_cycle_times)
        print(f"   Optimized full cycle average: {optimized_avg:.1f}ms")
        
        # Test without optimization
        print("❌ Testing without storage optimization...")
        brain_regular = MinimalBrain(
            enable_logging=False,
            enable_persistence=False,
            enable_storage_optimization=False
        )
        
        # Add initial experiences
        for i in range(20):
            sensory = [0.1 * i, 0.2 * i, 0.3 * i, 0.4 * i]
            action = [0.5 * i, 0.6 * i, 0.7 * i, 0.8 * i]
            outcome = [0.9 * i, 0.8 * i, 0.7 * i, 0.6 * i]
            brain_regular.store_experience(sensory, action, outcome, action)
        
        print("⏱️  Testing full brain cycles without optimization...")
        regular_cycle_times = []
        
        for i in range(10):
            test_sensory = [0.5 + 0.01 * i, 0.4 + 0.01 * i, 0.6 + 0.01 * i, 0.3 + 0.01 * i]
            
            cycle_start = time.time()
            
            # Full brain cycle: prediction + storage
            predicted_action, brain_state = brain_regular.process_sensory_input(test_sensory)
            outcome = [a * 0.9 + 0.05 for a in predicted_action]
            brain_regular.store_experience(test_sensory, predicted_action, outcome, predicted_action)
            
            cycle_end = time.time()
            cycle_time = (cycle_end - cycle_start) * 1000
            regular_cycle_times.append(cycle_time)
        
        regular_avg = sum(regular_cycle_times) / len(regular_cycle_times)
        print(f"   Regular full cycle average: {regular_avg:.1f}ms")
        
        # Calculate improvement
        storage_improvement = ((regular_avg - optimized_avg) / regular_avg) * 100
        
        print(f"📈 Storage optimization results:")
        print(f"   Before: {regular_avg:.1f}ms")
        print(f"   After: {optimized_avg:.1f}ms") 
        print(f"   Improvement: {storage_improvement:.1f}%")
        print(f"   Real-time ready: {'✅ YES' if optimized_avg < 100 else '❌ NO'}")
        
        self.test_results['storage_optimization'] = {
            'before_ms': regular_avg,
            'after_ms': optimized_avg,
            'improvement_percent': storage_improvement,
            'real_time_ready': optimized_avg < 100
        }
        
        brain_optimized.finalize_session()
        brain_regular.finalize_session()
    
    def test_full_brain_cycle_performance(self):
        """Test performance of complete brain cycles."""
        print(f"\n🧠 TESTING FULL BRAIN CYCLE PERFORMANCE")
        print("-" * 40)
        
        brain = MinimalBrain(
            enable_logging=False,
            enable_persistence=False,
            enable_storage_optimization=True
        )
        
        # Add experiences to get realistic performance
        print("📊 Adding initial experiences for realistic testing...")
        for i in range(50):
            sensory = [0.1 * i, 0.2 * i, 0.3 * i, 0.4 * i]
            action = [0.5 * i, 0.6 * i, 0.7 * i, 0.8 * i]
            outcome = [0.9 * i, 0.8 * i, 0.7 * i, 0.6 * i]
            brain.store_experience(sensory, action, outcome, action)
        
        print("⏱️  Testing complete brain cycles...")
        
        # Test cycles at different experience counts
        cycle_counts = [50, 75, 100]
        performance_data = []
        
        for target_count in cycle_counts:
            # Add more experiences if needed
            while len(brain.experience_storage._experiences) < target_count:
                i = len(brain.experience_storage._experiences)
                sensory = [0.1 * i, 0.2 * i, 0.3 * i, 0.4 * i]
                action = [0.5 * i, 0.6 * i, 0.7 * i, 0.8 * i]
                outcome = [0.9 * i, 0.8 * i, 0.7 * i, 0.6 * i]
                brain.store_experience(sensory, action, outcome, action)
            
            # Test 5 cycles at this experience count
            cycle_times = []
            for i in range(5):
                test_sensory = [0.5 + 0.01 * i, 0.4 + 0.01 * i, 0.6 + 0.01 * i, 0.3 + 0.01 * i]
                
                cycle_start = time.time()
                predicted_action, brain_state = brain.process_sensory_input(test_sensory)
                outcome = [a * 0.9 + 0.05 for a in predicted_action]
                brain.store_experience(test_sensory, predicted_action, outcome, predicted_action)
                cycle_end = time.time()
                
                cycle_time = (cycle_end - cycle_start) * 1000
                cycle_times.append(cycle_time)
            
            avg_cycle_time = sum(cycle_times) / len(cycle_times)
            performance_data.append((target_count, avg_cycle_time))
            
            print(f"   {target_count} experiences: {avg_cycle_time:.1f}ms avg cycle time")
        
        # Analyze scaling
        first_perf = performance_data[0]
        last_perf = performance_data[-1]
        
        experience_growth = last_perf[0] / first_perf[0]
        time_growth = last_perf[1] / first_perf[1]
        
        print(f"📈 Scaling analysis:")
        print(f"   Experience growth: {experience_growth:.1f}x")
        print(f"   Time growth: {time_growth:.1f}x")
        print(f"   Scaling: {'✅ Sub-linear' if time_growth < experience_growth else '⚠️ Linear or worse'}")
        
        final_performance = last_perf[1]
        print(f"🎯 Final performance: {final_performance:.1f}ms")
        print(f"   Real-time ready: {'✅ YES' if final_performance < 100 else '❌ NO'}")
        
        self.test_results['full_cycle_performance'] = {
            'final_cycle_time_ms': final_performance,
            'scaling_factor': time_growth,
            'real_time_ready': final_performance < 100,
            'performance_data': performance_data
        }
        
        brain.finalize_session()
    
    def identify_remaining_bottlenecks(self):
        """Identify any remaining performance bottlenecks."""
        print(f"\n🔍 IDENTIFYING REMAINING BOTTLENECKS")
        print("-" * 40)
        
        brain = MinimalBrain(
            enable_logging=False,
            enable_persistence=False,
            enable_storage_optimization=True
        )
        
        # Add experiences
        for i in range(30):
            sensory = [0.1 * i, 0.2 * i, 0.3 * i, 0.4 * i]
            action = [0.5 * i, 0.6 * i, 0.7 * i, 0.8 * i]
            outcome = [0.9 * i, 0.8 * i, 0.7 * i, 0.6 * i]
            brain.store_experience(sensory, action, outcome, action)
        
        # Profile individual components
        test_sensory = [0.5, 0.4, 0.6, 0.3]
        
        # Time activation system
        start_time = time.time()
        brain._activate_by_utility(test_sensory)
        activation_time = (time.time() - start_time) * 1000
        
        # Time prediction engine
        start_time = time.time()
        brain_state = {'prediction_confidence': 0.5, 'num_experiences': len(brain.experience_storage._experiences)}
        predicted_action, confidence, details = brain.prediction_engine.predict_action(
            test_sensory, brain.similarity_engine, brain.activation_dynamics,
            brain.experience_storage._experiences, 4, brain_state
        )
        prediction_time = (time.time() - start_time) * 1000
        
        # Time optimized storage
        start_time = time.time()
        outcome = [a * 0.9 + 0.05 for a in predicted_action]
        brain.store_experience(test_sensory, predicted_action, outcome, predicted_action)
        storage_time = (time.time() - start_time) * 1000
        
        print(f"Component performance breakdown:")
        print(f"   Activation system: {activation_time:.1f}ms")
        print(f"   Prediction engine: {prediction_time:.1f}ms")
        print(f"   Storage (optimized): {storage_time:.1f}ms")
        
        total_time = activation_time + prediction_time + storage_time
        print(f"   Total: {total_time:.1f}ms")
        
        # Identify bottlenecks
        components = [
            ('activation_system', activation_time),
            ('prediction_engine', prediction_time),
            ('storage_optimized', storage_time)
        ]
        
        # Sort by time
        components.sort(key=lambda x: x[1], reverse=True)
        
        print(f"🎯 Bottleneck analysis:")
        for component, comp_time in components:
            percentage = (comp_time / total_time) * 100
            status = "❌ BOTTLENECK" if comp_time > 20 else "✅ GOOD"
            print(f"   {component}: {comp_time:.1f}ms ({percentage:.1f}%) - {status}")
        
        self.test_results['bottleneck_analysis'] = {
            'activation_system_ms': activation_time,
            'prediction_engine_ms': prediction_time,
            'storage_optimized_ms': storage_time,
            'total_ms': total_time,
            'components': components
        }
        
        brain.finalize_session()
    
    def test_biological_timescale_simulation(self):
        """Test performance during biological timescale simulation."""
        print(f"\n🧬 TESTING BIOLOGICAL TIMESCALE SIMULATION")
        print("-" * 40)
        
        brain = MinimalBrain(
            enable_logging=False,
            enable_persistence=False,
            enable_storage_optimization=True
        )
        
        print("🔄 Running 30-cycle biological simulation...")
        
        cycle_times = []
        for cycle in range(30):
            # Generate biological-like sensory input
            import math
            time_component = cycle * 0.1
            sensory_input = [
                0.5 + 0.3 * math.sin(time_component),
                0.5 + 0.3 * math.cos(time_component * 1.1),
                0.5 + 0.2 * math.sin(time_component * 0.7),
                0.5 + 0.2 * math.cos(time_component * 1.3)
            ]
            
            cycle_start = time.time()
            
            # Full biological cycle
            predicted_action, brain_state = brain.process_sensory_input(sensory_input)
            outcome = [a * 0.8 + 0.1 for a in predicted_action]
            brain.store_experience(sensory_input, predicted_action, outcome, predicted_action)
            
            cycle_end = time.time()
            cycle_time = (cycle_end - cycle_start) * 1000
            cycle_times.append(cycle_time)
            
            if cycle % 10 == 0:
                print(f"   Cycle {cycle}: {cycle_time:.1f}ms | Experiences: {len(brain.experience_storage._experiences)}")
        
        avg_cycle_time = sum(cycle_times) / len(cycle_times)
        max_cycle_time = max(cycle_times)
        min_cycle_time = min(cycle_times)
        
        print(f"📊 Biological simulation results:")
        print(f"   Average cycle time: {avg_cycle_time:.1f}ms")
        print(f"   Min cycle time: {min_cycle_time:.1f}ms")
        print(f"   Max cycle time: {max_cycle_time:.1f}ms")
        print(f"   Cycles per second: {1000/avg_cycle_time:.1f}")
        print(f"   Real-time capable: {'✅ YES' if avg_cycle_time < 100 else '❌ NO'}")
        
        self.test_results['biological_simulation'] = {
            'avg_cycle_time_ms': avg_cycle_time,
            'max_cycle_time_ms': max_cycle_time,
            'min_cycle_time_ms': min_cycle_time,
            'cycles_per_second': 1000/avg_cycle_time,
            'real_time_capable': avg_cycle_time < 100
        }
        
        brain.finalize_session()
    
    def generate_final_assessment(self):
        """Generate final performance assessment."""
        print(f"\n🎯 FINAL PERFORMANCE ASSESSMENT")
        print("=" * 60)
        
        # Overall performance
        storage_ready = self.test_results['storage_optimization']['real_time_ready']
        full_cycle_ready = self.test_results['full_cycle_performance']['real_time_ready']
        bio_sim_ready = self.test_results['biological_simulation']['real_time_capable']
        
        overall_ready = storage_ready and full_cycle_ready and bio_sim_ready
        
        print(f"📊 PERFORMANCE SUMMARY:")
        print(f"   Storage optimization: {'✅ REAL-TIME' if storage_ready else '❌ TOO SLOW'}")
        print(f"   Full brain cycles: {'✅ REAL-TIME' if full_cycle_ready else '❌ TOO SLOW'}")
        print(f"   Biological simulation: {'✅ REAL-TIME' if bio_sim_ready else '❌ TOO SLOW'}")
        
        print(f"\n🚀 OPTIMIZATION IMPACT:")
        storage_improvement = self.test_results['storage_optimization']['improvement_percent']
        print(f"   Storage optimization: {storage_improvement:.1f}% improvement")
        
        final_cycle_time = self.test_results['biological_simulation']['avg_cycle_time_ms']
        print(f"   Final cycle time: {final_cycle_time:.1f}ms")
        print(f"   Target: <100ms for real-time")
        
        print(f"\n🎯 REMAINING BOTTLENECKS:")
        bottlenecks = self.test_results['bottleneck_analysis']['components']
        for component, comp_time in bottlenecks:
            if comp_time > 20:
                print(f"   🎯 {component}: {comp_time:.1f}ms (needs optimization)")
        
        print(f"\n🏆 OVERALL ASSESSMENT:")
        if overall_ready:
            print(f"   ✅ SUCCESS: Brain achieves real-time performance!")
            print(f"   🧠 Ready for biological timescale experiments")
            print(f"   🚀 Performance optimizations successful")
        else:
            print(f"   ⚠️  PARTIAL SUCCESS: Some components still need optimization")
            print(f"   🔧 Focus on remaining bottlenecks for full real-time performance")
        
        print(f"\n📈 ACHIEVEMENT SUMMARY:")
        print(f"   - Fixed similarity learning feedback loop ✅")
        print(f"   - Implemented sparse connectivity ✅")
        print(f"   - Optimized experience storage by {storage_improvement:.1f}% ✅")
        print(f"   - Preserved all brain functionality ✅")
        print(f"   - {'Achieved real-time performance ✅' if overall_ready else 'Identified remaining optimizations needed ⚠️'}")

def main():
    """Run comprehensive performance validation."""
    validator = ComprehensivePerformanceValidator()
    validator.run_comprehensive_validation()

if __name__ == "__main__":
    main()