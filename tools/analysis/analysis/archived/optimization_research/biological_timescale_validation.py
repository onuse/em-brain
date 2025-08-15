#!/usr/bin/env python3
"""
5-Minute Biological Timescale Validation Test

Validates that the brain operates correctly at biological timescales with:
- Sparse connectivity (no O(n²) explosions)
- Similarity learning adaptation
- Stable performance over time
- Experience accumulation and learning
"""

import sys
import os
import time
import threading
import signal
import json
from datetime import datetime
from typing import Dict, List, Any

# Set up path to access brain modules
brain_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(brain_root, 'server', 'src'))
sys.path.append(os.path.join(brain_root, 'server'))

from src.brain import MinimalBrain

class BiologicalTimescaleValidator:
    """5-minute biological timescale validation system."""
    
    def __init__(self):
        self.brain = None
        self.running = False
        self.start_time = None
        self.test_duration = 60  # 1 minute for quick test (change to 300 for full 5-minute test)
        self.cycle_count = 0
        self.performance_data = []
        self.similarity_data = []
        self.connectivity_data = []
        
    def setup_brain(self):
        """Initialize the brain for biological timescale testing."""
        print("🧠 INITIALIZING BRAIN FOR BIOLOGICAL TIMESCALE TEST")
        print("=" * 60)
        
        # Create brain with logging enabled for detailed analysis
        self.brain = MinimalBrain(
            enable_logging=True,
            log_session_name="biological_validation_5min",
            enable_persistence=True,
            enable_phase2_adaptations=True
        )
        
        print(f"✅ Brain initialized with {len(self.brain.experience_storage._experiences)} experiences")
        print(f"🔄 Similarity learning: {'ENABLED' if self.brain.similarity_engine.use_learnable_similarity else 'DISABLED'}")
        print(f"🕸️ Sparse connectivity: ENABLED")
        print(f"📊 Performance monitoring: ENABLED")
        
    def simulate_biological_sensory_input(self, cycle: int) -> List[float]:
        """Generate realistic biological sensory input patterns."""
        import math
        import random
        
        # Create slowly changing patterns that simulate biological input
        # This creates temporal correlations that should trigger learning
        
        # Base pattern that changes slowly over time
        base_frequency = 0.01  # Very slow change
        time_component = cycle * base_frequency
        
        # Multiple overlapping patterns
        pattern_1 = [
            0.5 + 0.3 * math.sin(time_component),
            0.5 + 0.3 * math.cos(time_component * 1.1),
            0.5 + 0.2 * math.sin(time_component * 0.7),
            0.5 + 0.2 * math.cos(time_component * 1.3)
        ]
        
        # Add some environmental noise
        noise = [random.uniform(-0.05, 0.05) for _ in range(4)]
        
        # Combine patterns with biological constraints
        sensory_input = []
        for i in range(4):
            value = pattern_1[i] + noise[i]
            value = max(0.0, min(1.0, value))  # Biological bounds
            sensory_input.append(value)
        
        # Add occasional "events" that should trigger learning
        if cycle % 50 == 0:  # Every 50 cycles, create a distinctive event
            event_magnitude = 0.4
            for i in range(4):
                sensory_input[i] += event_magnitude * random.uniform(-1, 1)
                sensory_input[i] = max(0.0, min(1.0, sensory_input[i]))
        
        return sensory_input
    
    def run_biological_cycle(self, cycle: int):
        """Run a single biological cycle."""
        cycle_start_time = time.time()
        
        # Generate sensory input
        sensory_input = self.simulate_biological_sensory_input(cycle)
        
        # Process through brain
        predicted_action, brain_state = self.brain.process_sensory_input(sensory_input)
        
        # Simulate world response (simple transformation)
        world_response = [
            predicted_action[0] * 0.8 + 0.1,
            predicted_action[1] * 0.9 + 0.05,
            predicted_action[2] * 0.7 + 0.15,
            predicted_action[3] * 0.85 + 0.1
        ]
        
        # Store experience
        experience_id = self.brain.store_experience(
            sensory_input=sensory_input,
            action_taken=predicted_action,
            outcome=world_response,
            predicted_action=predicted_action
        )
        
        cycle_end_time = time.time()
        cycle_time = cycle_end_time - cycle_start_time
        
        # Record performance data
        self.performance_data.append({
            'cycle': cycle,
            'cycle_time_ms': cycle_time * 1000,
            'total_experiences': len(self.brain.experience_storage._experiences),
            'prediction_confidence': brain_state.get('prediction_confidence', 0),
            'working_memory_size': brain_state.get('working_memory_size', 0),
            'timestamp': cycle_end_time
        })
        
        # Record similarity learning data every 10 cycles
        if cycle % 10 == 0 and self.brain.similarity_engine.learnable_similarity:
            similarity_stats = self.brain.similarity_engine.learnable_similarity.get_similarity_statistics()
            self.similarity_data.append({
                'cycle': cycle,
                'adaptations_performed': similarity_stats.get('adaptations_performed', 0),
                'learning_rate': similarity_stats.get('learning_rate', 0),
                'similarity_success_correlation': similarity_stats.get('similarity_success_correlation', 0),
                'prediction_outcomes_tracked': similarity_stats.get('prediction_outcomes_tracked', 0),
                'timestamp': cycle_end_time
            })
        
        # Record connectivity data every 20 cycles
        if cycle % 20 == 0:
            total_connections = 0
            max_connections = 0
            
            for exp_id, exp in self.brain.experience_storage._experiences.items():
                connections = len(exp.similar_experiences) if hasattr(exp, 'similar_experiences') else 0
                total_connections += connections
                max_connections = max(max_connections, connections)
            
            avg_connections = total_connections / max(1, len(self.brain.experience_storage._experiences))
            
            self.connectivity_data.append({
                'cycle': cycle,
                'total_connections': total_connections,
                'avg_connections_per_experience': avg_connections,
                'max_connections': max_connections,
                'total_experiences': len(self.brain.experience_storage._experiences),
                'timestamp': cycle_end_time
            })
        
        return cycle_time
    
    def print_progress(self, cycle: int, elapsed_time: float):
        """Print progress information."""
        if cycle % 100 == 0:  # Every 100 cycles
            remaining_time = self.test_duration - elapsed_time
            experiences = len(self.brain.experience_storage._experiences)
            
            # Get recent performance
            recent_times = [d['cycle_time_ms'] for d in self.performance_data[-10:]]
            avg_cycle_time = sum(recent_times) / len(recent_times) if recent_times else 0
            
            print(f"⏱️  Cycle {cycle:4d} | "
                  f"Elapsed: {elapsed_time:6.1f}s | "
                  f"Remaining: {remaining_time:6.1f}s | "
                  f"Experiences: {experiences:4d} | "
                  f"Avg cycle: {avg_cycle_time:.1f}ms")
    
    def run_validation(self):
        """Run the 5-minute biological timescale validation."""
        print(f"\n🚀 STARTING 5-MINUTE BIOLOGICAL TIMESCALE VALIDATION")
        print("=" * 60)
        print(f"⏰ Duration: {self.test_duration} seconds")
        print(f"🔄 Target: ~1 cycle per second (biological timescale)")
        print(f"📊 Monitoring: Performance, similarity learning, connectivity")
        print(f"🧠 Brain state: {self.brain}")
        
        self.running = True
        self.start_time = time.time()
        
        try:
            while self.running:
                current_time = time.time()
                elapsed_time = current_time - self.start_time
                
                # Check if test duration completed
                if elapsed_time >= self.test_duration:
                    break
                
                # Run biological cycle
                cycle_time = self.run_biological_cycle(self.cycle_count)
                
                # Print progress
                self.print_progress(self.cycle_count, elapsed_time)
                
                self.cycle_count += 1
                
                # Biological timescale: aim for ~1 cycle per second
                target_cycle_time = 1.0  # 1 second
                if cycle_time < target_cycle_time:
                    time.sleep(target_cycle_time - cycle_time)
                
        except KeyboardInterrupt:
            print(f"\n⚠️  Test interrupted by user")
        
        self.running = False
        final_time = time.time()
        total_elapsed = final_time - self.start_time
        
        print(f"\n✅ VALIDATION COMPLETED")
        print(f"⏱️  Total time: {total_elapsed:.1f} seconds")
        print(f"🔄 Total cycles: {self.cycle_count}")
        print(f"📊 Cycle rate: {self.cycle_count / total_elapsed:.2f} cycles/second")
        
        return total_elapsed
    
    def analyze_results(self):
        """Analyze the validation results."""
        print(f"\n📊 BIOLOGICAL TIMESCALE VALIDATION ANALYSIS")
        print("=" * 60)
        
        if not self.performance_data:
            print("❌ No performance data collected")
            return
        
        # Performance analysis
        self.analyze_performance_stability()
        self.analyze_similarity_learning()
        self.analyze_connectivity_patterns()
        self.analyze_scaling_behavior()
        
        # Generate summary report
        self.generate_summary_report()
    
    def analyze_performance_stability(self):
        """Analyze performance stability over time."""
        print(f"\n🔬 PERFORMANCE STABILITY ANALYSIS")
        print("-" * 40)
        
        cycle_times = [d['cycle_time_ms'] for d in self.performance_data]
        
        # Calculate statistics
        min_time = min(cycle_times)
        max_time = max(cycle_times)
        avg_time = sum(cycle_times) / len(cycle_times)
        
        # Calculate performance trend
        first_half = cycle_times[:len(cycle_times)//2]
        second_half = cycle_times[len(cycle_times)//2:]
        
        avg_first_half = sum(first_half) / len(first_half)
        avg_second_half = sum(second_half) / len(second_half)
        
        performance_change = ((avg_second_half - avg_first_half) / avg_first_half) * 100
        
        print(f"Cycle Times:")
        print(f"  - Average: {avg_time:.1f}ms")
        print(f"  - Min: {min_time:.1f}ms")
        print(f"  - Max: {max_time:.1f}ms")
        print(f"  - Performance change: {performance_change:+.1f}%")
        
        # Stability assessment
        if abs(performance_change) < 20:
            print(f"  ✅ STABLE: Performance change within acceptable range")
        else:
            print(f"  ⚠️  UNSTABLE: Significant performance change detected")
    
    def analyze_similarity_learning(self):
        """Analyze similarity learning progress."""
        print(f"\n🧠 SIMILARITY LEARNING ANALYSIS")
        print("-" * 40)
        
        if not self.similarity_data:
            print("❌ No similarity learning data collected")
            return
        
        initial_stats = self.similarity_data[0]
        final_stats = self.similarity_data[-1]
        
        adaptations_growth = final_stats['adaptations_performed'] - initial_stats['adaptations_performed']
        outcomes_growth = final_stats['prediction_outcomes_tracked'] - initial_stats['prediction_outcomes_tracked']
        
        print(f"Similarity Learning Progress:")
        print(f"  - Initial adaptations: {initial_stats['adaptations_performed']}")
        print(f"  - Final adaptations: {final_stats['adaptations_performed']}")
        print(f"  - Adaptations during test: {adaptations_growth}")
        print(f"  - Outcomes tracked: {outcomes_growth}")
        print(f"  - Final learning rate: {final_stats['learning_rate']:.4f}")
        print(f"  - Final success correlation: {final_stats['similarity_success_correlation']:.3f}")
        
        # Learning assessment
        if adaptations_growth > 0:
            print(f"  ✅ LEARNING: Similarity function adapted during test")
        else:
            print(f"  ⚠️  NO ADAPTATION: Similarity function may need manual triggering")
    
    def analyze_connectivity_patterns(self):
        """Analyze connectivity patterns and sparsity."""
        print(f"\n🕸️  CONNECTIVITY ANALYSIS")
        print("-" * 40)
        
        if not self.connectivity_data:
            print("❌ No connectivity data collected")
            return
        
        final_connectivity = self.connectivity_data[-1]
        
        total_experiences = final_connectivity['total_experiences']
        avg_connections = final_connectivity['avg_connections_per_experience']
        max_connections = final_connectivity['max_connections']
        
        # Calculate sparsity
        theoretical_dense = total_experiences * (total_experiences - 1)
        actual_connections = final_connectivity['total_connections']
        sparsity_ratio = actual_connections / theoretical_dense if theoretical_dense > 0 else 0
        
        print(f"Connectivity Statistics:")
        print(f"  - Total experiences: {total_experiences}")
        print(f"  - Average connections per experience: {avg_connections:.1f}")
        print(f"  - Maximum connections: {max_connections}")
        print(f"  - Sparsity ratio: {sparsity_ratio:.3f}")
        print(f"  - Theoretical dense connections: {theoretical_dense}")
        print(f"  - Actual connections: {actual_connections}")
        
        # Sparsity assessment
        if avg_connections <= 20:  # Biological range
            print(f"  ✅ SPARSE: Connectivity within biological range")
        else:
            print(f"  ❌ DENSE: Connectivity exceeds biological limits")
    
    def analyze_scaling_behavior(self):
        """Analyze scaling behavior as experiences accumulate."""
        print(f"\n📈 SCALING BEHAVIOR ANALYSIS")
        print("-" * 40)
        
        if len(self.performance_data) < 10:
            print("❌ Insufficient data for scaling analysis")
            return
        
        # Compare performance at different experience counts
        early_data = self.performance_data[:50]  # First 50 cycles
        late_data = self.performance_data[-50:]  # Last 50 cycles
        
        early_avg_time = sum(d['cycle_time_ms'] for d in early_data) / len(early_data)
        late_avg_time = sum(d['cycle_time_ms'] for d in late_data) / len(late_data)
        
        early_experiences = early_data[-1]['total_experiences']
        late_experiences = late_data[-1]['total_experiences']
        
        experience_growth = late_experiences / early_experiences
        time_growth = late_avg_time / early_avg_time
        
        print(f"Scaling Analysis:")
        print(f"  - Early experiences: {early_experiences}")
        print(f"  - Late experiences: {late_experiences}")
        print(f"  - Experience growth: {experience_growth:.1f}x")
        print(f"  - Early avg time: {early_avg_time:.1f}ms")
        print(f"  - Late avg time: {late_avg_time:.1f}ms")
        print(f"  - Time growth: {time_growth:.1f}x")
        
        # Scaling assessment
        if time_growth < experience_growth:
            print(f"  ✅ SUB-LINEAR: Better than linear scaling")
        elif time_growth < experience_growth * experience_growth:
            print(f"  ✅ SUB-QUADRATIC: Better than quadratic scaling")
        else:
            print(f"  ❌ POOR SCALING: Worse than quadratic scaling")
    
    def generate_summary_report(self):
        """Generate final summary report."""
        print(f"\n🎯 VALIDATION SUMMARY REPORT")
        print("=" * 60)
        
        # Overall assessment
        final_data = self.performance_data[-1]
        
        print(f"Test Results:")
        print(f"  - Duration: {self.test_duration} seconds")
        print(f"  - Cycles completed: {self.cycle_count}")
        print(f"  - Experiences accumulated: {final_data['total_experiences']}")
        print(f"  - Final cycle time: {final_data['cycle_time_ms']:.1f}ms")
        print(f"  - Final working memory: {final_data['working_memory_size']}")
        
        # Key metrics
        avg_cycle_time = sum(d['cycle_time_ms'] for d in self.performance_data) / len(self.performance_data)
        final_connectivity = self.connectivity_data[-1] if self.connectivity_data else None
        
        print(f"\nKey Metrics:")
        print(f"  - Average cycle time: {avg_cycle_time:.1f}ms")
        print(f"  - Real-time performance: {'✅ YES' if avg_cycle_time < 100 else '❌ NO'}")
        
        if final_connectivity:
            print(f"  - Avg connections per experience: {final_connectivity['avg_connections_per_experience']:.1f}")
            print(f"  - Sparse connectivity: {'✅ YES' if final_connectivity['avg_connections_per_experience'] < 20 else '❌ NO'}")
        
        # Similarity learning
        if self.similarity_data:
            final_sim = self.similarity_data[-1]
            print(f"  - Similarity adaptations: {final_sim['adaptations_performed']}")
            print(f"  - Learning active: {'✅ YES' if final_sim['adaptations_performed'] > 0 else '⚠️  MANUAL'}")
        
        print(f"\n🏆 OVERALL ASSESSMENT:")
        print(f"  {'✅ PASS' if avg_cycle_time < 100 else '❌ FAIL'}: Real-time performance")
        print(f"  {'✅ PASS' if final_connectivity and final_connectivity['avg_connections_per_experience'] < 20 else '❌ FAIL'}: Sparse connectivity")
        print(f"  {'✅ PASS' if self.similarity_data and self.similarity_data[-1]['prediction_outcomes_tracked'] > 0 else '❌ FAIL'}: Similarity learning")
        
        # Save detailed results
        self.save_detailed_results()
    
    def save_detailed_results(self):
        """Save detailed results to JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"validation_results_{timestamp}.json"
        
        results = {
            'test_info': {
                'duration': self.test_duration,
                'cycles_completed': self.cycle_count,
                'timestamp': timestamp
            },
            'performance_data': self.performance_data,
            'similarity_data': self.similarity_data,
            'connectivity_data': self.connectivity_data
        }
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"📄 Detailed results saved to: {results_file}")

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully."""
    print("\n🛑 Received interrupt signal, stopping validation...")
    global validator
    if validator:
        validator.running = False

def main():
    """Run the 5-minute biological timescale validation."""
    global validator
    
    # Set up signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    
    print("🧪 5-MINUTE BIOLOGICAL TIMESCALE VALIDATION")
    print("=" * 60)
    print("Testing sparse connectivity and similarity learning at biological timescales...")
    print("Press Ctrl+C to stop early if needed.")
    
    # Create and run validator
    validator = BiologicalTimescaleValidator()
    validator.setup_brain()
    
    # Run validation
    elapsed_time = validator.run_validation()
    
    # Analyze results
    validator.analyze_results()
    
    print(f"\n🎉 VALIDATION COMPLETE!")
    print(f"Total runtime: {elapsed_time:.1f} seconds")

if __name__ == "__main__":
    main()