#!/usr/bin/env python3
"""
Brain Performance Profiler

Investigates where the 177ms cycle time comes from:
- Is it fundamental computational complexity of true brain function?
- Or suboptimal implementation that can be optimized without losing intelligence?

This tool profiles each component of the brain cycle to identify bottlenecks.
"""

import sys
import os
import time
import cProfile
import pstats
import io
from typing import Dict, List, Any, Tuple
from contextlib import contextmanager

# Set up path to access brain modules
brain_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(brain_root, 'server', 'src'))
sys.path.append(os.path.join(brain_root, 'server'))

from src.brain import MinimalBrain

@contextmanager
def time_component(component_name: str, timings: Dict[str, List[float]]):
    """Context manager to time brain components."""
    start_time = time.time()
    try:
        yield
    finally:
        end_time = time.time()
        elapsed = (end_time - start_time) * 1000  # Convert to ms
        if component_name not in timings:
            timings[component_name] = []
        timings[component_name].append(elapsed)

class BrainPerformanceProfiler:
    """Detailed performance profiler for brain components."""
    
    def __init__(self):
        self.brain = None
        self.component_timings = {}
        self.cycle_data = []
        
    def setup_brain(self, num_initial_experiences: int = 50):
        """Setup brain with initial experiences for realistic profiling."""
        print("🧠 SETTING UP BRAIN FOR PERFORMANCE PROFILING")
        print("=" * 60)
        
        # Create brain with minimal logging to avoid measurement interference
        self.brain = MinimalBrain(
            enable_logging=False,
            enable_persistence=False,
            enable_phase2_adaptations=True
        )
        
        print(f"✅ Brain initialized")
        print(f"🔄 Adding {num_initial_experiences} initial experiences for realistic profiling...")
        
        # Add initial experiences to get realistic performance
        for i in range(num_initial_experiences):
            sensory = [0.1 + 0.01 * i, 0.2 + 0.01 * i, 0.3 + 0.01 * i, 0.4 + 0.01 * i]
            action = [0.5 + 0.01 * i, 0.6 + 0.01 * i, 0.7 + 0.01 * i, 0.8 + 0.01 * i]
            outcome = [0.9 + 0.01 * i, 0.8 + 0.01 * i, 0.7 + 0.01 * i, 0.6 + 0.01 * i]
            
            self.brain.store_experience(
                sensory_input=sensory,
                action_taken=action,
                outcome=outcome,
                predicted_action=action
            )
        
        print(f"✅ Added {len(self.brain.experience_storage._experiences)} experiences")
        print(f"🎯 Ready for performance profiling")
        
    def profile_single_cycle(self, cycle_id: int) -> Dict[str, Any]:
        """Profile a single brain cycle with detailed component timing."""
        
        # Generate test sensory input
        sensory_input = [0.5 + 0.01 * cycle_id, 0.4 + 0.01 * cycle_id, 0.6 + 0.01 * cycle_id, 0.3 + 0.01 * cycle_id]
        
        cycle_timings = {}
        cycle_start = time.time()
        
        # === BRAIN CYCLE PROFILING ===
        
        # 1. Activation System
        with time_component("activation_system", cycle_timings):
            if self.brain.use_utility_based_activation:
                self.brain._activate_by_utility(sensory_input)
            else:
                self.brain._activate_similar_experiences(sensory_input)
        
        # 2. Cognitive Autopilot Update
        with time_component("cognitive_autopilot", cycle_timings):
            last_confidence = getattr(self.brain, '_last_confidence', 0.5)
            prediction_error = 1.0 - last_confidence
            
            initial_brain_state = {
                'prediction_confidence': last_confidence,
                'num_experiences': len(self.brain.experience_storage._experiences)
            }
            
            autopilot_state = self.brain.cognitive_autopilot.update_cognitive_state(
                last_confidence, prediction_error, initial_brain_state
            )
        
        # 3. Prediction Engine
        with time_component("prediction_engine", cycle_timings):
            brain_state_for_prediction = {
                'prediction_confidence': last_confidence,
                'num_experiences': len(self.brain.experience_storage._experiences),
                'cognitive_autopilot': autopilot_state
            }
            
            predicted_action, confidence, prediction_details = self.brain.prediction_engine.predict_action(
                sensory_input,
                self.brain.similarity_engine,
                self.brain.activation_dynamics,
                self.brain.experience_storage._experiences,
                4,  # action_dimensions
                brain_state_for_prediction
            )
        
        # 4. Experience Storage
        with time_component("experience_storage", cycle_timings):
            world_response = [a * 0.9 + 0.05 for a in predicted_action]
            
            experience_id = self.brain.store_experience(
                sensory_input=sensory_input,
                action_taken=predicted_action,
                outcome=world_response,
                predicted_action=predicted_action
            )
        
        cycle_end = time.time()
        total_cycle_time = (cycle_end - cycle_start) * 1000  # Convert to ms
        
        # Detailed component analysis
        return {
            'cycle_id': cycle_id,
            'total_cycle_time_ms': total_cycle_time,
            'component_timings': {k: v[0] if v else 0 for k, v in cycle_timings.items()},
            'experiences_count': len(self.brain.experience_storage._experiences),
            'prediction_confidence': confidence,
            'prediction_details': prediction_details
        }
    
    def profile_detailed_components(self, cycle_id: int) -> Dict[str, Any]:
        """Profile detailed sub-components within major systems."""
        
        sensory_input = [0.5 + 0.01 * cycle_id, 0.4 + 0.01 * cycle_id, 0.6 + 0.01 * cycle_id, 0.3 + 0.01 * cycle_id]
        detailed_timings = {}
        
        # === DETAILED SIMILARITY ENGINE PROFILING ===
        
        # Get all experience vectors (this might be expensive)
        with time_component("similarity_get_vectors", detailed_timings):
            experience_vectors = []
            experience_ids = []
            for exp_id, exp in self.brain.experience_storage._experiences.items():
                experience_vectors.append(exp.get_context_vector())
                experience_ids.append(exp_id)
        
        # Similarity search
        with time_component("similarity_search", detailed_timings):
            similar_experiences = self.brain.similarity_engine.find_similar_experiences(
                sensory_input, experience_vectors, experience_ids,
                max_results=15, min_similarity=0.3
            )
        
        # === DETAILED ACTIVATION PROFILING ===
        
        if self.brain.use_utility_based_activation:
            with time_component("activation_utility_computation", detailed_timings):
                # This involves GPU operations and utility calculations
                self.brain.activation_dynamics.activate_by_prediction_utility(
                    sensory_input, self.brain.experience_storage._experiences, similar_experiences
                )
        
        # === DETAILED PREDICTION PROFILING ===
        
        # Pattern analysis (if enabled)
        with time_component("prediction_pattern_analysis", detailed_timings):
            if hasattr(self.brain.prediction_engine, 'pattern_analyzer'):
                # This might be doing complex pattern matching
                pass
        
        # === DETAILED STORAGE PROFILING ===
        
        # Similarity connection updates
        with time_component("storage_similarity_updates", detailed_timings):
            # This happens in store_experience - might be expensive
            pass
        
        return {
            'cycle_id': cycle_id,
            'detailed_timings': {k: v[0] if v else 0 for k, v in detailed_timings.items()},
            'similarity_results_count': len(similar_experiences),
            'experience_vectors_count': len(experience_vectors)
        }
    
    def run_profiling_session(self, num_cycles: int = 20) -> None:
        """Run comprehensive profiling session."""
        print(f"\n🔬 RUNNING COMPREHENSIVE PROFILING SESSION")
        print("=" * 60)
        print(f"📊 Profiling {num_cycles} cycles to identify bottlenecks")
        print(f"🎯 Goal: Distinguish fundamental complexity from implementation issues")
        
        # Profile regular cycles
        for cycle in range(num_cycles):
            cycle_data = self.profile_single_cycle(cycle)
            self.cycle_data.append(cycle_data)
            
            # Accumulate component timings
            for component, timing in cycle_data['component_timings'].items():
                if component not in self.component_timings:
                    self.component_timings[component] = []
                self.component_timings[component].append(timing)
            
            if cycle % 5 == 0:
                print(f"  Cycle {cycle:2d}: {cycle_data['total_cycle_time_ms']:6.1f}ms | "
                      f"Experiences: {cycle_data['experiences_count']:3d} | "
                      f"Confidence: {cycle_data['prediction_confidence']:.3f}")
        
        # Profile detailed components for last few cycles
        print(f"\n🔍 Detailed component profiling...")
        detailed_data = []
        for cycle in range(num_cycles - 3, num_cycles):
            detailed = self.profile_detailed_components(cycle)
            detailed_data.append(detailed)
        
        print(f"✅ Profiling session completed")
        
        # Analyze results
        self.analyze_profiling_results(detailed_data)
    
    def analyze_profiling_results(self, detailed_data: List[Dict]) -> None:
        """Analyze profiling results to identify bottlenecks."""
        print(f"\n📊 PERFORMANCE BOTTLENECK ANALYSIS")
        print("=" * 60)
        
        # Calculate average timings for each component
        avg_timings = {}
        for component, timings in self.component_timings.items():
            avg_timings[component] = sum(timings) / len(timings)
        
        # Sort components by average time
        sorted_components = sorted(avg_timings.items(), key=lambda x: x[1], reverse=True)
        
        print(f"⏱️  MAJOR COMPONENT PERFORMANCE BREAKDOWN:")
        total_accounted = 0
        for component, avg_time in sorted_components:
            percentage = (avg_time / sum(avg_timings.values())) * 100
            total_accounted += avg_time
            print(f"  {component:20s}: {avg_time:6.1f}ms ({percentage:5.1f}%)")
        
        print(f"  {'Total Accounted':20s}: {total_accounted:6.1f}ms")
        
        # Analyze scaling behavior
        self.analyze_scaling_behavior()
        
        # Identify optimization opportunities
        self.identify_optimization_opportunities(sorted_components, detailed_data)
        
        # Classify bottlenecks
        self.classify_bottlenecks(sorted_components)
    
    def analyze_scaling_behavior(self) -> None:
        """Analyze how performance scales with experience count."""
        print(f"\n📈 SCALING BEHAVIOR ANALYSIS:")
        
        if len(self.cycle_data) < 10:
            print("  ❌ Insufficient data for scaling analysis")
            return
        
        # Compare early vs late cycles
        early_cycles = self.cycle_data[:5]
        late_cycles = self.cycle_data[-5:]
        
        early_avg_time = sum(c['total_cycle_time_ms'] for c in early_cycles) / len(early_cycles)
        late_avg_time = sum(c['total_cycle_time_ms'] for c in late_cycles) / len(late_cycles)
        
        early_avg_exp = sum(c['experiences_count'] for c in early_cycles) / len(early_cycles)
        late_avg_exp = sum(c['experiences_count'] for c in late_cycles) / len(late_cycles)
        
        experience_growth = late_avg_exp / early_avg_exp
        time_growth = late_avg_time / early_avg_time
        
        print(f"  Early cycles: {early_avg_time:.1f}ms avg, {early_avg_exp:.0f} experiences")
        print(f"  Late cycles:  {late_avg_time:.1f}ms avg, {late_avg_exp:.0f} experiences")
        print(f"  Experience growth: {experience_growth:.2f}x")
        print(f"  Time growth: {time_growth:.2f}x")
        
        # Analyze scaling pattern
        if time_growth < 1.1:
            print(f"  ✅ CONSTANT TIME: Performance independent of experience count")
        elif time_growth < experience_growth:
            print(f"  ✅ SUB-LINEAR: Better than linear scaling")
        elif time_growth < experience_growth * 1.5:
            print(f"  ⚠️  LINEAR: Performance scales linearly with experiences")
        else:
            print(f"  ❌ SUPER-LINEAR: Performance scales worse than linearly")
    
    def identify_optimization_opportunities(self, sorted_components: List[Tuple[str, float]], 
                                          detailed_data: List[Dict]) -> None:
        """Identify specific optimization opportunities."""
        print(f"\n🔧 OPTIMIZATION OPPORTUNITIES:")
        
        # Check for expensive operations
        for component, avg_time in sorted_components:
            if avg_time > 20:  # Components taking >20ms
                print(f"\n  🎯 HIGH-IMPACT TARGET: {component} ({avg_time:.1f}ms)")
                
                if component == "experience_storage":
                    print(f"    - Potential Issue: O(n) operations in store_experience")
                    print(f"    - Solution: Batch operations, async storage")
                    print(f"    - Intelligence Impact: LOW (storage optimization)")
                
                elif component == "activation_system":
                    print(f"    - Potential Issue: Computing activations for all experiences")
                    print(f"    - Solution: Sparse activation updates, GPU batching")
                    print(f"    - Intelligence Impact: LOW (computational optimization)")
                
                elif component == "prediction_engine":
                    print(f"    - Potential Issue: Complex pattern analysis")
                    print(f"    - Solution: Simplified patterns, caching")
                    print(f"    - Intelligence Impact: MEDIUM (may affect prediction quality)")
                
                elif component == "cognitive_autopilot":
                    print(f"    - Potential Issue: Complex state management")
                    print(f"    - Solution: Simplified autopilot logic")
                    print(f"    - Intelligence Impact: LOW (meta-cognitive optimization)")
        
        # Check detailed timings
        if detailed_data:
            avg_detailed = {}
            for data in detailed_data:
                for component, timing in data['detailed_timings'].items():
                    if component not in avg_detailed:
                        avg_detailed[component] = []
                    avg_detailed[component].append(timing)
            
            print(f"\n  🔍 DETAILED BOTTLENECK ANALYSIS:")
            for component, timings in avg_detailed.items():
                if timings:
                    avg_time = sum(timings) / len(timings)
                    if avg_time > 5:  # Sub-components taking >5ms
                        print(f"    {component:25s}: {avg_time:6.1f}ms")
    
    def classify_bottlenecks(self, sorted_components: List[Tuple[str, float]]) -> None:
        """Classify bottlenecks as fundamental vs implementation issues."""
        print(f"\n🧠 BOTTLENECK CLASSIFICATION:")
        print("  (Fundamental = necessary for intelligence)")
        print("  (Implementation = can be optimized without losing intelligence)")
        
        total_time = sum(time for _, time in sorted_components)
        
        fundamental_time = 0
        implementation_time = 0
        
        for component, avg_time in sorted_components:
            percentage = (avg_time / total_time) * 100
            
            if component in ["prediction_engine", "similarity_search"]:
                category = "FUNDAMENTAL"
                fundamental_time += avg_time
                impact = "Critical for intelligence"
            elif component in ["activation_system", "cognitive_autopilot"]:
                category = "HYBRID"
                impact = "Important but optimizable"
            else:
                category = "IMPLEMENTATION"
                implementation_time += avg_time
                impact = "Can be optimized"
            
            print(f"  {component:20s}: {avg_time:6.1f}ms ({percentage:5.1f}%) - {category:13s} - {impact}")
        
        print(f"\n📊 OVERALL ASSESSMENT:")
        fundamental_pct = (fundamental_time / total_time) * 100
        implementation_pct = (implementation_time / total_time) * 100
        
        print(f"  Fundamental complexity: {fundamental_time:.1f}ms ({fundamental_pct:.1f}%)")
        print(f"  Implementation issues:  {implementation_time:.1f}ms ({implementation_pct:.1f}%)")
        
        if implementation_pct > 40:
            print(f"  ✅ GOOD NEWS: {implementation_pct:.1f}% can be optimized without losing intelligence")
        elif implementation_pct > 20:
            print(f"  ⚠️  MODERATE: {implementation_pct:.1f}% optimization potential")
        else:
            print(f"  ❌ FUNDAMENTAL: Most time is necessary computational complexity")
    
    def generate_optimization_roadmap(self) -> None:
        """Generate concrete optimization roadmap."""
        print(f"\n🗺️  OPTIMIZATION ROADMAP:")
        print("=" * 60)
        
        # Sort components by optimization potential
        avg_timings = {}
        for component, timings in self.component_timings.items():
            avg_timings[component] = sum(timings) / len(timings)
        
        sorted_components = sorted(avg_timings.items(), key=lambda x: x[1], reverse=True)
        
        print(f"PRIORITY 1 - HIGH IMPACT, LOW INTELLIGENCE RISK:")
        for component, avg_time in sorted_components:
            if avg_time > 20 and component in ["experience_storage", "cognitive_autopilot"]:
                print(f"  🎯 Optimize {component} ({avg_time:.1f}ms)")
                print(f"     - Use batch operations, async processing")
                print(f"     - Expected gain: 30-50% reduction")
                print(f"     - Intelligence impact: MINIMAL")
        
        print(f"\nPRIORITY 2 - MODERATE IMPACT, CAREFUL OPTIMIZATION:")
        for component, avg_time in sorted_components:
            if avg_time > 10 and component in ["activation_system", "prediction_engine"]:
                print(f"  ⚠️  Optimize {component} ({avg_time:.1f}ms)")
                print(f"     - Use GPU acceleration, sparse operations")
                print(f"     - Expected gain: 20-30% reduction")
                print(f"     - Intelligence impact: LOW-MODERATE")
        
        print(f"\nPRIORITY 3 - ALGORITHMIC IMPROVEMENTS:")
        print(f"  🧠 Implement PyTorch sparse tensors")
        print(f"  🧠 Add k-NN sparse connectivity")
        print(f"  🧠 Optimize similarity computation")
        print(f"     - Expected gain: 10-20% reduction")
        print(f"     - Intelligence impact: POTENTIALLY POSITIVE")

def main():
    """Run comprehensive brain performance profiling."""
    print("🔬 BRAIN PERFORMANCE PROFILER")
    print("=" * 60)
    print("Investigating whether 177ms cycle time is fundamental complexity")
    print("or suboptimal implementation that can be optimized safely.")
    
    profiler = BrainPerformanceProfiler()
    profiler.setup_brain(num_initial_experiences=50)
    profiler.run_profiling_session(num_cycles=15)
    profiler.generate_optimization_roadmap()
    
    print(f"\n🎯 CONCLUSION:")
    print("Use this analysis to optimize implementation issues while preserving")
    print("fundamental computational complexity necessary for intelligence.")

if __name__ == "__main__":
    main()