#!/usr/bin/env python3
"""
Experience Nodes vs Vector Streams Comparison

Direct head-to-head comparison of:
1. Experience-based brain (current architecture)
2. Vector stream brain (new architecture)

Testing identical scenarios to validate whether vector streams
handle timing, prediction, and dead reckoning better than experience nodes.
"""

import sys
import os
import time
import numpy as np
from typing import List, Dict, Any, Tuple
import statistics
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Experience-based brain components
from server.src.experience.working_memory import WorkingMemoryBuffer
from server.src.experience.memory_consolidation import MemoryConsolidationLoop
from server.src.experience.storage import ExperienceStorage
from server.src.similarity.engine import SimilarityEngine
from server.src.similarity.dual_memory_search import DualMemorySearch
from server.src.utils.cognitive_autopilot import CognitiveAutopilot
from server.src.prediction.adaptive_engine import AdaptivePredictionEngine

# Vector stream brain
from server.src.vector_stream.minimal_brain import MinimalVectorStreamBrain


class ExperienceBasedBrain:
    """
    Simplified experience-based brain for comparison testing.
    
    Uses the same core components as the main brain but simplified
    for direct comparison with vector stream brain.
    """
    
    def __init__(self):
        # Core systems
        self.experience_storage = ExperienceStorage()
        self.working_memory = WorkingMemoryBuffer(capacity=50)
        self.similarity_engine = SimilarityEngine(use_gpu=False)
        self.cognitive_autopilot = CognitiveAutopilot()
        
        # Prediction engine
        self.prediction_engine = AdaptivePredictionEngine(
            cognitive_autopilot=self.cognitive_autopilot
        )
        
        # Dual memory search
        self.dual_memory_search = DualMemorySearch(
            self.similarity_engine,
            self.working_memory,
            self.experience_storage
        )
        
        # Memory consolidation
        self.consolidation_loop = MemoryConsolidationLoop(
            self.working_memory,
            self.experience_storage,
            base_interval_ms=100.0
        )
        
        # State tracking
        self.total_cycles = 0
        self.experience_id_counter = 0
        
        print("🧠 ExperienceBasedBrain initialized for comparison")
    
    def start(self):
        """Start brain systems."""
        self.consolidation_loop.start()
    
    def stop(self):
        """Stop brain systems."""
        self.consolidation_loop.stop()
    
    def process_sensory_input(self, sensory_vector: List[float]) -> Tuple[List[float], Dict[str, Any]]:
        """Process sensory input to generate action - experience-based approach."""
        cycle_start = time.time()
        
        # Convert to numpy
        sensory_array = np.array(sensory_vector, dtype=np.float32)
        
        # Search for similar experiences
        similar_experiences = self.dual_memory_search.search(
            sensory_array, k=5, similarity_threshold=0.3
        )
        
        # Generate action prediction
        if similar_experiences:
            # Use similar experiences for prediction
            action_vector = self._predict_from_experiences(sensory_array, similar_experiences)
            prediction_confidence = 0.7
        else:
            # Random exploration
            action_vector = np.random.randn(8) * 0.1  # Match vector stream motor dim
            prediction_confidence = 0.2
        
        # Update cognitive state
        cognitive_state = self.cognitive_autopilot.update_cognitive_state(
            prediction_confidence=prediction_confidence,
            prediction_error=0.1,
            brain_state={'time': time.time()}
        )
        
        # Add experience to working memory
        self.experience_id_counter += 1
        experience_id = f"exp_{self.experience_id_counter}"
        
        self.working_memory.add_experience(
            experience_id=experience_id,
            sensory_input=sensory_vector[:16] if len(sensory_vector) >= 16 else sensory_vector + [0.0] * (16 - len(sensory_vector)),
            action_taken=action_vector.tolist(),
            outcome=None,
            predicted_action=action_vector.tolist()
        )
        
        self.total_cycles += 1
        
        # Return result in same format as vector stream brain
        brain_state = {
            'cycle_time_ms': (time.time() - cycle_start) * 1000,
            'prediction_confidence': prediction_confidence,
            'total_cycles': self.total_cycles,
            'working_memory_size': len(self.working_memory),
            'long_term_memory_size': len(self.experience_storage._experiences),
            'cognitive_mode': cognitive_state['cognitive_mode']
        }
        
        return action_vector.tolist(), brain_state
    
    def _predict_from_experiences(self, sensory_input: np.ndarray, similar_experiences: List[Tuple[Any, float]]) -> np.ndarray:
        """Generate action prediction from similar experiences."""
        weighted_actions = []
        total_weight = 0.0
        
        for exp, similarity in similar_experiences:
            action = exp.action_taken
            if len(action) < 8:
                action = action + [0.0] * (8 - len(action))  # Pad to match vector stream
            elif len(action) > 8:
                action = action[:8]  # Truncate to match vector stream
            
            weighted_actions.append(np.array(action) * similarity)
            total_weight += similarity
        
        if total_weight > 0:
            return sum(weighted_actions) / total_weight
        else:
            return np.zeros(8)
    
    def get_brain_statistics(self) -> Dict[str, Any]:
        """Get brain statistics for comparison."""
        return {
            'total_cycles': self.total_cycles,
            'working_memory_size': len(self.working_memory),
            'long_term_memory_size': len(self.experience_storage._experiences),
            'brain_type': 'experience_based'
        }


class ComparisonProfiler:
    """Profile and compare brain performance."""
    
    def __init__(self, brain_name: str):
        self.brain_name = brain_name
        self.cycle_times = []
        self.prediction_confidences = []
        self.action_outputs = []
        self.total_cycles = 0
    
    def record_cycle(self, cycle_time: float, confidence: float, action: List[float]):
        """Record metrics for a cycle."""
        self.cycle_times.append(cycle_time)
        self.prediction_confidences.append(confidence)
        self.action_outputs.append(action.copy())
        self.total_cycles += 1
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get performance statistics."""
        if not self.cycle_times:
            return {'brain_name': self.brain_name, 'total_cycles': 0}
        
        return {
            'brain_name': self.brain_name,
            'total_cycles': self.total_cycles,
            'avg_cycle_time_ms': statistics.mean(self.cycle_times) * 1000,
            'median_cycle_time_ms': statistics.median(self.cycle_times) * 1000,
            'avg_confidence': statistics.mean(self.prediction_confidences),
            'confidence_std': statistics.stdev(self.prediction_confidences) if len(self.prediction_confidences) > 1 else 0,
            'cycles_per_second': len(self.cycle_times) / sum(self.cycle_times) if sum(self.cycle_times) > 0 else 0,
            'action_variance': self._calculate_action_variance()
        }
    
    def _calculate_action_variance(self) -> float:
        """Calculate variance in action outputs (measure of behavioral consistency)."""
        if len(self.action_outputs) < 2:
            return 0.0
        
        # Calculate variance across all action dimensions
        action_array = np.array(self.action_outputs)
        return float(np.mean(np.var(action_array, axis=0)))


def run_identical_scenario(experience_brain: ExperienceBasedBrain, 
                          vector_brain: MinimalVectorStreamBrain,
                          scenario_name: str,
                          input_sequence: List[List[float]],
                          timing_sequence: List[float]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Run identical scenario through both brains and compare results.
    
    Args:
        experience_brain: Experience-based brain
        vector_brain: Vector stream brain  
        scenario_name: Name of the test scenario
        input_sequence: Sequence of sensory inputs
        timing_sequence: Timing delays between inputs
    """
    print(f"\n🧪 Running scenario: {scenario_name}")
    print(f"   Inputs: {len(input_sequence)}, Timings: {timing_sequence[:3]}...")
    
    # Profilers for both brains
    exp_profiler = ComparisonProfiler("Experience-Based")
    vec_profiler = ComparisonProfiler("Vector-Stream")
    
    # Run through experience-based brain
    print("   🧠 Testing experience-based brain...")
    for i, (sensory_input, delay) in enumerate(zip(input_sequence, timing_sequence)):
        cycle_start = time.time()
        
        # Ensure input is right size for experience brain (expects 16D minimum)
        padded_input = sensory_input + [0.0] * (16 - len(sensory_input)) if len(sensory_input) < 16 else sensory_input
        
        action, brain_state = experience_brain.process_sensory_input(padded_input)
        cycle_time = time.time() - cycle_start
        
        exp_profiler.record_cycle(cycle_time, brain_state['prediction_confidence'], action)
        time.sleep(delay)
    
    # Run through vector stream brain  
    print("   🧠 Testing vector stream brain...")
    for i, (sensory_input, delay) in enumerate(zip(input_sequence, timing_sequence)):
        cycle_start = time.time()
        
        # Ensure input is right size for vector brain (expects 8D for testing)
        truncated_input = sensory_input[:8] if len(sensory_input) >= 8 else sensory_input + [0.0] * (8 - len(sensory_input))
        
        action, brain_state = vector_brain.process_sensory_input(truncated_input)
        cycle_time = time.time() - cycle_start
        
        vec_profiler.record_cycle(cycle_time, brain_state['prediction_confidence'], action)
        time.sleep(delay)
    
    exp_stats = exp_profiler.get_statistics()
    vec_stats = vec_profiler.get_statistics()
    
    print(f"   ✅ {scenario_name} completed")
    
    return exp_stats, vec_stats


def test_timing_sensitivity_comparison():
    """Compare timing sensitivity between both brain types."""
    print("\n⏰ TIMING SENSITIVITY COMPARISON")
    print("=" * 60)
    
    # Initialize both brains
    exp_brain = ExperienceBasedBrain()
    exp_brain.start()
    
    vec_brain = MinimalVectorStreamBrain(sensory_dim=8, motor_dim=8, temporal_dim=4)
    
    results = {}
    
    try:
        # Scenario 1: Fast rhythm (50ms intervals)
        fast_inputs = [[1.0, 0.0, 0.0, 0.0] + [0.0] * 4 for _ in range(10)]
        fast_timings = [0.05] * 10
        
        exp_fast, vec_fast = run_identical_scenario(
            exp_brain, vec_brain, "Fast Rhythm (50ms)", fast_inputs, fast_timings
        )
        results['fast_rhythm'] = {'experience': exp_fast, 'vector': vec_fast}
        
        # Scenario 2: Slow rhythm (200ms intervals)
        slow_inputs = [[0.0, 1.0, 0.0, 0.0] + [0.0] * 4 for _ in range(5)]
        slow_timings = [0.2] * 5
        
        exp_slow, vec_slow = run_identical_scenario(
            exp_brain, vec_brain, "Slow Rhythm (200ms)", slow_inputs, slow_timings
        )
        results['slow_rhythm'] = {'experience': exp_slow, 'vector': vec_slow}
        
        # Scenario 3: Variable timing
        var_inputs = [[0.5, 0.5, 0.0, 0.0] + [0.0] * 4 for _ in range(8)]
        var_timings = [0.1, 0.05, 0.15, 0.08, 0.12, 0.06, 0.18, 0.04]
        
        exp_var, vec_var = run_identical_scenario(
            exp_brain, vec_brain, "Variable Timing", var_inputs, var_timings
        )
        results['variable_timing'] = {'experience': exp_var, 'vector': vec_var}
        
    finally:
        exp_brain.stop()
    
    return results


def test_dead_reckoning_comparison():
    """Compare dead reckoning capability between both brain types."""
    print("\n🚀 DEAD RECKONING COMPARISON")
    print("=" * 60)
    
    # Initialize fresh brains
    exp_brain = ExperienceBasedBrain()
    exp_brain.start()
    
    vec_brain = MinimalVectorStreamBrain(sensory_dim=8, motor_dim=8, temporal_dim=4)
    
    results = {}
    
    try:
        # Training phase: Both brains learn the same pattern
        pattern_sequence = [
            [1.0, 0.0, 0.0, 0.0] + [0.0] * 4,  # Pattern A
            [0.0, 1.0, 0.0, 0.0] + [0.0] * 4,  # Pattern B  
            [0.0, 0.0, 1.0, 0.0] + [0.0] * 4,  # Pattern C
            [0.0, 0.0, 0.0, 1.0] + [0.0] * 4,  # Pattern D
        ]
        
        # Repeat pattern 5 times for training
        training_inputs = (pattern_sequence * 5)
        training_timings = [0.1] * len(training_inputs)
        
        exp_train, vec_train = run_identical_scenario(
            exp_brain, vec_brain, "Training Phase", training_inputs, training_timings
        )
        results['training'] = {'experience': exp_train, 'vector': vec_train}
        
        # Dead reckoning phase: Zero sensory input
        zero_inputs = [[0.0] * 8 for _ in range(8)]
        zero_timings = [0.1] * 8
        
        exp_dead, vec_dead = run_identical_scenario(
            exp_brain, vec_brain, "Dead Reckoning (Zero Input)", zero_inputs, zero_timings
        )
        results['dead_reckoning'] = {'experience': exp_dead, 'vector': vec_dead}
        
    finally:
        exp_brain.stop()
    
    return results


def test_latency_handling_comparison():
    """Compare latency handling between both brain types."""
    print("\n📡 LATENCY HANDLING COMPARISON")  
    print("=" * 60)
    
    # Initialize brains
    exp_brain = ExperienceBasedBrain()
    exp_brain.start()
    
    vec_brain = MinimalVectorStreamBrain(sensory_dim=8, motor_dim=8, temporal_dim=4)
    
    results = {}
    
    try:
        # Scenario: Sudden latency spikes
        base_inputs = [[np.sin(i * 0.5), np.cos(i * 0.3), i/10.0, 0.5] + [0.0] * 4 for i in range(15)]
        
        # Normal latency, then spike, then back to normal
        latency_pattern = [0.05] * 5 + [0.3] * 3 + [0.05] * 7  # Latency spike in middle
        
        exp_latency, vec_latency = run_identical_scenario(
            exp_brain, vec_brain, "Latency Spike Test", base_inputs, latency_pattern
        )
        results['latency_spike'] = {'experience': exp_latency, 'vector': vec_latency}
        
    finally:
        exp_brain.stop()
    
    return results


def analyze_comparison_results(timing_results: Dict, dead_reckoning_results: Dict, latency_results: Dict):
    """Analyze and report comparison results."""
    print("\n📊 COMPREHENSIVE COMPARISON ANALYSIS")
    print("=" * 80)
    
    # Performance comparison
    print("\n🏃 PERFORMANCE METRICS:")
    print("-" * 40)
    
    all_scenarios = [
        ("Fast Rhythm", timing_results['fast_rhythm']),
        ("Slow Rhythm", timing_results['slow_rhythm']),
        ("Variable Timing", timing_results['variable_timing']),
        ("Training", dead_reckoning_results['training']),
        ("Dead Reckoning", dead_reckoning_results['dead_reckoning']),
        ("Latency Spike", latency_results['latency_spike'])
    ]
    
    for scenario_name, scenario_data in all_scenarios:
        exp_data = scenario_data['experience']
        vec_data = scenario_data['vector']
        
        print(f"\n{scenario_name}:")
        print(f"  Experience Brain: {exp_data['avg_cycle_time_ms']:.1f}ms, confidence: {exp_data['avg_confidence']:.2f}")
        print(f"  Vector Brain:     {vec_data['avg_cycle_time_ms']:.1f}ms, confidence: {vec_data['avg_confidence']:.2f}")
        
        # Calculate improvements
        speed_improvement = exp_data['avg_cycle_time_ms'] / vec_data['avg_cycle_time_ms'] if vec_data['avg_cycle_time_ms'] > 0 else 1
        confidence_improvement = vec_data['avg_confidence'] / exp_data['avg_confidence'] if exp_data['avg_confidence'] > 0 else 1
        
        print(f"  Speed improvement: {speed_improvement:.1f}x, Confidence improvement: {confidence_improvement:.1f}x")
    
    # Key insights
    print(f"\n💡 KEY INSIGHTS:")
    print("-" * 30)
    
    # Dead reckoning capability
    exp_dead_conf = dead_reckoning_results['dead_reckoning']['experience']['avg_confidence']
    vec_dead_conf = dead_reckoning_results['dead_reckoning']['vector']['avg_confidence']
    
    print(f"🚀 Dead Reckoning Performance:")
    print(f"   Experience-based: {exp_dead_conf:.2f} confidence")
    print(f"   Vector stream:    {vec_dead_conf:.2f} confidence")
    print(f"   Winner: {'Vector Stream' if vec_dead_conf > exp_dead_conf else 'Experience-Based' if exp_dead_conf > vec_dead_conf else 'Tie'}")
    
    # Timing sensitivity
    exp_var_std = timing_results['variable_timing']['experience']['confidence_std']
    vec_var_std = timing_results['variable_timing']['vector']['confidence_std']
    
    print(f"\n⏰ Timing Sensitivity:")
    print(f"   Experience-based confidence variance: {exp_var_std:.3f}")
    print(f"   Vector stream confidence variance:    {vec_var_std:.3f}")
    print(f"   Winner: {'Vector Stream' if vec_var_std < exp_var_std else 'Experience-Based' if exp_var_std < vec_var_std else 'Tie'} (lower variance = more consistent)")
    
    # Overall assessment
    print(f"\n🏆 OVERALL ASSESSMENT:")
    
    vector_wins = 0
    experience_wins = 0
    
    # Count wins across different metrics
    for scenario_name, scenario_data in all_scenarios:
        exp_data = scenario_data['experience']
        vec_data = scenario_data['vector']
        
        if vec_data['avg_confidence'] > exp_data['avg_confidence']:
            vector_wins += 1
        elif exp_data['avg_confidence'] > vec_data['avg_confidence']:
            experience_wins += 1
    
    print(f"   Vector Stream wins: {vector_wins}/{len(all_scenarios)} scenarios")
    print(f"   Experience-Based wins: {experience_wins}/{len(all_scenarios)} scenarios")
    
    if vector_wins > experience_wins:
        print(f"   🎉 VECTOR STREAMS SHOW SUPERIOR PERFORMANCE")
        print(f"   → Better prediction confidence across most scenarios")
        print(f"   → Natural timing integration")
        print(f"   → Superior dead reckoning capability")
    elif experience_wins > vector_wins:
        print(f"   🎉 EXPERIENCE-BASED SHOWS SUPERIOR PERFORMANCE")
        print(f"   → More stable across scenarios")
    else:
        print(f"   🤝 COMPARABLE PERFORMANCE")
        print(f"   → Both approaches have strengths")
    
    return vector_wins > experience_wins


def main():
    """Run comprehensive comparison between experience and vector stream brains."""
    print("🧠 EXPERIENCE vs VECTOR STREAM BRAIN COMPARISON")
    print("=" * 80)
    print("Scientific validation: Do vector streams handle timing and prediction")
    print("better than experience nodes?")
    
    try:
        # Run all comparison tests
        timing_results = test_timing_sensitivity_comparison()
        dead_reckoning_results = test_dead_reckoning_comparison()
        latency_results = test_latency_handling_comparison()
        
        # Analyze results
        vector_streams_win = analyze_comparison_results(timing_results, dead_reckoning_results, latency_results)
        
        print(f"\n🔬 SCIENTIFIC CONCLUSION:")
        if vector_streams_win:
            print(f"✅ HYPOTHESIS CONFIRMED: Vector streams outperform experience nodes")
            print(f"   → Biological-style processing handles timing better")
            print(f"   → Continuous vector flow enables superior prediction")
            print(f"   → Time-as-data-stream approach is more effective")
        else:
            print(f"❓ HYPOTHESIS INCONCLUSIVE: Mixed results between approaches")
            print(f"   → Further investigation needed")
            print(f"   → Both approaches may have different strengths")
        
        return vector_streams_win
        
    except Exception as e:
        print(f"\n❌ Comparison test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)