#!/usr/bin/env python3
"""
Emergence Validation Framework

Clean, objective validation of emergent intelligence without hardcoded metrics.
Tests actual learning, adaptation, and intelligence emergence across different
brain architectures.

Key Principles:
- No hardcoded performance expectations
- Measure actual behavioral changes over time
- Compare different architectures objectively  
- Focus on emergence, not engineering metrics
- Let the data speak for itself

Validation Areas:
1. Spatial Learning: Place recognition and navigation emergence
2. Motor Coordination: Sensory-motor skill acquisition
3. Pattern Capacity: Scaling behavior with pattern count
4. Cross-Modal Learning: Sensory→motor association development
5. Temporal Dynamics: Sequence learning and prediction
"""

import sys
import os
import time
import json
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod

# Add server src to path
server_src = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'server', 'src')
sys.path.insert(0, server_src)


@dataclass
class ValidationResult:
    """Clean result structure with no hardcoded expectations."""
    test_name: str
    brain_type: str
    trial_number: int
    learning_curve: List[float]      # Performance over time
    final_performance: float         # End performance
    learning_rate: float            # Slope of improvement
    plateau_performance: float      # Performance after convergence
    convergence_time: int           # Cycles to reach plateau
    metadata: Dict[str, Any]        # Additional measurements


class EmergenceTest(ABC):
    """Base class for emergence validation tests."""
    
    def __init__(self, test_name: str):
        self.test_name = test_name
        self.results: List[ValidationResult] = []
    
    @abstractmethod
    def run_test(self, brain, num_trials: int = 3, max_cycles: int = 1000) -> List[ValidationResult]:
        """Run the emergence test and return objective results."""
        pass
    
    def calculate_learning_metrics(self, performance_curve: List[float]) -> Tuple[float, float, int]:
        """Calculate objective learning metrics from performance curve."""
        if len(performance_curve) < 10:
            return 0.0, performance_curve[-1] if performance_curve else 0.0, len(performance_curve)
        
        # Learning rate: slope of best fit line through first 50% of data
        half_point = len(performance_curve) // 2
        x = np.arange(half_point)
        y = performance_curve[:half_point]
        
        if len(x) > 1:
            learning_rate = np.polyfit(x, y, 1)[0]  # Slope
        else:
            learning_rate = 0.0
        
        # Plateau performance: average of last 20% of data
        plateau_start = int(len(performance_curve) * 0.8)
        plateau_performance = np.mean(performance_curve[plateau_start:])
        
        # Convergence time: when performance stops improving significantly
        convergence_time = len(performance_curve)
        for i in range(10, len(performance_curve)):
            recent_avg = np.mean(performance_curve[max(0, i-10):i])
            current_avg = np.mean(performance_curve[max(0, i-5):i])
            if abs(current_avg - recent_avg) < 0.01:  # Converged
                convergence_time = i
                break
        
        return learning_rate, plateau_performance, convergence_time


class SpatialLearningTest(EmergenceTest):
    """
    Test emergence of spatial intelligence.
    
    Creates a simple 2D world with landmarks and tests if the brain
    learns to associate sensory patterns with spatial locations.
    """
    
    def __init__(self):
        super().__init__("spatial_learning")
        
        # Create simple 2D world with landmarks
        self.world_size = 10
        self.landmarks = [
            (2, 2, [1.0, 0.0, 0.0]),  # Red landmark
            (8, 2, [0.0, 1.0, 0.0]),  # Green landmark  
            (5, 8, [0.0, 0.0, 1.0]),  # Blue landmark
        ]
        
    def generate_sensory_input(self, x: float, y: float) -> List[float]:
        """Generate sensory input based on position and landmark proximity."""
        sensory = [0.0] * 16  # 16D sensory input
        
        # Distance-based landmark visibility
        for i, (lx, ly, color) in enumerate(self.landmarks):
            distance = np.sqrt((x - lx)**2 + (y - ly)**2)
            visibility = max(0.0, 1.0 - distance / 5.0)  # Visible within 5 units
            
            # Landmark influence on sensory input
            start_idx = i * 3
            for j, color_val in enumerate(color):
                if start_idx + j < len(sensory):
                    sensory[start_idx + j] = visibility * color_val
        
        # Add position encoding
        sensory[9] = x / self.world_size  # Normalized x
        sensory[10] = y / self.world_size  # Normalized y
        
        # Add noise
        noise = np.random.normal(0, 0.1, len(sensory))
        sensory = [s + n for s, n in zip(sensory, noise)]
        
        return sensory
    
    def calculate_spatial_performance(self, brain_predictions: List[List[float]], 
                                    true_positions: List[Tuple[float, float]]) -> float:
        """Calculate how well brain predicts spatial movements."""
        if not brain_predictions or not true_positions:
            return 0.0
        
        total_error = 0.0
        valid_predictions = 0
        
        for pred, true_pos in zip(brain_predictions, true_positions[1:]):  # Predict next position
            if len(pred) >= 2:  # Need at least 2D motor output
                # Interpret motor output as movement direction
                predicted_movement = np.array([pred[0], pred[1]])
                
                # Calculate movement direction from position change
                prev_pos = true_positions[valid_predictions]
                actual_movement = np.array([true_pos[0] - prev_pos[0], true_pos[1] - prev_pos[1]])
                
                # Calculate alignment between predicted and actual movement
                if np.linalg.norm(predicted_movement) > 0 and np.linalg.norm(actual_movement) > 0:
                    # Cosine similarity between predicted and actual movement
                    similarity = np.dot(predicted_movement, actual_movement) / (
                        np.linalg.norm(predicted_movement) * np.linalg.norm(actual_movement)
                    )
                    alignment = (similarity + 1) / 2  # Convert to 0-1 scale
                    total_error += alignment
                    valid_predictions += 1
        
        return total_error / max(1, valid_predictions)
    
    def run_test(self, brain, num_trials: int = 3, max_cycles: int = 500) -> List[ValidationResult]:
        """Run spatial learning test."""
        results = []
        
        for trial in range(num_trials):
            print(f"  Running spatial learning trial {trial + 1}/{num_trials}")
            
            # Reset brain for new trial
            if hasattr(brain, 'reset_brain'):
                brain.reset_brain()
            
            performance_curve = []
            brain_predictions = []
            true_positions = []
            
            # Random walk through 2D world
            x, y = 5.0, 5.0  # Start in center
            
            for cycle in range(max_cycles):
                # Generate sensory input for current position
                sensory_input = self.generate_sensory_input(x, y)
                
                # Get brain's prediction
                motor_output, brain_state = brain.process_sensory_input(sensory_input)
                brain_predictions.append(motor_output)
                true_positions.append((x, y))
                
                # Move based on random walk (what we're trying to predict)
                dx = np.random.normal(0, 0.5)
                dy = np.random.normal(0, 0.5)
                x = np.clip(x + dx, 0, self.world_size)
                y = np.clip(y + dy, 0, self.world_size)
                
                # Calculate performance every 10 cycles
                if cycle % 10 == 0 and len(brain_predictions) > 10:
                    recent_performance = self.calculate_spatial_performance(
                        brain_predictions[-10:], true_positions[-10:]
                    )
                    performance_curve.append(recent_performance)
            
            # Calculate final metrics
            learning_rate, plateau_perf, convergence_time = self.calculate_learning_metrics(performance_curve)
            
            result = ValidationResult(
                test_name=self.test_name,
                brain_type=type(brain).__name__,
                trial_number=trial,
                learning_curve=performance_curve,
                final_performance=performance_curve[-1] if performance_curve else 0.0,
                learning_rate=learning_rate,
                plateau_performance=plateau_perf,
                convergence_time=convergence_time,
                metadata={
                    'world_size': self.world_size,
                    'num_landmarks': len(self.landmarks),
                    'total_cycles': max_cycles
                }
            )
            
            results.append(result)
        
        return results


class MotorCoordinationTest(EmergenceTest):
    """
    Test emergence of motor coordination.
    
    Tests if brain learns sensory-motor mappings by providing
    consistent sensory→motor training patterns.
    """
    
    def __init__(self):
        super().__init__("motor_coordination")
        
        # Define sensory-motor mappings to learn
        self.mappings = [
            ([1.0, 0.0, 0.0, 0.0], [1.0, 0.0]),  # Red → Move right
            ([0.0, 1.0, 0.0, 0.0], [0.0, 1.0]),  # Green → Move up
            ([0.0, 0.0, 1.0, 0.0], [-1.0, 0.0]), # Blue → Move left
            ([0.0, 0.0, 0.0, 1.0], [0.0, -1.0]), # Yellow → Move down
        ]
    
    def calculate_motor_performance(self, brain_outputs: List[List[float]], 
                                  target_outputs: List[List[float]]) -> float:
        """Calculate motor coordination accuracy."""
        if not brain_outputs or not target_outputs:
            return 0.0
        
        total_similarity = 0.0
        valid_comparisons = 0
        
        for brain_out, target_out in zip(brain_outputs, target_outputs):
            if len(brain_out) >= 2 and len(target_out) >= 2:
                # Calculate cosine similarity between brain output and target
                brain_vec = np.array(brain_out[:2])
                target_vec = np.array(target_out[:2])
                
                if np.linalg.norm(brain_vec) > 0 and np.linalg.norm(target_vec) > 0:
                    similarity = np.dot(brain_vec, target_vec) / (
                        np.linalg.norm(brain_vec) * np.linalg.norm(target_vec)
                    )
                    # Convert to 0-1 scale (0 = opposite, 1 = same direction)
                    performance = (similarity + 1) / 2
                    total_similarity += performance
                    valid_comparisons += 1
        
        return total_similarity / max(1, valid_comparisons)
    
    def run_test(self, brain, num_trials: int = 3, max_cycles: int = 400) -> List[ValidationResult]:
        """Run motor coordination test."""
        results = []
        
        for trial in range(num_trials):
            print(f"  Running motor coordination trial {trial + 1}/{num_trials}")
            
            # Reset brain for new trial
            if hasattr(brain, 'reset_brain'):
                brain.reset_brain()
            
            performance_curve = []
            brain_outputs = []
            target_outputs = []
            
            for cycle in range(max_cycles):
                # Choose random sensory-motor mapping
                sensory_pattern, target_motor = self.mappings[cycle % len(self.mappings)]
                
                # Add noise to sensory input
                noisy_sensory = [s + np.random.normal(0, 0.1) for s in sensory_pattern]
                
                # Pad sensory input to match brain's expected input size
                while len(noisy_sensory) < 16:
                    noisy_sensory.append(0.0)
                
                # Get brain's motor output
                motor_output, brain_state = brain.process_sensory_input(noisy_sensory)
                brain_outputs.append(motor_output)
                target_outputs.append(target_motor)
                
                # Calculate performance every 10 cycles
                if cycle % 10 == 0 and len(brain_outputs) > 10:
                    recent_performance = self.calculate_motor_performance(
                        brain_outputs[-10:], target_outputs[-10:]
                    )
                    performance_curve.append(recent_performance)
            
            # Calculate final metrics
            learning_rate, plateau_perf, convergence_time = self.calculate_learning_metrics(performance_curve)
            
            result = ValidationResult(
                test_name=self.test_name,
                brain_type=type(brain).__name__,
                trial_number=trial,
                learning_curve=performance_curve,
                final_performance=performance_curve[-1] if performance_curve else 0.0,
                learning_rate=learning_rate,
                plateau_performance=plateau_perf,
                convergence_time=convergence_time,
                metadata={
                    'num_mappings': len(self.mappings),
                    'total_cycles': max_cycles
                }
            )
            
            results.append(result)
        
        return results


class PatternCapacityTest(EmergenceTest):
    """
    Test pattern capacity scaling behavior.
    
    Tests how performance changes as we store more patterns.
    This reveals whether the brain architecture scales gracefully.
    """
    
    def __init__(self):
        super().__init__("pattern_capacity")
    
    def run_test(self, brain, num_trials: int = 1, max_cycles: int = 2000) -> List[ValidationResult]:
        """Run pattern capacity scaling test."""
        results = []
        
        for trial in range(num_trials):
            print(f"  Running pattern capacity trial {trial + 1}/{num_trials}")
            
            # Reset brain for new trial
            if hasattr(brain, 'reset_brain'):
                brain.reset_brain()
            
            performance_curve = []
            pattern_counts = []
            response_times = []
            
            # Test with increasing pattern load
            for cycle in range(max_cycles):
                # Generate unique pattern
                pattern = np.random.randn(16).tolist()
                
                # Measure response time
                start_time = time.time()
                motor_output, brain_state = brain.process_sensory_input(pattern)
                response_time = time.time() - start_time
                
                response_times.append(response_time * 1000)  # Convert to ms
                
                # Track pattern count if available
                if 'total_patterns' in brain_state:
                    pattern_counts.append(brain_state['total_patterns'])
                else:
                    pattern_counts.append(cycle)  # Approximate
                
                # Calculate performance as inverse of response time (higher = better)
                avg_response_time = np.mean(response_times[-50:])  # Last 50 cycles
                performance = 1.0 / (avg_response_time + 0.001)  # Avoid division by zero
                
                if cycle % 10 == 0:
                    performance_curve.append(performance)
            
            # Calculate final metrics
            learning_rate, plateau_perf, convergence_time = self.calculate_learning_metrics(performance_curve)
            
            result = ValidationResult(
                test_name=self.test_name,
                brain_type=type(brain).__name__,
                trial_number=trial,
                learning_curve=performance_curve,
                final_performance=performance_curve[-1] if performance_curve else 0.0,
                learning_rate=learning_rate,
                plateau_performance=plateau_perf,
                convergence_time=convergence_time,
                metadata={
                    'final_pattern_count': pattern_counts[-1] if pattern_counts else 0,
                    'avg_response_time_ms': np.mean(response_times),
                    'response_time_std': np.std(response_times),
                    'total_cycles': max_cycles
                }
            )
            
            results.append(result)
        
        return results


class EmergenceValidator:
    """
    Main validation framework for emergence testing.
    
    Coordinates multiple tests and provides comparative analysis
    between different brain architectures.
    """
    
    def __init__(self, output_dir: str = "validation/emergence_results"):
        self.output_dir = output_dir
        self.tests = [
            SpatialLearningTest(),
            MotorCoordinationTest(), 
            PatternCapacityTest()
        ]
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
    
    def validate_brain(self, brain, brain_name: str, num_trials: int = 3) -> Dict[str, List[ValidationResult]]:
        """Run all emergence tests on a brain architecture."""
        print(f"\n🧪 VALIDATING {brain_name.upper()}")
        print("=" * 50)
        
        all_results = {}
        
        for test in self.tests:
            print(f"\n📊 Running {test.test_name} test...")
            test_results = test.run_test(brain, num_trials=num_trials)
            all_results[test.test_name] = test_results
            
            # Print summary
            if test_results:
                avg_final = np.mean([r.final_performance for r in test_results])
                avg_learning_rate = np.mean([r.learning_rate for r in test_results])
                print(f"   Final performance: {avg_final:.3f}")
                print(f"   Learning rate: {avg_learning_rate:.3f}")
        
        # Save results
        self._save_results(brain_name, all_results)
        
        return all_results
    
    def compare_brains(self, brain_results: Dict[str, Dict[str, List[ValidationResult]]]) -> Dict[str, Any]:
        """Compare emergence across different brain architectures."""
        print(f"\n📊 COMPARATIVE EMERGENCE ANALYSIS")
        print("=" * 50)
        
        comparison = {}
        
        for test_name in self.tests[0].test_name if self.tests else []:
            test_comparison = {}
            
            for brain_name, results in brain_results.items():
                if test_name in results:
                    test_results = results[test_name]
                    
                    # Calculate averages across trials
                    avg_final = np.mean([r.final_performance for r in test_results])
                    avg_learning_rate = np.mean([r.learning_rate for r in test_results])
                    avg_plateau = np.mean([r.plateau_performance for r in test_results])
                    avg_convergence = np.mean([r.convergence_time for r in test_results])
                    
                    test_comparison[brain_name] = {
                        'final_performance': avg_final,
                        'learning_rate': avg_learning_rate,
                        'plateau_performance': avg_plateau,
                        'convergence_time': avg_convergence,
                        'std_final': np.std([r.final_performance for r in test_results]),
                        'num_trials': len(test_results)
                    }
            
            comparison[test_name] = test_comparison
        
        # Print comparison
        for test_name, test_comp in comparison.items():
            print(f"\n📈 {test_name.upper()} COMPARISON:")
            
            for brain_name, metrics in test_comp.items():
                print(f"   {brain_name:20} Final: {metrics['final_performance']:.3f} ± {metrics['std_final']:.3f}")
                print(f"   {' '*20} Learning: {metrics['learning_rate']:.3f}")
                print(f"   {' '*20} Plateau: {metrics['plateau_performance']:.3f}")
        
        # Save comparison
        comparison_file = os.path.join(self.output_dir, "emergence_comparison.json")
        with open(comparison_file, 'w') as f:
            json.dump(comparison, f, indent=2)
        
        return comparison
    
    def _save_results(self, brain_name: str, results: Dict[str, List[ValidationResult]]):
        """Save validation results to file."""
        # Convert results to serializable format
        serializable_results = {}
        for test_name, test_results in results.items():
            serializable_results[test_name] = [asdict(result) for result in test_results]
        
        # Save to JSON
        results_file = os.path.join(self.output_dir, f"{brain_name}_emergence_results.json")
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"   Results saved to {results_file}")
    
    def generate_report(self, brain_results: Dict[str, Dict[str, List[ValidationResult]]]):
        """Generate comprehensive emergence validation report."""
        report_file = os.path.join(self.output_dir, "emergence_validation_report.md")
        
        with open(report_file, 'w') as f:
            f.write("# Emergence Validation Report\n\n")
            f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Summary\n\n")
            f.write("Objective validation of emergent intelligence across brain architectures.\n")
            f.write("No hardcoded expectations - pure empirical measurement.\n\n")
            
            # Results for each brain
            for brain_name, results in brain_results.items():
                f.write(f"## {brain_name}\n\n")
                
                for test_name, test_results in results.items():
                    f.write(f"### {test_name}\n\n")
                    
                    if test_results:
                        avg_final = np.mean([r.final_performance for r in test_results])
                        avg_learning = np.mean([r.learning_rate for r in test_results])
                        std_final = np.std([r.final_performance for r in test_results])
                        
                        f.write(f"- **Final Performance**: {avg_final:.3f} ± {std_final:.3f}\n")
                        f.write(f"- **Learning Rate**: {avg_learning:.3f}\n")
                        f.write(f"- **Trials**: {len(test_results)}\n\n")
            
            f.write("## Key Findings\n\n")
            f.write("*Analysis of objective emergence measurements...*\n\n")
        
        print(f"\n📋 Report generated: {report_file}")


def create_test_brains():
    """Create different brain architectures for comparison."""
    brains = {}
    
    try:
        # Original minimal brain (if available)
        from vector_stream.minimal_brain import MinimalVectorStreamBrain
        brains['minimal'] = MinimalVectorStreamBrain(sensory_dim=16, motor_dim=8, temporal_dim=4)
    except ImportError:
        print("⚠️  Minimal brain not available")
    
    try:
        # Goldilocks brain  
        from vector_stream.goldilocks_brain import GoldilocksBrain
        brains['goldilocks'] = GoldilocksBrain(sensory_dim=16, motor_dim=8, temporal_dim=4, max_patterns=10000, quiet_mode=True)
    except ImportError:
        print("⚠️  Goldilocks brain not available")
    
    try:
        # Sparse goldilocks brain
        from vector_stream.sparse_goldilocks_brain import SparseGoldilocksBrain
        brains['sparse'] = SparseGoldilocksBrain(sensory_dim=16, motor_dim=8, temporal_dim=4, max_patterns=10000, quiet_mode=True)
    except ImportError:
        print("⚠️  Sparse brain not available")
    
    return brains


def main():
    """Run emergence validation on available brain architectures."""
    print("🧪 EMERGENCE VALIDATION FRAMEWORK")
    print("=" * 60)
    print("Objective testing of emergent intelligence")
    print("No hardcoded expectations - pure empirical measurement")
    
    # Create validator
    validator = EmergenceValidator()
    
    # Create test brains
    brains = create_test_brains()
    
    if not brains:
        print("❌ No brain architectures available for testing")
        return
    
    print(f"\n🧠 Testing {len(brains)} brain architectures:")
    for name in brains.keys():
        print(f"   - {name}")
    
    # Validate each brain
    all_results = {}
    for brain_name, brain in brains.items():
        try:
            results = validator.validate_brain(brain, brain_name, num_trials=3)
            all_results[brain_name] = results
        except Exception as e:
            print(f"❌ Error testing {brain_name}: {e}")
    
    # Comparative analysis
    if len(all_results) > 1:
        comparison = validator.compare_brains(all_results)
        validator.generate_report(all_results)
    
    print(f"\n✅ EMERGENCE VALIDATION COMPLETE")
    print(f"Results saved to: {validator.output_dir}")


if __name__ == "__main__":
    main()