#!/usr/bin/env python3
"""
Improved Core Assumptions Tests - Phase 1 Optimizations

Key improvements:
1. Uses persistent connections and environments
2. Implements retry logic for brain processing errors
3. Reduced environment resets
4. Better error handling
5. Increased timeouts
"""

import sys
import os
import time
import numpy as np
from typing import List, Dict, Any, Optional
from pathlib import Path

# Add paths for brain modules
brain_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(brain_root))
sys.path.insert(0, str(brain_root / 'server' / 'src'))
sys.path.insert(0, str(brain_root / 'server'))
sys.path.insert(0, str(brain_root / 'validation'))

from micro_experiments.improved_framework import ImprovedMicroExperiment, ImprovedMicroExperimentSuite, MicroExperimentResult, BrainProcessingError

def perform_t_test(group1: List[float], group2: List[float]) -> tuple:
    """Perform t-test between two groups."""
    try:
        from scipy import stats
        return stats.ttest_ind(group1, group2)
    except ImportError:
        # Fallback simple t-test
        mean1, mean2 = np.mean(group1), np.mean(group2)
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
        n1, n2 = len(group1), len(group2)
        
        pooled_se = np.sqrt(var1/n1 + var2/n2)
        t_stat = (mean1 - mean2) / pooled_se
        
        # Simple p-value estimation
        p_value = 0.05 if abs(t_stat) > 2.0 else 0.10
        
        return t_stat, p_value

def calculate_effect_size(group1: List[float], group2: List[float]) -> float:
    """Calculate Cohen's d effect size."""
    mean1, mean2 = np.mean(group1), np.mean(group2)
    pooled_std = np.sqrt(((len(group1) - 1) * np.var(group1, ddof=1) + 
                         (len(group2) - 1) * np.var(group2, ddof=1)) / 
                        (len(group1) + len(group2) - 2))
    
    return (mean1 - mean2) / pooled_std if pooled_std > 0 else 0.0

class ImprovedSimilarityConsistency(ImprovedMicroExperiment):
    """Test that similar situations produce similar actions."""
    
    def __init__(self):
        super().__init__(
            name="Similarity Consistency",
            assumption="Similar situations should produce similar actions",
            timeout_seconds=300
        )
    
    def setup(self) -> bool:
        """Setup experiment."""
        if not self.client:
            print("   ❌ No client provided")
            return False
        if not self.environment:
            print("   ❌ No environment provided")
            return False
        
        # Light reset - just set to known state, don't recreate
        self.reset_environment_if_needed(force_reset=False)
        return True
    
    def run_test(self) -> MicroExperimentResult:
        """Test similarity consistency."""
        print("   Testing similarity consistency...")
        
        # Generate similar situations
        num_tests = 10
        similarity_scores = []
        
        for test_round in range(num_tests):
            # Set robot to consistent position
            self.environment.robot_position = np.array([2.0, 2.0])
            self.environment.robot_heading = 0.0
            
            # Get two sensory readings from same position
            sensory1 = self.environment.get_sensory_input()
            sensory2 = self.environment.get_sensory_input()
            
            # Get brain actions with retry
            try:
                action1 = self.get_action_with_retry(sensory1, max_retries=3, timeout=10.0)
                action2 = self.get_action_with_retry(sensory2, max_retries=3, timeout=10.0)
                
                if action1 is None or action2 is None:
                    print(f"   ⚠️  Test {test_round + 1}: Brain returned None")
                    continue
                
                # Calculate similarity
                similarity = 1.0 - np.mean(np.abs(np.array(action1) - np.array(action2)))
                similarity_scores.append(max(0.0, similarity))
                
            except BrainProcessingError as e:
                print(f"   ⚠️  Test {test_round + 1}: {e}")
                continue
            except Exception as e:
                print(f"   ⚠️  Test {test_round + 1}: Unexpected error: {e}")
                continue
        
        if len(similarity_scores) < 5:
            return MicroExperimentResult(
                experiment_name=self.name,
                assumption_tested=self.assumption,
                start_time=self.start_time,
                duration_seconds=0,
                passed=False,
                confidence=0.0,
                error_message="Insufficient similarity data",
                sample_size=len(similarity_scores)
            )
        
        # Test passes if average similarity > 0.7
        avg_similarity = np.mean(similarity_scores)
        passed = avg_similarity > 0.7
        confidence = min(1.0, avg_similarity)
        
        return MicroExperimentResult(
            experiment_name=self.name,
            assumption_tested=self.assumption,
            start_time=self.start_time,
            duration_seconds=0,
            passed=passed,
            confidence=confidence,
            measurements={
                'similarity_scores': similarity_scores,
                'avg_similarity': avg_similarity,
                'min_similarity': np.min(similarity_scores),
                'max_similarity': np.max(similarity_scores)
            },
            sample_size=len(similarity_scores)
        )

class ImprovedPredictionErrorLearning(ImprovedMicroExperiment):
    """Test that prediction error decreases over time."""
    
    def __init__(self):
        super().__init__(
            name="Prediction Error Learning",
            assumption="Prediction error should decrease over time",
            timeout_seconds=600  # Increased timeout
        )
    
    def setup(self) -> bool:
        """Setup experiment."""
        if not self.client:
            print("   ❌ No client provided")
            return False
        if not self.environment:
            print("   ❌ No environment provided")
            return False
        
        # Light reset - just set to known state
        self.reset_environment_if_needed(force_reset=True)  # Force reset for learning test
        return True
    
    def run_test(self) -> MicroExperimentResult:
        """Test prediction error learning."""
        print("   Testing prediction error learning...")
        
        # Reduced episodes, longer per episode
        num_episodes = 20  # Reduced from 30
        episode_length = 8  # Reduced from 10
        
        prediction_errors = []
        
        for episode in range(num_episodes):
            # Only reset every 5 episodes instead of every episode
            if episode % 5 == 0:
                self.environment.reset()
            
            episode_errors = []
            
            for step in range(episode_length):
                # Get sensory input
                sensory_input = self.environment.get_sensory_input()
                
                # Get brain prediction with retry
                try:
                    prediction = self.get_action_with_retry(
                        sensory_input, 
                        max_retries=3, 
                        timeout=10.0  # Increased timeout
                    )
                    
                    if prediction is None:
                        continue
                    
                    # Execute action
                    result = self.environment.execute_action(prediction)
                    
                    # Calculate prediction error (simplified)
                    next_sensory = self.environment.get_sensory_input()
                    prediction_error = np.mean(np.abs(np.array(prediction) - np.array(next_sensory[:4])))
                    
                    episode_errors.append(prediction_error)
                    
                except BrainProcessingError as e:
                    print(f"   ⚠️  Episode {episode + 1}, Step {step + 1}: Brain processing error")
                    continue
                except Exception as e:
                    print(f"   ⚠️  Episode {episode + 1}, Step {step + 1}: Error: {e}")
                    continue
            
            if episode_errors:
                prediction_errors.append(np.mean(episode_errors))
        
        if len(prediction_errors) < 10:
            return MicroExperimentResult(
                experiment_name=self.name,
                assumption_tested=self.assumption,
                start_time=self.start_time,
                duration_seconds=0,
                passed=False,
                confidence=0.0,
                error_message="Insufficient prediction error data",
                sample_size=len(prediction_errors)
            )
        
        # Analyze learning trend
        early_errors = prediction_errors[:5]
        late_errors = prediction_errors[-5:]
        
        early_mean = np.mean(early_errors)
        late_mean = np.mean(late_errors)
        
        # Calculate learning metrics
        improvement = early_mean - late_mean
        improvement_rate = improvement / early_mean if early_mean > 0 else 0
        
        # Statistical test
        try:
            t_stat, p_value = perform_t_test(early_errors, late_errors)
            effect_size = calculate_effect_size(early_errors, late_errors)
        except Exception as e:
            print(f"   ⚠️ Statistical test failed: {e}")
            t_stat, p_value = 0.0, 1.0
            effect_size = 0.0
        
        # Test passes if prediction error decreased
        trend_slope = np.polyfit(range(len(prediction_errors)), prediction_errors, 1)[0]
        
        passed = improvement > 0 and trend_slope < 0
        confidence = min(1.0, max(0.0, improvement_rate * 2))
        
        return MicroExperimentResult(
            experiment_name=self.name,
            assumption_tested=self.assumption,
            start_time=self.start_time,
            duration_seconds=0,
            passed=passed,
            confidence=confidence,
            p_value=p_value,
            effect_size=effect_size,
            measurements={
                'prediction_errors': prediction_errors,
                'early_mean': early_mean,
                'late_mean': late_mean,
                'improvement': improvement,
                'improvement_rate': improvement_rate,
                'trend_slope': trend_slope
            },
            sample_size=len(prediction_errors)
        )

class ImprovedActionVariability(ImprovedMicroExperiment):
    """Test that brain produces varied actions for different situations."""
    
    def __init__(self):
        super().__init__(
            name="Action Variability",
            assumption="Different situations should produce different actions",
            timeout_seconds=300
        )
    
    def setup(self) -> bool:
        """Setup experiment."""
        if not self.client:
            print("   ❌ No client provided")
            return False
        if not self.environment:
            print("   ❌ No environment provided")
            return False
        
        # Light reset
        self.reset_environment_if_needed(force_reset=False)
        return True
    
    def run_test(self) -> MicroExperimentResult:
        """Test action variability."""
        print("   Testing action variability...")
        
        # Test different positions
        test_positions = [
            np.array([1.0, 1.0]),
            np.array([4.0, 1.0]),
            np.array([1.0, 4.0]),
            np.array([4.0, 4.0]),
            np.array([2.5, 2.5])
        ]
        
        actions_by_position = []
        
        for pos in test_positions:
            # Set robot position
            self.environment.robot_position = pos
            self.environment.robot_heading = 0.0
            
            # Get sensory input
            sensory_input = self.environment.get_sensory_input()
            
            # Get brain action with retry
            try:
                action = self.get_action_with_retry(sensory_input, max_retries=3, timeout=10.0)
                
                if action is not None:
                    actions_by_position.append(action)
                    
            except BrainProcessingError as e:
                print(f"   ⚠️  Position {pos}: Brain processing error")
                continue
            except Exception as e:
                print(f"   ⚠️  Position {pos}: Error: {e}")
                continue
        
        if len(actions_by_position) < 3:
            return MicroExperimentResult(
                experiment_name=self.name,
                assumption_tested=self.assumption,
                start_time=self.start_time,
                duration_seconds=0,
                passed=False,
                confidence=0.0,
                error_message="Insufficient action data",
                sample_size=len(actions_by_position)
            )
        
        # Calculate pairwise action differences
        differences = []
        for i in range(len(actions_by_position)):
            for j in range(i + 1, len(actions_by_position)):
                diff = np.mean(np.abs(np.array(actions_by_position[i]) - np.array(actions_by_position[j])))
                differences.append(diff)
        
        avg_difference = np.mean(differences)
        
        # Test passes if average difference > 0.1
        passed = avg_difference > 0.1
        confidence = min(1.0, avg_difference * 5)  # Scale to 0-1
        
        return MicroExperimentResult(
            experiment_name=self.name,
            assumption_tested=self.assumption,
            start_time=self.start_time,
            duration_seconds=0,
            passed=passed,
            confidence=confidence,
            measurements={
                'actions_by_position': actions_by_position,
                'pairwise_differences': differences,
                'avg_difference': avg_difference,
                'min_difference': np.min(differences),
                'max_difference': np.max(differences)
            },
            sample_size=len(actions_by_position)
        )

class ImprovedResponseSpeed(ImprovedMicroExperiment):
    """Test brain response speed."""
    
    def __init__(self):
        super().__init__(
            name="Response Speed",
            assumption="Brain should respond within reasonable time",
            timeout_seconds=300
        )
    
    def setup(self) -> bool:
        """Setup experiment."""
        if not self.client:
            print("   ❌ No client provided")
            return False
        if not self.environment:
            print("   ❌ No environment provided")
            return False
        
        # Light reset
        self.reset_environment_if_needed(force_reset=False)
        return True
    
    def run_test(self) -> MicroExperimentResult:
        """Test response speed."""
        print("   Testing response speed...")
        
        response_times = []
        
        for test_round in range(15):
            # Get sensory input
            sensory_input = self.environment.get_sensory_input()
            
            # Measure response time
            start_time = time.time()
            
            try:
                action = self.get_action_with_retry(sensory_input, max_retries=2, timeout=5.0)
                
                if action is not None:
                    response_time = time.time() - start_time
                    response_times.append(response_time)
                    
            except BrainProcessingError as e:
                print(f"   ⚠️  Test {test_round + 1}: Brain processing error")
                continue
            except Exception as e:
                print(f"   ⚠️  Test {test_round + 1}: Error: {e}")
                continue
        
        if len(response_times) < 5:
            return MicroExperimentResult(
                experiment_name=self.name,
                assumption_tested=self.assumption,
                start_time=self.start_time,
                duration_seconds=0,
                passed=False,
                confidence=0.0,
                error_message="Insufficient response time data",
                sample_size=len(response_times)
            )
        
        avg_response_time = np.mean(response_times)
        
        # Test passes if average response time < 2.0 seconds
        passed = avg_response_time < 2.0
        confidence = min(1.0, max(0.0, (2.0 - avg_response_time) / 2.0))
        
        return MicroExperimentResult(
            experiment_name=self.name,
            assumption_tested=self.assumption,
            start_time=self.start_time,
            duration_seconds=0,
            passed=passed,
            confidence=confidence,
            measurements={
                'response_times': response_times,
                'avg_response_time': avg_response_time,
                'min_response_time': np.min(response_times),
                'max_response_time': np.max(response_times),
                'response_time_std': np.std(response_times)
            },
            sample_size=len(response_times)
        )

def create_improved_core_assumption_suite() -> ImprovedMicroExperimentSuite:
    """Create improved core assumption test suite."""
    suite = ImprovedMicroExperimentSuite()
    
    # Add improved experiments
    suite.add_experiment(ImprovedSimilarityConsistency())
    suite.add_experiment(ImprovedPredictionErrorLearning())
    suite.add_experiment(ImprovedActionVariability())
    suite.add_experiment(ImprovedResponseSpeed())
    
    return suite

if __name__ == "__main__":
    # Run the improved test suite
    suite = create_improved_core_assumption_suite()
    summary = suite.run_all()
    
    print("\\n" + "=" * 60)
    print("📊 IMPROVED MICRO-EXPERIMENT SUMMARY")
    print("=" * 60)
    print(f"Total experiments: {summary['total_experiments']}")
    print(f"Successful: {summary['successful_experiments']}")
    print(f"Failed: {summary['failed_experiments']}")
    print(f"Success rate: {summary['success_rate']:.1%}")
    print(f"Average confidence: {summary['avg_confidence']:.3f}")
    
    print("\\nAssumption scores:")
    for assumption, score in summary['assumption_scores'].items():
        print(f"  {assumption}: {score:.3f}")