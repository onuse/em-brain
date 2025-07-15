#!/usr/bin/env python3
"""
Core Assumption Micro-Experiments

Tests the fundamental assumptions about brain architecture:
1. Similarity-based intelligence
2. Prediction error learning
3. Experience-based improvement
4. Sensory-motor coordination
"""

import sys
import os
import time
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path

# Add paths
brain_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(brain_root))
sys.path.insert(0, str(brain_root / 'server'))

from validation.micro_experiments.framework import MicroExperiment, MicroExperimentResult
from validation.micro_experiments.framework import calculate_confidence_interval, perform_t_test, calculate_effect_size

class SimilarityConsistencyExperiment(MicroExperiment):
    """Test that similar inputs produce similar outputs consistently."""
    
    def __init__(self):
        super().__init__(
            name="Similarity Consistency",
            assumption="Similar situations should produce similar actions",
            timeout_seconds=180
        )
    
    def setup(self) -> bool:
        """Setup experiment."""
        return (self.connect_to_brain() and 
                self.create_environment(random_seed=42))
    
    def run_test(self) -> MicroExperimentResult:
        """Test similarity consistency."""
        print("   Testing similarity consistency...")
        
        # Generate test scenarios
        base_input = [0.5, 0.5, 0.0, 0.5] * 4  # 16D neutral input
        similar_inputs = []
        
        # Create similar inputs by adding small noise
        for i in range(10):
            noise = np.random.normal(0, 0.05, 16)  # 5% noise
            similar_input = np.clip(np.array(base_input) + noise, 0, 1)
            similar_inputs.append(similar_input.tolist())
        
        # Test base input multiple times
        base_responses = []
        for i in range(5):
            response = self.client.get_action(base_input, timeout=3.0)
            if response is not None:
                base_responses.append(response)
        
        # Test similar inputs
        similar_responses = []
        for similar_input in similar_inputs:
            response = self.client.get_action(similar_input, timeout=3.0)
            if response is not None:
                similar_responses.append(response)
        
        if len(base_responses) < 3 or len(similar_responses) < 5:
            return MicroExperimentResult(
                experiment_name=self.name,
                assumption_tested=self.assumption,
                start_time=self.start_time,
                duration_seconds=0,
                passed=False,
                confidence=0.0,
                error_message="Insufficient responses from brain"
            )
        
        # Calculate consistency metrics
        base_consistency = self._calculate_response_consistency(base_responses)
        similar_consistency = self._calculate_response_similarity(base_responses[0], similar_responses)
        
        # Test passes if:
        # 1. Base input is consistent (low variation)
        # 2. Similar inputs produce similar outputs
        base_consistent = base_consistency < 0.3  # Less than 30% variation
        similar_consistent = similar_consistency > 0.7  # More than 70% similarity
        
        passed = base_consistent and similar_consistent
        confidence = min(1.0, (1.0 - base_consistency) * 0.5 + similar_consistency * 0.5)
        
        measurements = {
            'base_consistency': base_consistency,
            'similar_consistency': similar_consistency,
            'base_responses': base_responses,
            'similar_responses': similar_responses[:5],  # Limit for storage
            'sample_size': len(base_responses) + len(similar_responses)
        }
        
        return MicroExperimentResult(
            experiment_name=self.name,
            assumption_tested=self.assumption,
            start_time=self.start_time,
            duration_seconds=0,
            passed=passed,
            confidence=confidence,
            measurements=measurements,
            sample_size=len(base_responses) + len(similar_responses)
        )
    
    def _calculate_response_consistency(self, responses: List[List[float]]) -> float:
        """Calculate consistency of responses (lower is more consistent)."""
        if len(responses) < 2:
            return 1.0
        
        # Calculate coefficient of variation for each dimension
        responses_array = np.array(responses)
        means = np.mean(responses_array, axis=0)
        stds = np.std(responses_array, axis=0)
        
        # Avoid division by zero
        cv = np.where(means != 0, stds / np.abs(means), 0)
        
        return np.mean(cv)
    
    def _calculate_response_similarity(self, base_response: List[float], similar_responses: List[List[float]]) -> float:
        """Calculate similarity between base and similar responses."""
        if not similar_responses:
            return 0.0
        
        base_array = np.array(base_response)
        similarities = []
        
        for response in similar_responses:
            response_array = np.array(response)
            # Calculate cosine similarity
            dot_product = np.dot(base_array, response_array)
            norm_base = np.linalg.norm(base_array)
            norm_response = np.linalg.norm(response_array)
            
            if norm_base == 0 or norm_response == 0:
                similarity = 0.0
            else:
                similarity = dot_product / (norm_base * norm_response)
            
            similarities.append(similarity)
        
        return np.mean(similarities)
    
    def cleanup(self):
        """Clean up resources."""
        if self.client:
            self.client.disconnect()

class PredictionErrorLearningExperiment(MicroExperiment):
    """Test that prediction error decreases with experience."""
    
    def __init__(self):
        super().__init__(
            name="Prediction Error Learning",
            assumption="Prediction error should decrease over time",
            timeout_seconds=300
        )
    
    def setup(self) -> bool:
        """Setup experiment."""
        return (self.connect_to_brain() and 
                self.create_environment(random_seed=42))
    
    def run_test(self) -> MicroExperimentResult:
        """Test prediction error learning."""
        print("   Testing prediction error learning...")
        
        # Run learning episodes
        num_episodes = 30
        episode_length = 10  # actions per episode
        
        prediction_errors = []
        
        for episode in range(num_episodes):
            # Reset environment
            self.environment.reset()
            
            episode_errors = []
            
            for step in range(episode_length):
                # Get sensory input
                sensory_input = self.environment.get_sensory_input()
                
                # Get brain prediction
                prediction = self.client.get_action(sensory_input, timeout=3.0)
                
                if prediction is None:
                    continue
                
                # Execute action
                result = self.environment.execute_action(prediction)
                
                # Calculate prediction error (simplified)
                next_sensory = self.environment.get_sensory_input()
                prediction_error = np.mean(np.abs(np.array(prediction) - np.array(next_sensory[:4])))
                
                episode_errors.append(prediction_error)
            
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
                error_message="Insufficient prediction error data"
            )
        
        # Analyze learning trend
        early_errors = prediction_errors[:10]
        late_errors = prediction_errors[-10:]
        
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
        
        # Test passes if:
        # 1. Prediction error decreased (any improvement)
        # 2. Trend is downward overall
        trend_slope = np.polyfit(range(len(prediction_errors)), prediction_errors, 1)[0]
        
        passed = improvement > 0 and trend_slope < 0
        confidence = min(1.0, max(0.0, improvement_rate * 2))  # Scale improvement to confidence
        
        measurements = {
            'prediction_errors': prediction_errors,
            'early_mean': early_mean,
            'late_mean': late_mean,
            'improvement': improvement,
            'improvement_rate': improvement_rate,
            'trend_slope': trend_slope,
            't_stat': t_stat,
            'p_value': p_value,
            'effect_size': effect_size
        }
        
        return MicroExperimentResult(
            experiment_name=self.name,
            assumption_tested=self.assumption,
            start_time=self.start_time,
            duration_seconds=0,
            passed=passed,
            confidence=confidence,
            p_value=p_value,
            effect_size=effect_size,
            measurements=measurements,
            sample_size=len(prediction_errors)
        )
    
    def cleanup(self):
        """Clean up resources."""
        if self.client:
            self.client.disconnect()

class ExperienceScalingExperiment(MicroExperiment):
    """Test that performance improves with more experience."""
    
    def __init__(self):
        super().__init__(
            name="Experience Scaling",
            assumption="More experience should lead to better performance",
            timeout_seconds=240
        )
    
    def setup(self) -> bool:
        """Setup experiment."""
        return (self.connect_to_brain() and 
                self.create_environment(random_seed=42))
    
    def run_test(self) -> MicroExperimentResult:
        """Test experience scaling."""
        print("   Testing experience scaling...")
        
        # Define experience levels
        experience_levels = [5, 10, 15, 20, 25]  # Number of actions
        performance_scores = []
        
        for num_actions in experience_levels:
            # Reset environment
            self.environment.reset()
            
            # Accumulate experience
            for i in range(num_actions):
                sensory_input = self.environment.get_sensory_input()
                action = self.client.get_action(sensory_input, timeout=3.0)
                
                if action is not None:
                    self.environment.execute_action(action)
            
            # Test performance
            performance = self._measure_performance()
            performance_scores.append(performance)
        
        if len(performance_scores) < 3:
            return MicroExperimentResult(
                experiment_name=self.name,
                assumption_tested=self.assumption,
                start_time=self.start_time,
                duration_seconds=0,
                passed=False,
                confidence=0.0,
                error_message="Insufficient performance data"
            )
        
        # Analyze scaling trend
        correlation = np.corrcoef(experience_levels, performance_scores)[0, 1]
        if np.isnan(correlation):
            correlation = 0.0
        
        slope = np.polyfit(experience_levels, performance_scores, 1)[0]
        if np.isnan(slope):
            slope = 0.0
        
        # Test passes if performance correlates positively with experience
        passed = correlation > 0.3 and slope > 0
        confidence = min(1.0, abs(correlation))
        
        measurements = {
            'experience_levels': experience_levels,
            'performance_scores': performance_scores,
            'correlation': correlation,
            'slope': slope
        }
        
        return MicroExperimentResult(
            experiment_name=self.name,
            assumption_tested=self.assumption,
            start_time=self.start_time,
            duration_seconds=0,
            passed=passed,
            confidence=confidence,
            measurements=measurements,
            sample_size=len(performance_scores)
        )
    
    def _measure_performance(self) -> float:
        """Measure current performance."""
        # Run short performance test
        total_score = 0
        num_tests = 5
        
        for i in range(num_tests):
            sensory_input = self.environment.get_sensory_input()
            action = self.client.get_action(sensory_input, timeout=3.0)
            
            if action is not None:
                result = self.environment.execute_action(action)
                
                # Simple performance metric: distance to light
                metrics = result.get('metrics', {})
                light_distance = metrics.get('min_light_distance', 1.0)
                
                # Convert to score (closer to light = higher score)
                score = 1.0 - min(light_distance, 1.0)
                total_score += score
        
        return total_score / num_tests
    
    def cleanup(self):
        """Clean up resources."""
        if self.client:
            self.client.disconnect()

class SensoryMotorCoordinationExperiment(MicroExperiment):
    """Test that 16D sensory input enables meaningful 4D actions."""
    
    def __init__(self):
        super().__init__(
            name="Sensory-Motor Coordination",
            assumption="16D sensory input should provide sufficient information for 4D actions",
            timeout_seconds=180
        )
    
    def setup(self) -> bool:
        """Setup experiment."""
        return (self.connect_to_brain() and 
                self.create_environment(random_seed=42))
    
    def run_test(self) -> MicroExperimentResult:
        """Test sensory-motor coordination."""
        print("   Testing sensory-motor coordination...")
        
        # Test different sensory scenarios
        test_scenarios = [
            "light_close",      # Light source nearby
            "light_far",        # Light source far away
            "obstacle_close",   # Obstacle nearby
            "obstacle_far",     # Obstacle far away
            "low_battery",      # Low battery state
            "high_battery"      # High battery state
        ]
        
        coordination_scores = []
        
        for scenario in test_scenarios:
            try:
                # Setup scenario
                self._setup_scenario(scenario)
                
                # Test coordination
                score = self._test_coordination()
                coordination_scores.append(score)
            except Exception as e:
                print(f"   ⚠️ Error in scenario {scenario}: {e}")
                coordination_scores.append(0.0)
        
        if len(coordination_scores) < 3:
            return MicroExperimentResult(
                experiment_name=self.name,
                assumption_tested=self.assumption,
                start_time=self.start_time,
                duration_seconds=0,
                passed=False,
                confidence=0.0,
                error_message="Insufficient coordination data"
            )
        
        # Analyze coordination
        avg_coordination = np.mean(coordination_scores)
        coordination_consistency = max(0.0, 1.0 - np.std(coordination_scores))  # Higher is more consistent
        
        # Test passes if coordination is reasonably good and consistent
        passed = avg_coordination > 0.4 and coordination_consistency > 0.7
        confidence = min(1.0, avg_coordination)
        
        measurements = {
            'scenarios': test_scenarios,
            'coordination_scores': coordination_scores,
            'avg_coordination': avg_coordination,
            'coordination_consistency': coordination_consistency
        }
        
        return MicroExperimentResult(
            experiment_name=self.name,
            assumption_tested=self.assumption,
            start_time=self.start_time,
            duration_seconds=0,
            passed=passed,
            confidence=confidence,
            measurements=measurements,
            sample_size=len(coordination_scores)
        )
    
    def _setup_scenario(self, scenario: str):
        """Setup specific test scenario."""
        self.environment.reset()
        
        # Modify environment based on scenario
        if scenario == "light_close":
            # Move robot near light
            self.environment.robot_state.position = [2.0, 2.0]
        elif scenario == "light_far":
            # Move robot far from lights
            self.environment.robot_state.position = [8.0, 8.0]
        elif scenario == "low_battery":
            # Set low battery
            self.environment.robot_state.battery = 100.0  # 10% of capacity
        elif scenario == "high_battery":
            # Set high battery
            self.environment.robot_state.battery = 900.0  # 90% of capacity
        
        # Note: obstacle scenarios would require environment modification
    
    def _test_coordination(self) -> float:
        """Test coordination in current scenario."""
        num_actions = 10
        coordination_score = 0
        
        for i in range(num_actions):
            # Get sensory input
            sensory_input = self.environment.get_sensory_input()
            
            # Get action
            action = self.client.get_action(sensory_input, timeout=3.0)
            
            if action is None:
                continue
            
            # Execute action
            result = self.environment.execute_action(action)
            
            # Evaluate coordination (simplified)
            if result.get('success', False):
                # Check if action was appropriate for situation
                metrics = result.get('metrics', {})
                light_distance = metrics.get('min_light_distance', 1.0)
                
                # Good coordination = moving toward light when far, exploring when close
                if light_distance > 0.5:
                    # Should move toward light
                    action_type = result.get('action_executed', 0)
                    if action_type == 0:  # MOVE_FORWARD
                        coordination_score += 1.0
                    elif action_type in [1, 2]:  # TURN_LEFT, TURN_RIGHT
                        coordination_score += 0.5
                else:
                    # Should explore or be strategic
                    coordination_score += 0.7  # Any reasonable action
        
        return coordination_score / num_actions
    
    def cleanup(self):
        """Clean up resources."""
        if self.client:
            self.client.disconnect()

def create_core_assumption_suite():
    """Create suite of core assumption experiments."""
    from validation.micro_experiments.framework import MicroExperimentSuite
    
    suite = MicroExperimentSuite()
    
    # Add core experiments
    suite.add_experiment(SimilarityConsistencyExperiment())
    suite.add_experiment(PredictionErrorLearningExperiment())
    suite.add_experiment(ExperienceScalingExperiment())
    suite.add_experiment(SensoryMotorCoordinationExperiment())
    
    return suite

if __name__ == "__main__":
    # Run core assumption experiments
    suite = create_core_assumption_suite()
    summary = suite.run_all()
    suite.print_summary()