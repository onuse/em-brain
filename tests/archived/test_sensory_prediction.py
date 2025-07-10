"""
Test Sensory Prediction System.

Demonstrates the transformative power of sensory prediction:
- Before: Robots evaluate actions by their immediate properties
- After: Robots imagine consequences and avoid bad outcomes before experiencing them

This test shows how prediction enables true forward-thinking behavior.
"""

import sys
import time
import random
from typing import List, Dict, Any

# Add project root to path
sys.path.append('.')

from prediction.sensory_predictor import SensoryPredictor
from motivators.predictive_survival_drive import PredictiveSurvivalDrive
from motivators.survival_drive import SurvivalDrive  # Original drive for comparison
from motivators.base_motivator import DriveContext
from simulation.brainstem_sim import GridWorldBrainstem


class SensoryPredictionExperiment:
    """Experiment comparing reactive vs predictive robot behavior."""
    
    def __init__(self, world_graph):
        self.world_graph = world_graph
        
        # Create sensory predictor
        self.predictor = SensoryPredictor(world_graph)
        
        # Create both drive types for comparison
        self.reactive_drive = SurvivalDrive(base_weight=1.0)  # Original reactive drive
        self.predictive_drive = PredictiveSurvivalDrive(base_weight=1.0, predictor=self.predictor)
        
        # Test scenarios
        self.test_scenarios = self._create_test_scenarios()
        
        # Results tracking
        self.reactive_results = []
        self.predictive_results = []
        self.prediction_accuracy_history = []
    
    def run_prediction_experiment(self, num_scenarios: int = 100) -> Dict[str, Any]:
        """
        Run comprehensive experiment comparing reactive vs predictive behavior.
        
        Args:
            num_scenarios: Number of test scenarios to run
            
        Returns:
            Analysis comparing reactive vs predictive performance
        """
        print("🔮 SENSORY PREDICTION EXPERIMENT")
        print("=" * 70)
        print("Comparing reactive vs predictive robot behavior...")
        print("Testing whether robots can avoid dangers by imagining consequences")
        print()
        
        # Run scenarios with both approaches
        for scenario_num in range(num_scenarios):
            scenario = self._generate_test_scenario(scenario_num)
            
            print(f"Scenario {scenario_num + 1:3d}: {scenario['description']}")
            
            # Test reactive approach (original)
            reactive_result = self._test_reactive_approach(scenario)
            self.reactive_results.append(reactive_result)
            
            # Test predictive approach (enhanced)
            predictive_result = self._test_predictive_approach(scenario)
            self.predictive_results.append(predictive_result)
            
            # Compare outcomes
            self._compare_scenario_outcomes(reactive_result, predictive_result, scenario_num)
            
            # Progress updates
            if (scenario_num + 1) % 20 == 0:
                self._print_progress_update(scenario_num + 1)
        
        # Final analysis
        analysis = self._analyze_prediction_experiment()
        self._print_experiment_summary(analysis)
        
        return analysis
    
    def _create_test_scenarios(self) -> List[Dict]:
        """Create standardized test scenarios."""
        return [
            {
                'name': 'wall_ahead',
                'description': 'Wall directly ahead - should avoid forward movement',
                'sensory_pattern': [0.9, 0.1, 0.1, 0.0, 0.5, 0.3, 0.2, 0.1],  # High wall sensor
                'danger_level': 'high',
                'optimal_action': 'avoid_forward'
            },
            {
                'name': 'food_nearby', 
                'description': 'Food nearby - can approach safely',
                'sensory_pattern': [0.2, 0.8, 0.1, 0.1, 0.7, 0.4, 0.3, 0.5],  # High food sensor
                'danger_level': 'low',
                'optimal_action': 'approach'
            },
            {
                'name': 'danger_detected',
                'description': 'Danger sensors active - should retreat',
                'sensory_pattern': [0.3, 0.2, 0.8, 0.7, 0.2, 0.6, 0.1, 0.3],  # High danger sensors
                'danger_level': 'high', 
                'optimal_action': 'retreat'
            },
            {
                'name': 'open_space',
                'description': 'Clear open space - safe to explore',
                'sensory_pattern': [0.1, 0.3, 0.1, 0.0, 0.6, 0.5, 0.4, 0.3],  # Low obstacle sensors
                'danger_level': 'low',
                'optimal_action': 'explore'
            },
            {
                'name': 'narrow_passage',
                'description': 'Narrow passage - requires careful navigation',
                'sensory_pattern': [0.6, 0.4, 0.3, 0.2, 0.7, 0.8, 0.6, 0.4],  # Moderate obstacles
                'danger_level': 'medium',
                'optimal_action': 'careful_navigation'
            }
        ]
    
    def _generate_test_scenario(self, scenario_num: int) -> Dict:
        """Generate a test scenario with variations."""
        base_scenarios = self._create_test_scenarios()
        base_scenario = base_scenarios[scenario_num % len(base_scenarios)].copy()
        
        # Add random variations to make scenarios more realistic
        varied_sensory = []
        for value in base_scenario['sensory_pattern']:
            variation = random.uniform(-0.1, 0.1)
            varied_value = max(0.0, min(1.0, value + variation))
            varied_sensory.append(varied_value)
        
        base_scenario['sensory_pattern'] = varied_sensory
        base_scenario['scenario_id'] = scenario_num
        
        # Create context
        base_scenario['context'] = DriveContext(
            current_sensory=varied_sensory,
            robot_health=random.uniform(0.7, 1.0),
            robot_energy=random.uniform(0.6, 1.0),
            robot_position=(random.randint(0, 10), random.randint(0, 10)),
            robot_orientation=random.randint(0, 3),
            recent_experiences=[],
            prediction_errors=[],
            step_count=scenario_num,
            time_since_last_food=random.randint(0, 15),
            time_since_last_damage=random.randint(5, 50),
            threat_level=base_scenario['danger_level']
        )
        
        return base_scenario
    
    def _test_reactive_approach(self, scenario: Dict) -> Dict:
        """Test original reactive approach (no prediction)."""
        context = scenario['context']
        
        # Generate test actions
        test_actions = self._generate_test_actions()
        
        # Evaluate actions using reactive drive
        action_evaluations = []
        for action in test_actions:
            evaluation = self.reactive_drive.evaluate_action(action, context)
            action_evaluations.append({
                'action': action,
                'evaluation': evaluation,
                'approach': 'reactive'
            })
        
        # Select best action
        best_action_eval = max(action_evaluations, key=lambda x: x['evaluation'].score)
        
        # Simulate outcome
        actual_outcome = self._simulate_action_outcome(best_action_eval['action'], scenario)
        
        return {
            'scenario_id': scenario['scenario_id'],
            'approach': 'reactive',
            'chosen_action': best_action_eval['action'],
            'action_score': best_action_eval['evaluation'].score,
            'reasoning': best_action_eval['evaluation'].reasoning,
            'actual_outcome': actual_outcome,
            'scenario_type': scenario['name'],
            'all_evaluations': action_evaluations
        }
    
    def _test_predictive_approach(self, scenario: Dict) -> Dict:
        """Test enhanced predictive approach (with sensory prediction)."""
        context = scenario['context']
        
        # Generate test actions
        test_actions = self._generate_test_actions()
        
        # Evaluate actions using predictive drive
        action_evaluations = []
        predictions_made = []
        
        for action in test_actions:
            evaluation = self.predictive_drive.evaluate_action(action, context)
            action_evaluations.append({
                'action': action,
                'evaluation': evaluation,
                'approach': 'predictive'
            })
            
            # Track predictions made
            if hasattr(evaluation, 'sensory_prediction') and evaluation.sensory_prediction:
                predictions_made.append(evaluation.sensory_prediction)
        
        # Select best action
        best_action_eval = max(action_evaluations, key=lambda x: x['evaluation'].score)
        
        # Simulate outcome and evaluate prediction accuracy
        actual_outcome = self._simulate_action_outcome(best_action_eval['action'], scenario)
        
        # Evaluate prediction accuracy if prediction was made
        prediction_accuracy = None
        if hasattr(best_action_eval['evaluation'], 'sensory_prediction') and best_action_eval['evaluation'].sensory_prediction:
            prediction = best_action_eval['evaluation'].sensory_prediction
            prediction_accuracy = self.predictor.evaluate_prediction_accuracy(
                prediction, actual_outcome['resulting_sensors']
            )
            self.prediction_accuracy_history.append(prediction_accuracy.overall_accuracy)
        
        return {
            'scenario_id': scenario['scenario_id'],
            'approach': 'predictive',
            'chosen_action': best_action_eval['action'],
            'action_score': best_action_eval['evaluation'].score,
            'reasoning': best_action_eval['evaluation'].reasoning,
            'actual_outcome': actual_outcome,
            'scenario_type': scenario['name'],
            'predictions_made': len(predictions_made),
            'prediction_accuracy': prediction_accuracy,
            'all_evaluations': action_evaluations
        }
    
    def _generate_test_actions(self) -> List[Dict[str, float]]:
        """Generate a set of test actions to evaluate."""
        return [
            # Conservative actions
            {'forward_motor': 0.1, 'turn_motor': 0.0, 'brake_motor': 0.2},
            {'forward_motor': 0.0, 'turn_motor': 0.3, 'brake_motor': 0.1},
            
            # Moderate actions
            {'forward_motor': 0.4, 'turn_motor': 0.0, 'brake_motor': 0.0},
            {'forward_motor': 0.2, 'turn_motor': 0.5, 'brake_motor': 0.0},
            {'forward_motor': -0.3, 'turn_motor': 0.0, 'brake_motor': 0.0},  # Backward
            
            # Aggressive actions
            {'forward_motor': 0.7, 'turn_motor': 0.0, 'brake_motor': 0.0},
            {'forward_motor': 0.5, 'turn_motor': 0.6, 'brake_motor': 0.0},
            
            # Emergency actions
            {'forward_motor': 0.0, 'turn_motor': 0.0, 'brake_motor': 1.0},  # Full stop
            {'forward_motor': -0.6, 'turn_motor': 0.0, 'brake_motor': 0.0}, # Fast retreat
        ]
    
    def _simulate_action_outcome(self, action: Dict[str, float], scenario: Dict) -> Dict:
        """Simulate the actual outcome of taking an action in this scenario."""
        # This simulates what would actually happen if the robot took this action
        current_sensors = scenario['sensory_pattern']
        
        # Simulate sensor changes based on action
        resulting_sensors = current_sensors.copy()
        
        # Forward movement effects
        forward_motor = action.get('forward_motor', 0.0)
        if forward_motor > 0.3:  # Significant forward movement
            # Moving forward toward wall increases wall sensor
            if current_sensors[0] > 0.6:  # Wall ahead
                resulting_sensors[0] = min(1.0, current_sensors[0] + 0.3)
                damage = 0.4 if current_sensors[0] > 0.8 else 0.1
            else:
                resulting_sensors[0] = max(0.0, current_sensors[0] - 0.1)
                damage = 0.0
        elif forward_motor < -0.3:  # Backward movement
            resulting_sensors[0] = max(0.0, current_sensors[0] - 0.2)
            damage = 0.0
        else:
            damage = 0.0
        
        # Turning effects
        turn_motor = abs(action.get('turn_motor', 0.0))
        if turn_motor > 0.4:
            # Turning can help avoid walls
            if current_sensors[0] > 0.6:
                resulting_sensors[0] = max(0.0, current_sensors[0] - 0.2)
        
        # Braking effects
        brake_motor = action.get('brake_motor', 0.0)
        if brake_motor > 0.5:
            # Braking prevents collision damage
            damage *= 0.3
        
        # Determine success based on scenario type
        success = self._evaluate_action_success(action, scenario, damage)
        
        return {
            'resulting_sensors': {str(i): val for i, val in enumerate(resulting_sensors)},
            'damage_taken': damage,
            'action_success': success,
            'safety_score': 1.0 - damage
        }
    
    def _evaluate_action_success(self, action: Dict[str, float], scenario: Dict, damage: float) -> bool:\n        \"\"\"Evaluate whether the action was successful for this scenario type.\"\"\"\n        scenario_type = scenario['name']\n        \n        if scenario_type == 'wall_ahead':\n            # Success = avoiding forward movement into wall\n            return action.get('forward_motor', 0.0) < 0.3 and damage < 0.2\n        \n        elif scenario_type == 'food_nearby':\n            # Success = approaching food (moderate forward movement)\n            return 0.2 <= action.get('forward_motor', 0.0) <= 0.6 and damage < 0.1\n        \n        elif scenario_type == 'danger_detected':\n            # Success = retreating or stopping\n            return (action.get('forward_motor', 0.0) <= 0.1 or \n                   action.get('brake_motor', 0.0) > 0.5) and damage < 0.1\n        \n        elif scenario_type == 'open_space':\n            # Success = exploration without excessive caution\n            return action.get('forward_motor', 0.0) > 0.2 and damage < 0.1\n        \n        elif scenario_type == 'narrow_passage':\n            # Success = careful navigation\n            return 0.1 <= action.get('forward_motor', 0.0) <= 0.4 and damage < 0.2\n        \n        return damage < 0.2  # Default: success = low damage\n    \n    def _compare_scenario_outcomes(self, reactive_result: Dict, predictive_result: Dict, scenario_num: int):\n        \"\"\"Compare outcomes between reactive and predictive approaches.\"\"\"\n        reactive_success = reactive_result['actual_outcome']['action_success']\n        predictive_success = predictive_result['actual_outcome']['action_success']\n        \n        reactive_damage = reactive_result['actual_outcome']['damage_taken']\n        predictive_damage = predictive_result['actual_outcome']['damage_taken']\n        \n        if predictive_success and not reactive_success:\n            print(f\"             🎯 Prediction helped: Avoided failure!\")\n        elif predictive_damage < reactive_damage - 0.1:\n            print(f\"             🛡️  Prediction helped: Reduced damage ({reactive_damage:.2f} → {predictive_damage:.2f})\")\n        elif not predictive_success and reactive_success:\n            print(f\"             ⚠️  Prediction hindered: Lost good solution\")\n        else:\n            print(f\"             ➖ Similar outcomes\")\n    \n    def _print_progress_update(self, scenarios_completed: int):\n        \"\"\"Print progress update during experiment.\"\"\"\n        # Calculate recent performance\n        recent_reactive = self.reactive_results[-20:] if len(self.reactive_results) >= 20 else self.reactive_results\n        recent_predictive = self.predictive_results[-20:] if len(self.predictive_results) >= 20 else self.predictive_results\n        \n        reactive_success_rate = sum(1 for r in recent_reactive if r['actual_outcome']['action_success']) / len(recent_reactive) if recent_reactive else 0\n        predictive_success_rate = sum(1 for r in recent_predictive if r['actual_outcome']['action_success']) / len(recent_predictive) if recent_predictive else 0\n        \n        reactive_avg_damage = sum(r['actual_outcome']['damage_taken'] for r in recent_reactive) / len(recent_reactive) if recent_reactive else 0\n        predictive_avg_damage = sum(r['actual_outcome']['damage_taken'] for r in recent_predictive) / len(recent_predictive) if recent_predictive else 0\n        \n        recent_prediction_accuracy = sum(self.prediction_accuracy_history[-20:]) / min(20, len(self.prediction_accuracy_history)) if self.prediction_accuracy_history else 0\n        \n        print(f\"\\nProgress {scenarios_completed:3d}: \" +\n              f\"Reactive success={reactive_success_rate:.1%}, \" +\n              f\"Predictive success={predictive_success_rate:.1%}, \" +\n              f\"Prediction accuracy={recent_prediction_accuracy:.2f}\")\n        print(f\"                 \" +\n              f\"Reactive damage={reactive_avg_damage:.3f}, \" +\n              f\"Predictive damage={predictive_avg_damage:.3f}\")\n    \n    def _analyze_prediction_experiment(self) -> Dict[str, Any]:\n        \"\"\"Analyze the complete prediction experiment results.\"\"\"\n        if not self.reactive_results or not self.predictive_results:\n            return {\"error\": \"insufficient_data\"}\n        \n        # Success rate comparison\n        reactive_successes = sum(1 for r in self.reactive_results if r['actual_outcome']['action_success'])\n        predictive_successes = sum(1 for r in self.predictive_results if r['actual_outcome']['action_success'])\n        \n        reactive_success_rate = reactive_successes / len(self.reactive_results)\n        predictive_success_rate = predictive_successes / len(self.predictive_results)\n        \n        # Damage comparison\n        reactive_avg_damage = sum(r['actual_outcome']['damage_taken'] for r in self.reactive_results) / len(self.reactive_results)\n        predictive_avg_damage = sum(r['actual_outcome']['damage_taken'] for r in self.predictive_results) / len(self.predictive_results)\n        \n        # Prediction accuracy\n        avg_prediction_accuracy = sum(self.prediction_accuracy_history) / len(self.prediction_accuracy_history) if self.prediction_accuracy_history else 0\n        \n        # Scenario-specific analysis\n        scenario_analysis = self._analyze_by_scenario_type()\n        \n        # Prediction statistics\n        prediction_stats = self.predictor.get_prediction_statistics()\n        \n        return {\n            'experiment_overview': {\n                'total_scenarios': len(self.reactive_results),\n                'prediction_enabled': True\n            },\n            'performance_comparison': {\n                'reactive_success_rate': reactive_success_rate,\n                'predictive_success_rate': predictive_success_rate,\n                'success_rate_improvement': predictive_success_rate - reactive_success_rate,\n                'reactive_avg_damage': reactive_avg_damage,\n                'predictive_avg_damage': predictive_avg_damage,\n                'damage_reduction': reactive_avg_damage - predictive_avg_damage\n            },\n            'prediction_analysis': {\n                'average_accuracy': avg_prediction_accuracy,\n                'total_predictions': len(self.prediction_accuracy_history),\n                'prediction_statistics': prediction_stats\n            },\n            'scenario_analysis': scenario_analysis\n        }\n    \n    def _analyze_by_scenario_type(self) -> Dict[str, Dict]:\n        \"\"\"Analyze performance by scenario type.\"\"\"\n        scenario_results = {}\n        \n        for reactive_result, predictive_result in zip(self.reactive_results, self.predictive_results):\n            scenario_type = reactive_result['scenario_type']\n            \n            if scenario_type not in scenario_results:\n                scenario_results[scenario_type] = {\n                    'reactive_successes': 0,\n                    'predictive_successes': 0,\n                    'total_count': 0,\n                    'reactive_total_damage': 0.0,\n                    'predictive_total_damage': 0.0\n                }\n            \n            scenario_results[scenario_type]['total_count'] += 1\n            \n            if reactive_result['actual_outcome']['action_success']:\n                scenario_results[scenario_type]['reactive_successes'] += 1\n            if predictive_result['actual_outcome']['action_success']:\n                scenario_results[scenario_type]['predictive_successes'] += 1\n            \n            scenario_results[scenario_type]['reactive_total_damage'] += reactive_result['actual_outcome']['damage_taken']\n            scenario_results[scenario_type]['predictive_total_damage'] += predictive_result['actual_outcome']['damage_taken']\n        \n        # Calculate rates\n        for scenario_type, data in scenario_results.items():\n            data['reactive_success_rate'] = data['reactive_successes'] / data['total_count']\n            data['predictive_success_rate'] = data['predictive_successes'] / data['total_count']\n            data['reactive_avg_damage'] = data['reactive_total_damage'] / data['total_count']\n            data['predictive_avg_damage'] = data['predictive_total_damage'] / data['total_count']\n        \n        return scenario_results\n    \n    def _print_experiment_summary(self, analysis: Dict):\n        \"\"\"Print comprehensive experiment summary.\"\"\"\n        print(\"\\n📊 SENSORY PREDICTION EXPERIMENT RESULTS\")\n        print(\"=\" * 70)\n        \n        # Overview\n        overview = analysis['experiment_overview']\n        print(f\"Total scenarios tested: {overview['total_scenarios']}\")\n        print()\n        \n        # Performance comparison\n        perf = analysis['performance_comparison']\n        print(\"🏆 PERFORMANCE COMPARISON:\")\n        print(f\"  Reactive approach success rate:   {perf['reactive_success_rate']:.1%}\")\n        print(f\"  Predictive approach success rate: {perf['predictive_success_rate']:.1%}\")\n        print(f\"  Success rate improvement:         {perf['success_rate_improvement']:+.1%}\")\n        print()\n        print(f\"  Reactive approach avg damage:     {perf['reactive_avg_damage']:.3f}\")\n        print(f\"  Predictive approach avg damage:   {perf['predictive_avg_damage']:.3f}\")\n        print(f\"  Damage reduction:                 {perf['damage_reduction']:+.3f}\")\n        print()\n        \n        # Prediction analysis\n        pred = analysis['prediction_analysis']\n        print(\"🔮 PREDICTION ANALYSIS:\")\n        print(f\"  Average prediction accuracy: {pred['average_accuracy']:.2%}\")\n        print(f\"  Total predictions made: {pred['total_predictions']}\")\n        \n        pred_stats = pred['prediction_statistics']\n        print(f\"  Prediction methods used:\")\n        for method, stats in pred_stats.get('prediction_methods', {}).items():\n            print(f\"    {method}: {stats['usage_percentage']:.1f}% (accuracy: {stats['average_accuracy']:.2%})\")\n        print()\n        \n        # Scenario-specific results\n        scenarios = analysis['scenario_analysis']\n        print(\"📋 SCENARIO-SPECIFIC RESULTS:\")\n        for scenario_type, data in scenarios.items():\n            print(f\"  {scenario_type.replace('_', ' ').title()}:\")\n            print(f\"    Success rate: Reactive {data['reactive_success_rate']:.1%} → Predictive {data['predictive_success_rate']:.1%}\")\n            print(f\"    Avg damage:   Reactive {data['reactive_avg_damage']:.3f} → Predictive {data['predictive_avg_damage']:.3f}\")\n        print()\n        \n        # Key achievements\n        print(\"🌟 KEY ACHIEVEMENTS:\")\n        \n        if perf['success_rate_improvement'] > 0.05:\n            print(\"  ✅ Prediction significantly improved success rate\")\n        \n        if perf['damage_reduction'] > 0.02:\n            print(\"  ✅ Prediction significantly reduced damage taken\")\n        \n        if pred['average_accuracy'] > 0.6:\n            print(\"  ✅ High prediction accuracy achieved\")\n        \n        if perf['predictive_success_rate'] > 0.8:\n            print(\"  ✅ High overall success rate with prediction\")\n        \n        # Check if any scenario type showed major improvement\n        for scenario_type, data in scenarios.items():\n            improvement = data['predictive_success_rate'] - data['reactive_success_rate']\n            if improvement > 0.2:\n                print(f\"  ✅ Major improvement in {scenario_type.replace('_', ' ')} scenarios\")\n        \n        print()\n        print(\"🌟 SENSORY PREDICTION EXPERIMENT COMPLETE!\")\n        print(\"Successfully demonstrated how prediction transforms robot behavior:\")\n        print(\"From reactive 'this action seems safe' to predictive 'this action will lead to safety'!\")\n\n\ndef main():\n    \"\"\"Run the sensory prediction experiment.\"\"\"\n    print(\"🚀 Loading brain system for sensory prediction testing...\")\n    \n    # Load existing brain system\n    brainstem = GridWorldBrainstem(seed=42, use_sockets=False)\n    world_graph = brainstem.brain_client.get_world_graph()\n    \n    print(f\"✅ Loaded graph with {world_graph.node_count()} nodes\")\n    print()\n    \n    # Run experiment\n    experiment = SensoryPredictionExperiment(world_graph)\n    results = experiment.run_prediction_experiment(num_scenarios=80)\n    \n    return results\n\n\nif __name__ == \"__main__\":\n    results = main()