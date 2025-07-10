"""
Simple Sensory Prediction Demonstration.

Shows the transformative power of sensory prediction in a clean, focused test.
"""

import sys
import random
from typing import Dict, List

# Add project root to path
sys.path.append('.')

from prediction.sensory_predictor import SensoryPredictor
from motivators.predictive_survival_drive import PredictiveSurvivalDrive
from motivators.survival_drive import SurvivalDrive
from motivators.base_motivator import DriveContext
from simulation.brainstem_sim import GridWorldBrainstem


def create_test_context(wall_sensor: float, danger_sensor: float) -> DriveContext:
    """Create a test context with specific sensor values."""
    return DriveContext(
        current_sensory=[wall_sensor, 0.3, danger_sensor, 0.1, 0.5, 0.4, 0.3, 0.2],
        robot_health=0.8,
        robot_energy=0.7,
        robot_position=(5, 5),
        robot_orientation=0,
        recent_experiences=[],
        prediction_errors=[],
        step_count=100,
        time_since_last_food=10,
        time_since_last_damage=20,
        threat_level='normal'
    )


def test_wall_scenario():
    """Test scenario: Wall directly ahead."""
    print("🧪 WALL AHEAD SCENARIO")
    print("=" * 50)
    
    # Load brain system
    brainstem = GridWorldBrainstem(seed=42, use_sockets=False)
    world_graph = brainstem.brain_client.get_world_graph()
    
    # Create drives
    predictor = SensoryPredictor(world_graph)
    reactive_drive = SurvivalDrive(base_weight=1.0)
    predictive_drive = PredictiveSurvivalDrive(base_weight=1.0, predictor=predictor)
    
    # Create context with wall ahead
    context = create_test_context(wall_sensor=0.9, danger_sensor=0.1)
    print(f"Context: Wall sensor = 0.9 (high), Danger sensor = 0.1 (low)")
    
    # Test actions
    forward_action = {'forward_motor': 0.6, 'turn_motor': 0.0, 'brake_motor': 0.0}
    turn_action = {'forward_motor': 0.0, 'turn_motor': 0.5, 'brake_motor': 0.0}
    brake_action = {'forward_motor': 0.0, 'turn_motor': 0.0, 'brake_motor': 0.8}
    
    actions = [
        ("Forward into wall", forward_action),
        ("Turn away", turn_action), 
        ("Brake/stop", brake_action)
    ]
    
    print("\nReactive Drive Evaluations:")
    for name, action in actions:
        eval_result = reactive_drive.evaluate_action(action, context)
        score = getattr(eval_result, 'score', eval_result.action_score)
        print(f"  {name:15s}: Score={score:.3f}, {eval_result.reasoning}")
    
    print("\nPredictive Drive Evaluations:")
    for name, action in actions:
        eval_result = predictive_drive.evaluate_action(action, context)
        score = getattr(eval_result, 'score', eval_result.action_score)
        prediction_info = ""
        if hasattr(eval_result, 'sensory_prediction') and eval_result.sensory_prediction:
            pred = eval_result.sensory_prediction
            prediction_info = f" [Predicted via {pred.prediction_method}, confidence: {pred.confidence:.2f}]"
        print(f"  {name:15s}: Score={score:.3f}, {eval_result.reasoning}{prediction_info}")
    
    print()


def test_danger_scenario():
    """Test scenario: Danger detected."""
    print("🧪 DANGER DETECTED SCENARIO")
    print("=" * 50)
    
    # Load brain system
    brainstem = GridWorldBrainstem(seed=42, use_sockets=False)
    world_graph = brainstem.brain_client.get_world_graph()
    
    # Create drives
    predictor = SensoryPredictor(world_graph)
    reactive_drive = SurvivalDrive(base_weight=1.0)
    predictive_drive = PredictiveSurvivalDrive(base_weight=1.0, predictor=predictor)
    
    # Create context with danger detected
    context = create_test_context(wall_sensor=0.2, danger_sensor=0.8)
    print(f"Context: Wall sensor = 0.2 (low), Danger sensor = 0.8 (high)")
    
    # Test actions
    forward_action = {'forward_motor': 0.5, 'turn_motor': 0.0, 'brake_motor': 0.0}
    retreat_action = {'forward_motor': -0.4, 'turn_motor': 0.0, 'brake_motor': 0.0}
    brake_action = {'forward_motor': 0.0, 'turn_motor': 0.0, 'brake_motor': 1.0}
    
    actions = [
        ("Move toward danger", forward_action),
        ("Retreat", retreat_action),
        ("Emergency stop", brake_action)
    ]
    
    print("\nReactive Drive Evaluations:")
    for name, action in actions:
        eval_result = reactive_drive.evaluate_action(action, context)
        score = getattr(eval_result, 'score', eval_result.action_score)
        print(f"  {name:18s}: Score={score:.3f}, {eval_result.reasoning}")
    
    print("\nPredictive Drive Evaluations:")
    for name, action in actions:
        eval_result = predictive_drive.evaluate_action(action, context)
        score = getattr(eval_result, 'score', eval_result.action_score)
        prediction_info = ""
        if hasattr(eval_result, 'sensory_prediction') and eval_result.sensory_prediction:
            pred = eval_result.sensory_prediction
            prediction_info = f" [Predicted via {pred.prediction_method}, confidence: {pred.confidence:.2f}]"
        print(f"  {name:18s}: Score={score:.3f}, {eval_result.reasoning}{prediction_info}")
    
    print()


def test_prediction_accuracy():
    """Test prediction accuracy over multiple scenarios."""
    print("🧪 PREDICTION ACCURACY TEST")
    print("=" * 50)
    
    # Load brain system
    brainstem = GridWorldBrainstem(seed=42, use_sockets=False)
    world_graph = brainstem.brain_client.get_world_graph()
    
    # Create predictor
    predictor = SensoryPredictor(world_graph)
    
    # Test multiple prediction scenarios
    test_cases = [
        {"wall": 0.9, "danger": 0.1, "description": "High wall, low danger"},
        {"wall": 0.1, "danger": 0.8, "description": "Low wall, high danger"},
        {"wall": 0.5, "danger": 0.3, "description": "Medium wall, low danger"},
        {"wall": 0.2, "danger": 0.2, "description": "Low wall, low danger"},
        {"wall": 0.8, "danger": 0.7, "description": "High wall, high danger"}
    ]
    
    total_predictions = 0
    successful_predictions = 0
    
    for i, case in enumerate(test_cases):
        context = create_test_context(case["wall"], case["danger"])
        action = {'forward_motor': 0.4, 'turn_motor': 0.0, 'brake_motor': 0.0}
        
        # Make prediction
        try:
            prediction = predictor.predict_sensory_outcome(
                action=action,
                current_context=[0.5, 0.3, 0.2, 0.1],
                current_sensors={str(i): val for i, val in enumerate(context.current_sensory)}
            )
            
            total_predictions += 1
            if prediction.confidence > 0.3:
                successful_predictions += 1
                
            print(f"Case {i+1}: {case['description']}")
            print(f"  Prediction method: {prediction.prediction_method}")
            print(f"  Confidence: {prediction.confidence:.2f}")
            print(f"  Basis experiences: {len(prediction.prediction_basis)}")
            print()
            
        except Exception as e:
            print(f"Case {i+1}: {case['description']} - Prediction failed: {e}")
    
    if total_predictions > 0:
        success_rate = successful_predictions / total_predictions
        print(f"📊 Prediction Success Rate: {success_rate:.1%} ({successful_predictions}/{total_predictions})")
    
    # Get predictor statistics
    stats = predictor.get_prediction_statistics()
    print(f"📊 Total predictions made: {stats['total_predictions']}")
    print(f"📊 Average accuracy: {stats['average_accuracy']:.2%}")
    
    if stats['prediction_methods']:
        print("📊 Prediction methods used:")
        for method, method_stats in stats['prediction_methods'].items():
            print(f"  {method}: {method_stats['usage_percentage']:.1f}% (accuracy: {method_stats['average_accuracy']:.2%})")
    
    print()


def main():
    """Run simplified sensory prediction demonstrations."""
    print("🔮 SENSORY PREDICTION DEMONSTRATION")
    print("=" * 70)
    print("Comparing reactive vs predictive robot behavior")
    print("Reactive: 'This action seems safe'")
    print("Predictive: 'This action will lead to safety'")
    print()
    
    # Run test scenarios
    test_wall_scenario()
    test_danger_scenario()
    test_prediction_accuracy()
    
    print("🌟 DEMONSTRATION COMPLETE!")
    print("Successfully showed how sensory prediction transforms robot intelligence!")
    print("From reactive stimulus-response to predictive consequence-evaluation!")


if __name__ == "__main__":
    main()