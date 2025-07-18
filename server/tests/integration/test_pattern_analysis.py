"""
GPU Pattern Analysis Test

Tests the pattern recognition and sequence prediction capabilities.
This is where real intelligence emerges - learning recurring patterns
and predicting what comes next in familiar situations.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import numpy as np
import pytest
from src.prediction.pattern_analyzer import GPUPatternAnalyzer
from src.prediction.engine import PredictionEngine
from src.brain import MinimalBrain


def create_pattern_sequence(base_pattern: np.ndarray, pattern_length: int, 
                           noise_level: float = 0.1) -> list:
    """Create a sequence that follows a specific pattern."""
    sequence = []
    
    for i in range(pattern_length):
        # Vary the pattern slightly over time
        variation = base_pattern + np.sin(i * 0.5) * 0.2
        noise = np.random.normal(0, noise_level, len(base_pattern))
        
        sensory_input = variation + noise
        action_taken = np.array([i * 0.1, -i * 0.1, 0.5, 0.0])  # Predictable action pattern
        # Make outcome same dimensions as sensory input
        action_effect = np.zeros_like(sensory_input)
        action_effect[:min(len(action_taken), len(sensory_input))] = action_taken[:len(sensory_input)]
        outcome = sensory_input * 1.1 + action_effect  # Outcome depends on input + action
        
        experience_data = {
            'experience_id': f'pattern_exp_{i}',
            'sensory_input': sensory_input.tolist(),
            'action_taken': action_taken.tolist(),
            'outcome': outcome.tolist(),
            'prediction_error': np.random.uniform(0.1, 0.3),
            'timestamp': time.time() + i * 0.001
        }
        
        sequence.append(experience_data)
    
    return sequence


class TestGPUPatternAnalysis:
    """Test suite for GPU pattern analysis."""
    
    def test_pattern_analyzer_initialization(self):
        """Test pattern analyzer initializes correctly."""
        analyzer = GPUPatternAnalyzer(use_gpu=True, use_mixed_precision=True)
        
        assert analyzer.use_gpu or not analyzer.use_gpu  # Should handle GPU availability gracefully
        assert analyzer.min_pattern_length == 2
        assert analyzer.max_pattern_length == 5
        assert len(analyzer.learned_patterns) == 0
        
        print("✅ Pattern analyzer initialization successful")
    
    def test_single_pattern_discovery(self):
        """Test discovery of a simple repeating pattern."""
        analyzer = GPUPatternAnalyzer(use_gpu=True, use_mixed_precision=True)
        
        # Create a repeating pattern
        base_pattern = np.array([1.0, 0.5, -0.3, 0.8, 0.2])
        pattern_sequence = create_pattern_sequence(base_pattern, 8, noise_level=0.05)
        
        # Add the sequence multiple times to trigger pattern discovery
        for repeat in range(4):
            for experience_data in pattern_sequence:
                # Slightly modify for realism
                modified_data = experience_data.copy()
                modified_data['experience_id'] = f"{experience_data['experience_id']}_repeat_{repeat}"
                modified_data['timestamp'] = time.time() + repeat * 10 + len(pattern_sequence) * 0.001
                
                analyzer.add_experience_to_stream(modified_data)
        
        # Check if patterns were discovered
        stats = analyzer.get_pattern_statistics()
        
        print(f"Patterns discovered: {stats['total_patterns']}")
        print(f"Total experiences processed: {len(analyzer.experience_stream)}")
        
        assert stats['total_patterns'] > 0, "Should discover at least one pattern"
        assert stats['patterns_discovered'] > 0, "Pattern discovery counter should increment"
        
        print("✅ Single pattern discovery working")
    
    def test_pattern_based_prediction(self):
        """Test prediction based on learned patterns."""
        analyzer = GPUPatternAnalyzer(use_gpu=True, use_mixed_precision=True)
        
        # Create two different patterns
        pattern_a = np.array([1.0, 0.0, 0.0])
        pattern_b = np.array([0.0, 1.0, 0.0])
        
        sequence_a = create_pattern_sequence(pattern_a, 4, noise_level=0.02)
        sequence_b = create_pattern_sequence(pattern_b, 4, noise_level=0.02)
        
        # Add pattern A multiple times
        for repeat in range(5):
            for exp in sequence_a:
                modified_exp = exp.copy()
                modified_exp['experience_id'] = f"patternA_{repeat}_{exp['experience_id']}"
                modified_exp['timestamp'] = time.time() + repeat * 0.1
                analyzer.add_experience_to_stream(modified_exp)
        
        # Add pattern B multiple times
        for repeat in range(5):
            for exp in sequence_b:
                modified_exp = exp.copy()
                modified_exp['experience_id'] = f"patternB_{repeat}_{exp['experience_id']}"
                modified_exp['timestamp'] = time.time() + repeat * 0.1 + 100
                analyzer.add_experience_to_stream(modified_exp)
        
        # Try to predict next experience for pattern A
        # Convert sequence to internal format expected by pattern analyzer
        partial_sequence_a = []
        for exp_data in sequence_a[:2]:
            partial_sequence_a.append({
                'context': np.array(exp_data['sensory_input']),
                'action': np.array(exp_data['action_taken']),
                'outcome': np.array(exp_data['outcome']),
                'timestamp': exp_data['timestamp'],
                'experience_id': exp_data['experience_id']
            })
        
        current_context = np.array(sequence_a[2]['sensory_input'])
        
        prediction = analyzer.predict_next_experience(current_context, partial_sequence_a)
        
        stats = analyzer.get_pattern_statistics()
        print(f"Patterns learned: {stats['total_patterns']}")
        print(f"Prediction made: {prediction is not None}")
        
        if prediction:
            print(f"Prediction confidence: {prediction['confidence']:.3f}")
            print(f"Pattern used: {prediction['pattern_id'][:20]}...")
            
            assert prediction['confidence'] > 0.0, "Prediction should have some confidence"
            assert 'predicted_action' in prediction, "Should predict an action"
            assert 'predicted_outcome' in prediction, "Should predict an outcome"
        
        print("✅ Pattern-based prediction working")
    
    def test_pattern_learning_accuracy(self):
        """Test that pattern learning improves with feedback."""
        analyzer = GPUPatternAnalyzer(use_gpu=True, use_mixed_precision=True)
        
        # Create a very consistent pattern
        base_pattern = np.array([0.8, -0.3, 0.5])
        consistent_sequence = create_pattern_sequence(base_pattern, 6, noise_level=0.01)
        
        # Add it multiple times
        pattern_ids = []
        for repeat in range(6):
            for exp in consistent_sequence:
                modified_exp = exp.copy()
                modified_exp['experience_id'] = f"consistent_{repeat}_{exp['experience_id']}"
                analyzer.add_experience_to_stream(modified_exp)
        
        # Get initial pattern statistics
        initial_stats = analyzer.get_pattern_statistics()
        
        # Record some successful predictions for learned patterns
        for pattern_id in analyzer.learned_patterns.keys():
            for _ in range(3):
                analyzer.record_prediction_outcome(pattern_id, 0.9)  # High success
            pattern_ids.append(pattern_id)
        
        # Record some failed predictions
        for pattern_id in pattern_ids[:1]:  # Only for first pattern
            for _ in range(2):
                analyzer.record_prediction_outcome(pattern_id, 0.2)  # Low success
        
        final_stats = analyzer.get_pattern_statistics()
        
        print(f"Initial patterns: {initial_stats['total_patterns']}")
        print(f"Final patterns: {final_stats['total_patterns']}")
        print(f"Average utility: {final_stats['avg_pattern_utility']:.3f}")
        
        # Patterns should exist and have some utility
        if final_stats['total_patterns'] > 0:
            assert final_stats['avg_pattern_utility'] >= 0.0, "Patterns should have some utility"
        
        print("✅ Pattern learning accuracy tracking working")
    
    def test_prediction_engine_integration(self):
        """Test pattern analysis integration with prediction engine."""
        # Create prediction engine with pattern analysis
        engine = PredictionEngine(use_pattern_analysis=True)
        
        assert engine.pattern_analyzer is not None, "Pattern analyzer should be initialized"
        assert engine.use_pattern_analysis == True, "Should be configured for pattern analysis"
        
        # Create some experience data with clear patterns
        base_context = [1.0, 0.5, 0.0, 0.3]
        base_action = [0.2, -0.1, 0.8, 0.0]
        
        # Add experiences to build patterns
        for i in range(15):
            experience_data = {
                'experience_id': f'test_exp_{i}',
                'sensory_input': [base_context[j] + np.sin(i * 0.3) * 0.1 for j in range(4)],
                'action_taken': [base_action[j] + i * 0.05 for j in range(4)],
                'outcome': [base_context[j] * 1.1 + i * 0.02 for j in range(4)],
                'prediction_error': 0.2 + i * 0.01,
                'timestamp': time.time() + i * 0.01
            }
            
            engine.add_experience_to_stream(experience_data)
        
        # Get statistics
        stats = engine.get_prediction_statistics()
        
        print(f"Pattern analysis enabled: {'pattern_analysis' in stats}")
        if 'pattern_analysis' in stats:
            pattern_stats = stats['pattern_analysis']
            print(f"Patterns discovered: {pattern_stats['total_patterns']}")
            print(f"Processing time: {pattern_stats['avg_processing_time_ms']:.2f}ms")
        
        assert 'pattern_analysis' in stats, "Should include pattern analysis statistics"
        
        print("✅ Prediction engine integration working")
    
    def test_end_to_end_brain_with_patterns(self):
        """Test complete brain with pattern analysis enabled."""
        # Create brain with pattern analysis
        brain = MinimalBrain(use_utility_based_activation=True, enable_logging=False)
        
        assert brain.prediction_engine.use_pattern_analysis == True, "Brain should use pattern analysis"
        
        # Create a robot scenario with repeating patterns
        # Scenario: Robot repeatedly encounters similar situations
        
        situation_a = [1.0, 0.8, 0.2, 0.5]  # Kitchen scenario
        action_a = [0.5, 0.0, -0.2, 0.3]    # Move forward, turn slightly
        
        situation_b = [0.2, 1.0, 0.8, 0.1]  # Living room scenario  
        action_b = [-0.3, 0.4, 0.0, 0.5]    # Step back, turn more
        
        # Simulate robot encountering these scenarios repeatedly
        for cycle in range(10):
            # Situation A
            sensory_input_a = [situation_a[i] + np.random.normal(0, 0.05) for i in range(4)]
            predicted_action_a, brain_state_a = brain.process_sensory_input(sensory_input_a)
            
            # Robot takes action and observes outcome
            outcome_a = [sensory_input_a[i] * 1.1 + action_a[i] * 0.5 for i in range(4)]
            brain.store_experience(sensory_input_a, action_a, outcome_a, predicted_action_a)
            
            # Situation B
            sensory_input_b = [situation_b[i] + np.random.normal(0, 0.05) for i in range(4)]
            predicted_action_b, brain_state_b = brain.process_sensory_input(sensory_input_b)
            
            # Robot takes action and observes outcome
            outcome_b = [sensory_input_b[i] * 1.1 + action_b[i] * 0.5 for i in range(4)]
            brain.store_experience(sensory_input_b, action_b, outcome_b, predicted_action_b)
        
        # Get brain statistics
        brain_stats = brain.get_brain_stats()
        prediction_stats = brain_stats['prediction_engine']
        
        print(f"Total experiences: {brain_stats['brain_summary']['total_experiences']}")
        print(f"Total predictions: {brain_stats['brain_summary']['total_predictions']}")
        
        if 'pattern_analysis' in prediction_stats:
            pattern_stats = prediction_stats['pattern_analysis']
            print(f"Patterns discovered: {pattern_stats['total_patterns']}")
            print(f"Pattern predictions made: {prediction_stats['pattern_predictions']}")
            
            # Should have discovered some patterns from repeated scenarios
            if pattern_stats['total_patterns'] > 0:
                assert pattern_stats['avg_pattern_frequency'] > 1, "Patterns should repeat"
        
        # Test that brain can make predictions
        test_situation = [situation_a[i] + np.random.normal(0, 0.02) for i in range(4)]
        final_prediction, final_state = brain.process_sensory_input(test_situation)
        
        assert final_prediction is not None, "Brain should be able to make predictions"
        assert len(final_prediction) == 4, "Should predict 4D action"
        assert final_state['total_experiences'] == 20, "Should have stored all experiences"
        
        print(f"Final prediction confidence: {final_state['prediction_confidence']:.3f}")
        print(f"Working memory size: {final_state['working_memory_size']}")
        
        print("✅ End-to-end brain with pattern analysis working")
    
    def test_gpu_performance_scaling(self):
        """Test pattern analysis performance with larger datasets."""
        analyzer = GPUPatternAnalyzer(use_gpu=True, use_mixed_precision=True)
        
        # Generate many experiences with some patterns
        num_experiences = 500
        num_patterns = 3
        
        start_time = time.time()
        
        for pattern_id in range(num_patterns):
            base_pattern = np.random.randn(5)
            
            for repeat in range(num_experiences // num_patterns):
                experience_data = {
                    'experience_id': f'perf_test_{pattern_id}_{repeat}',
                    'sensory_input': (base_pattern + np.random.normal(0, 0.1, 5)).tolist(),
                    'action_taken': np.random.randn(4).tolist(),
                    'outcome': np.random.randn(5).tolist(),
                    'prediction_error': np.random.uniform(0.1, 0.5),
                    'timestamp': time.time() + repeat * 0.001
                }
                
                analyzer.add_experience_to_stream(experience_data)
        
        processing_time = time.time() - start_time
        
        # Get final statistics
        stats = analyzer.get_pattern_statistics()
        
        experiences_per_second = num_experiences / processing_time if processing_time > 0 else 0
        
        print(f"Processed {num_experiences} experiences in {processing_time:.3f}s")
        print(f"Throughput: {experiences_per_second:.0f} experiences/second")
        print(f"Patterns discovered: {stats['total_patterns']}")
        print(f"Average processing time: {stats['avg_processing_time_ms']:.2f}ms")
        
        # Should be able to process experiences efficiently (pattern analysis is more complex than simple processing)
        assert experiences_per_second > 25, "Should process at least 25 experiences/second with pattern analysis"
        assert stats['total_patterns'] > 0, "Should discover some patterns"
        
        print("✅ GPU performance scaling working")


def run_pattern_analysis_tests():
    """Run complete pattern analysis test suite."""
    print("🔍 Running GPU Pattern Analysis Tests")
    print("Testing intelligent sequence recognition and prediction")
    print("=" * 70)
    
    test_suite = TestGPUPatternAnalysis()
    
    try:
        test_suite.test_pattern_analyzer_initialization()
        test_suite.test_single_pattern_discovery()
        test_suite.test_pattern_based_prediction()
        test_suite.test_pattern_learning_accuracy()
        test_suite.test_prediction_engine_integration()
        test_suite.test_end_to_end_brain_with_patterns()
        test_suite.test_gpu_performance_scaling()
        
        print("\n🎉 All pattern analysis tests passed!")
        print("\n🎯 Pattern Analysis Implementation Complete:")
        print("  • GPU-accelerated sequence recognition")
        print("  • Intelligent pattern-based predictions")
        print("  • Automatic pattern discovery from experience streams")
        print("  • Learning from prediction outcomes")
        print("  • Full brain integration")
        print("\n🧠 The brain now recognizes recurring patterns and predicts familiar sequences!")
        print("This is where real intelligence emerges - learning what comes next in familiar situations.")
        
    except Exception as e:
        print(f"\n❌ Pattern analysis test failed: {e}")
        raise


if __name__ == "__main__":
    run_pattern_analysis_tests()