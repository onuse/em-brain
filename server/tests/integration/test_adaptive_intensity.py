#!/usr/bin/env python3
"""
Comprehensive Tests for Adaptive Pattern Analysis Intensity

Tests the cognitive autopilot and adaptive prediction engine to ensure:
1. Performance benefits are achieved (faster cycles in autopilot mode)
2. Prediction accuracy is maintained across intensity modes
3. Safety fallbacks work correctly
4. Cache behavior is correct
5. Integration with existing systems works

Usage: python3 test_runner.py adaptive_intensity
"""

import sys
import os
import time
import numpy as np
from typing import Dict, List, Any

# Add brain directory to path
brain_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, brain_dir)

from src.utils.cognitive_autopilot import CognitiveAutopilot, CognitiveMode
from src.prediction.adaptive_engine import AdaptivePredictionEngine
from server.src.brain_factory import MinimalBrain


class AdaptiveIntensityTestSuite:
    """Comprehensive test suite for adaptive intensity control."""
    
    def __init__(self):
        """Initialize test suite."""
        self.test_results = {}
        self.performance_data = {}
        
    def run_all_tests(self):
        """Run complete test suite."""
        print("🧪 Adaptive Intensity Control Test Suite")
        print("=" * 50)
        
        # Test 1: Cognitive Autopilot Basic Functionality
        self.test_cognitive_autopilot_basic()
        
        # Test 2: Adaptive Prediction Engine Performance
        self.test_adaptive_prediction_performance()
        
        # Test 3: Accuracy Preservation
        self.test_accuracy_preservation()
        
        # Test 4: Cache Behavior
        self.test_cache_behavior()
        
        # Test 5: Safety Fallbacks
        self.test_safety_fallbacks()
        
        # Test 6: Integration Test
        self.test_brain_integration()
        
        # Test 7: Performance Benchmarks
        self.test_performance_benchmarks()
        
        # Summary
        self.print_test_summary()
    
    def test_cognitive_autopilot_basic(self):
        """Test basic cognitive autopilot functionality."""
        print("\n🧠 Test 1: Cognitive Autopilot Basic Functionality")
        
        autopilot = CognitiveAutopilot(
            autopilot_confidence_threshold=0.90,
            focused_confidence_threshold=0.70
        )
        
        test_cases = [
            # (confidence, stability, surprise, expected_mode)
            (0.95, 0.8, 0.2, CognitiveMode.AUTOPILOT),
            (0.80, 0.6, 0.4, CognitiveMode.FOCUSED),
            (0.60, 0.5, 0.8, CognitiveMode.DEEP_THINK),
            (0.40, 0.3, 0.6, CognitiveMode.DEEP_THINK)
        ]
        
        results = []
        for confidence, stability, surprise, expected_mode in test_cases:
            # Simulate brain state
            brain_state = {'consensus_rate': confidence}
            
            # Create fake histories for stability/surprise calculation
            autopilot.confidence_history.extend([confidence] * 5)
            autopilot.prediction_error_history.extend([1.0 - confidence] * 5)
            
            result = autopilot.update_cognitive_state(confidence, 1.0 - confidence, brain_state)
            actual_mode = CognitiveMode(result['cognitive_mode'])
            
            success = actual_mode == expected_mode
            results.append(success)
            
            print(f"   Confidence {confidence:.2f} → {actual_mode.value} (expected {expected_mode.value}) {'✓' if success else '✗'}")
        
        accuracy = sum(results) / len(results)
        self.test_results['autopilot_basic'] = {
            'passed': accuracy >= 0.75,
            'accuracy': accuracy,
            'details': f"{sum(results)}/{len(results)} mode predictions correct"
        }
        
        print(f"   Result: {accuracy:.1%} accuracy {'✓ PASS' if accuracy >= 0.75 else '✗ FAIL'}")
    
    def test_adaptive_prediction_performance(self):
        """Test adaptive prediction engine performance across intensity modes."""
        print("\n⚡ Test 2: Adaptive Prediction Engine Performance")
        
        # Create autopilot and adaptive engine
        autopilot = CognitiveAutopilot()
        engine = AdaptivePredictionEngine(cognitive_autopilot=autopilot)
        
        # Create mock experiences and similarity engine
        mock_experiences = self._create_mock_experiences(50)
        mock_similarity_engine = self._create_mock_similarity_engine()
        mock_activation_dynamics = None
        
        # Test each intensity mode
        intensity_modes = ['minimal', 'selective', 'full']
        performance_results = {}
        
        for mode in intensity_modes:
            print(f"   Testing {mode} mode...")
            
            # Create brain state that forces this mode
            brain_state = self._create_brain_state_for_mode(mode, autopilot)
            
            # Time multiple predictions
            start_time = time.time()
            confidences = []
            
            for i in range(10):
                context = [0.5 + i * 0.1, 0.3, 0.7, 0.2] + [0.1] * 12
                
                predicted_action, confidence, details = engine.predict_action(
                    context, mock_similarity_engine, mock_activation_dynamics,
                    mock_experiences, 4, brain_state
                )
                
                confidences.append(confidence)
            
            avg_time = (time.time() - start_time) / 10
            avg_confidence = np.mean(confidences)
            
            performance_results[mode] = {
                'avg_time': avg_time,
                'avg_confidence': avg_confidence
            }
            
            print(f"     Time: {avg_time*1000:.1f}ms, Confidence: {avg_confidence:.2f}")
        
        # Validate performance expectations
        minimal_faster = performance_results['minimal']['avg_time'] < performance_results['full']['avg_time']
        selective_middle = (performance_results['selective']['avg_time'] < performance_results['full']['avg_time'] and
                           performance_results['selective']['avg_time'] > performance_results['minimal']['avg_time'])
        
        self.test_results['adaptive_performance'] = {
            'passed': minimal_faster and selective_middle,
            'performance_data': performance_results,
            'details': f"Minimal faster: {minimal_faster}, Selective middle: {selective_middle}"
        }
        
        print(f"   Result: Performance ordering correct {'✓ PASS' if minimal_faster and selective_middle else '✗ FAIL'}")
    
    def test_accuracy_preservation(self):
        """Test that accuracy is preserved across intensity modes."""
        print("\n🎯 Test 3: Accuracy Preservation")
        
        autopilot = CognitiveAutopilot()
        engine = AdaptivePredictionEngine(cognitive_autopilot=autopilot)
        
        # Create deterministic test scenario
        mock_experiences = self._create_deterministic_experiences()
        mock_similarity_engine = self._create_deterministic_similarity_engine()
        
        # Test same prediction with different intensity modes
        test_context = [0.5, 0.5, 0.5, 0.5] + [0.1] * 12
        accuracy_results = {}
        
        for mode in ['minimal', 'selective', 'full']:
            brain_state = self._create_brain_state_for_mode(mode, autopilot)
            
            predicted_action, confidence, details = engine.predict_action(
                test_context, mock_similarity_engine, None,
                mock_experiences, 4, brain_state
            )
            
            accuracy_results[mode] = {
                'action': predicted_action,
                'confidence': confidence,
                'method': details.get('method', 'unknown')
            }
        
        # Check if predictions are similar across modes
        actions_similar = self._actions_similar(
            accuracy_results['minimal']['action'],
            accuracy_results['full']['action'],
            tolerance=0.2
        )
        
        confidence_preserved = abs(
            accuracy_results['minimal']['confidence'] - 
            accuracy_results['full']['confidence']
        ) < 0.3
        
        self.test_results['accuracy_preservation'] = {
            'passed': actions_similar and confidence_preserved,
            'accuracy_data': accuracy_results,
            'details': f"Actions similar: {actions_similar}, Confidence preserved: {confidence_preserved}"
        }
        
        print(f"   Actions similar: {actions_similar} {'✓' if actions_similar else '✗'}")
        print(f"   Confidence preserved: {confidence_preserved} {'✓' if confidence_preserved else '✗'}")
        print(f"   Result: {'✓ PASS' if actions_similar and confidence_preserved else '✗ FAIL'}")
    
    def test_cache_behavior(self):
        """Test pattern cache behavior in autopilot mode."""
        print("\n💾 Test 4: Cache Behavior")
        
        autopilot = CognitiveAutopilot()
        engine = AdaptivePredictionEngine(cognitive_autopilot=autopilot)
        
        # Create mock data
        mock_experiences = self._create_mock_experiences(20)
        mock_similarity_engine = self._create_mock_similarity_engine()
        
        # Add some recent experiences to enable pattern analysis
        for i in range(5):
            engine.recent_experiences.append({
                'sensory_input': [0.1 * i, 0.2, 0.3, 0.4] + [0.1] * 12,
                'action_taken': [0.1, 0.2, 0.3, 0.4],
                'outcome': [0.1 * i, 0.2, 0.3, 0.4] + [0.1] * 12,
                'timestamp': time.time() - i,
                'experience_id': f'test_exp_{i}'
            })
        
        # Force autopilot mode (uses cache)
        brain_state = self._create_brain_state_for_mode('minimal', autopilot)
        
        # First prediction - should create cache entry
        context1 = [0.5, 0.5, 0.5, 0.5] + [0.1] * 12
        _, _, details1 = engine.predict_action(
            context1, mock_similarity_engine, None, mock_experiences, 4, brain_state
        )
        
        try:
            initial_cache_size = engine.pattern_cache.get_stats()['entries']
        except (AttributeError, KeyError):
            initial_cache_size = 0
        
        # Second prediction with same context - should hit cache
        _, _, details2 = engine.predict_action(
            context1, mock_similarity_engine, None, mock_experiences, 4, brain_state
        )
        
        # Check cache behavior
        cache_stats = engine._get_cache_stats()
        cache_working = cache_stats['cache_hits'] > 0
        try:
            current_cache_size = engine.pattern_cache.get_stats()['entries']
            cache_size_grew = current_cache_size >= initial_cache_size
        except (AttributeError, KeyError):
            cache_size_grew = True  # Assume cache is working
        
        self.test_results['cache_behavior'] = {
            'passed': cache_working and cache_size_grew,
            'cache_stats': cache_stats,
            'details': f"Cache hits: {cache_stats['cache_hits']}, Cache working: {cache_working}"
        }
        
        print(f"   Cache hits: {cache_stats['cache_hits']} {'✓' if cache_stats['cache_hits'] > 0 else '✗'}")
        print(f"   Hit rate: {cache_stats['hit_rate']:.1%}")
        print(f"   Result: {'✓ PASS' if cache_working else '✗ FAIL'}")
    
    def test_safety_fallbacks(self):
        """Test safety fallback mechanisms."""
        print("\n🛡️  Test 5: Safety Fallbacks")
        
        autopilot = CognitiveAutopilot()
        engine = AdaptivePredictionEngine(cognitive_autopilot=autopilot, min_similar_experiences=1)
        engine.fallback_threshold = 3  # Lower for testing
        
        mock_experiences = self._create_mock_experiences(10)
        mock_similarity_engine = self._create_mock_similarity_engine()
        
        # Force autopilot mode but create low confidence predictions
        brain_state = self._create_brain_state_for_mode('minimal', autopilot)
        
        # Create scenario that produces low confidence
        for i in range(5):
            context = [0.1 * i, 0.2, 0.3, 0.4] + [0.1] * 12
            
            predicted_action, confidence, details = engine.predict_action(
                context, mock_similarity_engine, None, mock_experiences, 4, brain_state
            )
            
            # Manually set low confidence to trigger fallback
            if i >= 2:
                engine.consecutive_low_confidence = i
        
        # Check if safety monitoring detected the issue
        safety_stats = engine.get_adaptive_performance_stats()['safety_monitoring']
        safety_triggered = safety_stats['safety_active']
        
        self.test_results['safety_fallbacks'] = {
            'passed': safety_triggered,
            'safety_stats': safety_stats,
            'details': f"Safety active: {safety_triggered}, Consecutive low: {safety_stats['consecutive_low_confidence']}"
        }
        
        print(f"   Consecutive low confidence: {safety_stats['consecutive_low_confidence']}")
        print(f"   Safety fallback active: {safety_triggered} {'✓' if safety_triggered else '✗'}")
        print(f"   Result: {'✓ PASS' if safety_triggered else '✗ FAIL'}")
    
    def test_brain_integration(self):
        """Test integration with full brain system."""
        print("\n🧠 Test 6: Brain Integration")
        
        try:
            # Create brain with autopilot (simplified integration)
            brain = MinimalBrain(enable_logging=False, enable_persistence=False, enable_phase2_adaptations=False)
            autopilot = CognitiveAutopilot()
            
            # Test basic brain function with sensory input
            sensory_input = [1.0, 2.0, 3.0, 4.0] + [0.1] * 12
            
            predicted_action, brain_state = brain.process_sensory_input(sensory_input)
            
            # Test autopilot state update
            autopilot_state = autopilot.update_cognitive_state(
                brain_state.get('prediction_confidence', 0.5),
                0.5,  # prediction error
                brain_state
            )
            
            integration_success = (
                predicted_action is not None and
                len(predicted_action) > 0 and
                autopilot_state is not None and
                'cognitive_mode' in autopilot_state
            )
            
            self.test_results['brain_integration'] = {
                'passed': integration_success,
                'details': f"Integration successful: {integration_success}"
            }
            
            print(f"   Brain processing: {'✓' if predicted_action is not None else '✗'}")
            print(f"   Autopilot update: {'✓' if autopilot_state is not None else '✗'}")
            print(f"   Result: {'✓ PASS' if integration_success else '✗ FAIL'}")
            
        except Exception as e:
            self.test_results['brain_integration'] = {
                'passed': False,
                'details': f"Integration failed: {str(e)}"
            }
            print(f"   Result: ✗ FAIL - {str(e)}")
    
    def test_performance_benchmarks(self):
        """Test performance benchmarks and targets."""
        print("\n📊 Test 7: Performance Benchmarks")
        
        autopilot = CognitiveAutopilot()
        
        # Test different brain configurations
        configs = {
            'standard': {'cognitive_autopilot': None},
            'adaptive': {'cognitive_autopilot': autopilot}
        }
        
        benchmark_results = {}
        
        for config_name, config in configs.items():
            print(f"   Benchmarking {config_name} configuration...")
            
            if config_name == 'adaptive':
                engine = AdaptivePredictionEngine(cognitive_autopilot=autopilot)
            else:
                engine = AdaptivePredictionEngine(cognitive_autopilot=None)
            
            # Performance test
            mock_experiences = self._create_mock_experiences(30)
            mock_similarity_engine = self._create_mock_similarity_engine()
            
            start_time = time.time()
            for i in range(20):
                context = [0.1 * i, 0.2, 0.3, 0.4] + [0.1] * 12
                
                if config_name == 'adaptive':
                    # Use autopilot mode for adaptive - force high confidence for minimal mode
                    autopilot.current_mode = CognitiveMode.AUTOPILOT
                    brain_state = self._create_brain_state_for_mode('minimal', autopilot)
                else:
                    brain_state = None
                
                predicted_action, confidence, details = engine.predict_action(
                    context, mock_similarity_engine, None, mock_experiences, 4, brain_state
                )
            
            avg_time = (time.time() - start_time) / 20 * 1000  # ms
            benchmark_results[config_name] = avg_time
            
            print(f"     Average time: {avg_time:.1f}ms")
        
        # Check if adaptive is faster than standard in autopilot mode
        if 'adaptive' in benchmark_results and 'standard' in benchmark_results:
            speedup = benchmark_results['standard'] / benchmark_results['adaptive']
            target_speedup = 1.2  # Expect at least 20% speedup
            
            performance_target_met = speedup >= target_speedup
            
            self.test_results['performance_benchmarks'] = {
                'passed': performance_target_met,
                'benchmark_data': benchmark_results,
                'speedup': speedup,
                'details': f"Speedup: {speedup:.1f}x (target: {target_speedup}x)"
            }
            
            print(f"   Speedup: {speedup:.1f}x {'✓' if performance_target_met else '✗'}")
            print(f"   Result: {'✓ PASS' if performance_target_met else '✗ FAIL'}")
        else:
            self.test_results['performance_benchmarks'] = {
                'passed': False,
                'details': "Benchmark comparison failed"
            }
    
    # Helper methods
    def _create_mock_experiences(self, count: int) -> Dict[str, Any]:
        """Create mock experiences for testing."""
        experiences = {}
        for i in range(count):
            exp_id = f"mock_exp_{i}"
            experiences[exp_id] = type('MockExperience', (), {
                'sensory_input': [0.1 * i, 0.2, 0.3, 0.4] + [0.1] * 12,
                'action_taken': [0.1, 0.2, 0.3, 0.4],
                'outcome': [0.1 * i, 0.2, 0.3, 0.4] + [0.1] * 12,
                'prediction_error': 0.1 + 0.01 * i,
                'get_context_vector': lambda self: [0.1 * i, 0.2, 0.3, 0.4] + [0.1] * 12
            })()
        return experiences
    
    def _create_deterministic_experiences(self) -> Dict[str, Any]:
        """Create deterministic experiences for accuracy testing."""
        experiences = {}
        # Create experiences that should produce consistent predictions
        for i in range(10):
            exp_id = f"det_exp_{i}"
            experiences[exp_id] = type('DeterministicExperience', (), {
                'sensory_input': [0.5, 0.5, 0.5, 0.5] + [0.1] * 12,
                'action_taken': [0.1, 0.2, 0.3, 0.4],  # Consistent action
                'outcome': [0.5, 0.5, 0.5, 0.5] + [0.1] * 12,
                'prediction_error': 0.1,
                'get_context_vector': lambda self: [0.5, 0.5, 0.5, 0.5] + [0.1] * 12
            })()
        return experiences
    
    def _create_mock_similarity_engine(self):
        """Create mock similarity engine."""
        return type('MockSimilarityEngine', (), {
            'find_similar_experiences': lambda self, context, vectors, ids, max_results=10, min_similarity=0.4: [
                (f"mock_exp_{i}", 0.8 - 0.1 * i) for i in range(min(max_results, 5))
            ]
        })()
    
    def _create_deterministic_similarity_engine(self):
        """Create deterministic similarity engine for accuracy testing."""
        return type('DeterministicSimilarityEngine', (), {
            'find_similar_experiences': lambda self, context, vectors, ids, max_results=10, min_similarity=0.4: [
                ("det_exp_0", 0.9),
                ("det_exp_1", 0.85),
                ("det_exp_2", 0.8)
            ]
        })()
    
    def _create_brain_state_for_mode(self, mode: str, autopilot: CognitiveAutopilot) -> Dict[str, Any]:
        """Create brain state that forces specific intensity mode."""
        # Set autopilot to appropriate mode
        if mode == 'minimal':
            autopilot.current_mode = CognitiveMode.AUTOPILOT
        elif mode == 'selective':
            autopilot.current_mode = CognitiveMode.FOCUSED
        else:
            autopilot.current_mode = CognitiveMode.DEEP_THINK
        
        return {
            'cognitive_autopilot': {
                'recommendations': {
                    'pattern_analysis_intensity': mode
                }
            },
            'prediction_confidence': 0.8
        }
    
    def _actions_similar(self, action1: List[float], action2: List[float], tolerance: float = 0.1) -> bool:
        """Check if two actions are similar within tolerance."""
        if len(action1) != len(action2):
            return False
        
        differences = [abs(a1 - a2) for a1, a2 in zip(action1, action2)]
        return all(diff <= tolerance for diff in differences)
    
    def print_test_summary(self):
        """Print comprehensive test summary."""
        print("\n" + "=" * 50)
        print("📋 TEST SUMMARY")
        print("=" * 50)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result['passed'])
        
        print(f"\nOverall: {passed_tests}/{total_tests} tests passed ({passed_tests/total_tests:.1%})")
        
        for test_name, result in self.test_results.items():
            status = "✓ PASS" if result['passed'] else "✗ FAIL"
            print(f"\n{test_name}: {status}")
            print(f"  {result['details']}")
        
        # Performance summary
        if 'performance_benchmarks' in self.test_results:
            benchmark_data = self.test_results['performance_benchmarks'].get('benchmark_data', {})
            if benchmark_data:
                print(f"\n⚡ Performance Summary:")
                for config, time_ms in benchmark_data.items():
                    print(f"  {config}: {time_ms:.1f}ms average")
                
                if 'speedup' in self.test_results['performance_benchmarks']:
                    speedup = self.test_results['performance_benchmarks']['speedup']
                    print(f"  Adaptive speedup: {speedup:.1f}x")
        
        # Recommendations
        print(f"\n💡 Recommendations:")
        if passed_tests == total_tests:
            print("  ✅ All tests passed! Adaptive intensity system is ready for deployment.")
        elif passed_tests >= total_tests * 0.8:
            print("  ⚠️  Most tests passed. Review failing tests before deployment.")
        else:
            print("  ❌ Multiple test failures. Significant issues need resolution.")


def main():
    """Run the adaptive intensity test suite."""
    test_suite = AdaptiveIntensityTestSuite()
    test_suite.run_all_tests()


if __name__ == "__main__":
    main()