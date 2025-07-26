#!/usr/bin/env python3
"""
Behavioral Test-Driven Development Framework for Brain Intelligence

This framework tests actual intelligent behaviors rather than just technical functionality.
It can drive automated development: "develop and evaluate until X intelligence level is reached"
"""

import sys
import os
from pathlib import Path
import time
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import json

# Add brain server to path
brain_server_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(brain_server_path))

from src.brain_factory import BrainFactory

class IntelligenceMetric(Enum):
    """Core intelligence metrics we want to achieve"""
    PREDICTION_LEARNING = "prediction_learning"
    EXPLORATION_EXPLOITATION = "exploration_exploitation" 
    FIELD_STABILIZATION = "field_stabilization"
    PATTERN_RECOGNITION = "pattern_recognition"
    GOAL_SEEKING = "goal_seeking"
    BIOLOGICAL_REALISM = "biological_realism"
    COMPUTATIONAL_EFFICIENCY = "computational_efficiency"

@dataclass
class BehavioralTarget:
    """A specific behavioral goal with measurable criteria"""
    metric: IntelligenceMetric
    target_value: float
    tolerance: float
    description: str
    test_duration_cycles: int = 100
    
    def is_achieved(self, measured_value: float) -> bool:
        """Check if the behavioral target is achieved"""
        return measured_value >= (self.target_value - self.tolerance)

@dataclass 
class IntelligenceProfile:
    """Complete intelligence profile with multiple behavioral targets"""
    name: str
    targets: List[BehavioralTarget]
    
    def overall_achievement(self, results: Dict[IntelligenceMetric, float]) -> float:
        """Calculate overall intelligence achievement (0.0 to 1.0)"""
        if not self.targets:
            return 0.0
        
        achievements = []
        for target in self.targets:
            measured = results.get(target.metric, 0.0)
            if target.target_value > 0:
                achievement = min(1.0, measured / target.target_value)
            else:
                achievement = 1.0 if target.is_achieved(measured) else 0.0
            achievements.append(achievement)
        
        return np.mean(achievements)

class BehavioralTestFramework:
    """Framework for testing and driving brain intelligence development"""
    
    def __init__(self, quiet_mode: bool = True):
        self.quiet_mode = quiet_mode
        self.test_results_history = []
        
    def create_brain(self, config: Dict[str, Any] = None) -> BrainFactory:
        """Create a brain for testing with optional configuration"""
        # Load default configuration from settings.json to ensure proper memory paths
        import json
        import os
        
        try:
            # Try to load settings.json for proper configuration
            settings_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '..', 'settings.json')
            with open(settings_path, 'r') as f:
                default_config = json.load(f)
        except:
            # Fallback minimal config if settings.json not available
            default_config = {
                'memory': {'persistent_memory_path': './server/robot_memory', 'enable_persistence': True},
                'brain': {'type': 'field', 'sensory_dim': 16, 'motor_dim': 4}
            }
        
        # Merge with provided config
        if config is None:
            config = {}
        
        # Ensure proper memory configuration is present
        if 'memory' not in config:
            config['memory'] = default_config.get('memory', {})
        
        # PERFORMANCE FIX: Disable persistence for testing
        config['memory']['enable_persistence'] = False
        
        if 'brain_implementation' not in config:
            config['brain_implementation'] = 'field'
            
        return BrainFactory(config=config, quiet_mode=self.quiet_mode, enable_logging=False)
    
    def test_prediction_learning(self, brain: BrainFactory, cycles: int = 100, divergent_test: bool = False) -> float:
        """Test how well the brain learns to predict patterns"""
        # BEHAVIORAL TEST FIX: Reset confidence history for fresh learning curves
        if hasattr(brain, 'unified_brain') and hasattr(brain.unified_brain, '_prediction_confidence_history'):
            brain.unified_brain._prediction_confidence_history = []
            brain.unified_brain._improvement_rate_history = []
        
        if divergent_test:
            # Extremely different pattern to test paradigm shifting
            pattern = [0.01, 0.99, 0.02, 0.98, 0.03, 0.97, 0.04, 0.96] * 2  # High contrast, different scale
        else:
            # Original similar pattern
            pattern = [0.5, 0.3, 0.8, 0.2, 0.6, 0.1, 0.9, 0.4] * 2  # 16D sensory input
        
        prediction_errors = []
        prediction_confidences = []
        
        for i in range(cycles):
            # Present pattern and get brain prediction
            action, brain_state = brain.process_sensory_input(pattern)
            
            # Measure prediction quality (higher confidence = better prediction)  
            # UNIFIED BRAIN FIX: Use the correct key from UnifiedFieldBrain state
            prediction_confidence = brain_state.get('last_action_confidence', 
                                    brain_state.get('prediction_confidence', 0.0))
            prediction_errors.append(1.0 - prediction_confidence)
            prediction_confidences.append(prediction_confidence)
        
        # Learning success: prediction errors should decrease over time
        if len(prediction_errors) < 20:
            return 0.0
        
        # Use a sliding window approach to measure learning
        # This is more robust to pre-existing confidence history
        quarter_size = cycles // 4
        first_quarter = np.mean(prediction_errors[:quarter_size])
        last_quarter = np.mean(prediction_errors[-quarter_size:])
        
        # DEBUG: Print confidence progression for troubleshooting
        if not self.quiet_mode:
            early_conf = np.mean(prediction_confidences[:quarter_size])
            late_conf = np.mean(prediction_confidences[-quarter_size:])
            print(f"   Debug: early_conf={early_conf:.3f}, late_conf={late_conf:.3f}")
            print(f"   Debug: early_error={first_quarter:.3f}, late_error={last_quarter:.3f}")
            print(f"   Debug: improvement={(first_quarter - last_quarter) / first_quarter:.3f} if first_quarter > 0")
        
        if first_quarter == 0:
            return 0.0
        
        improvement = max(0.0, (first_quarter - last_quarter) / first_quarter)
        return min(1.0, improvement)
    
    def test_exploration_exploitation_balance(self, brain: BrainFactory, cycles: int = 200) -> float:
        """Test proper balance between exploration and exploitation"""
        try:
            consistent_input = [0.5] * 16  # Consistent environment
            
            early_actions = []
            late_actions = []
            
            for i in range(cycles):
                action, _ = brain.process_sensory_input(consistent_input)
                
                if i < 50:  # Early phase - should explore
                    early_actions.append(action)
                elif i >= 150:  # Late phase - should exploit/converge
                    late_actions.append(action)
            
            if not early_actions or not late_actions:
                print(f"❌ No actions collected: early={len(early_actions)}, late={len(late_actions)}")
                return 0.0
            
            # Calculate action variance
            early_variance = np.var([np.var(action) for action in early_actions])
            late_variance = np.var([np.var(action) for action in late_actions])
            
            # Good exploration-exploitation: high early variance, lower late variance
            exploration_score = min(1.0, early_variance * 10)  # Scale to 0-1
            exploitation_score = 1.0 if late_variance < early_variance else 0.0
            # FIXED: Low variance in late phase is GOOD (exploitation), not bad
            convergence_score = 1.0 if late_variance < early_variance * 0.5 else 0.5  # Reward strong convergence
            
            final_score = (exploration_score + exploitation_score + convergence_score) / 3.0
            
            # DEBUG: Print scoring breakdown 
            if not self.quiet_mode:
                print(f"   Debug: early_var={early_variance:.6f}, late_var={late_variance:.6f}")
                print(f"   Scores: exploration={exploration_score:.3f}, exploitation={exploitation_score:.3f}, convergence={convergence_score:.3f}")
                print(f"   Final: {final_score:.3f}")
            
            return final_score
            
        except Exception as e:
            print(f"❌ Exception in exploration_exploitation test: {e}")
            return 0.0
    
    def test_field_stabilization(self, brain: BrainFactory, cycles: int = 100) -> float:
        """Test if field energy stabilizes with learning (biological realism)"""
        pattern = [0.7, 0.3, 0.6, 0.4] * 4  # Repeating pattern
        
        field_energies = []
        
        for i in range(cycles):
            _, brain_state = brain.process_sensory_input(pattern)
            field_energy = brain_state.get('field_energy', 0.0)
            field_energies.append(field_energy)
        
        if len(field_energies) < 40:
            return 0.0
        
        # Energy should decrease and stabilize
        early_energy = np.mean(field_energies[:20])
        late_energy = np.mean(field_energies[-20:])
        late_variance = np.var(field_energies[-20:])
        
        # Scoring criteria
        energy_decrease_score = 1.0 if late_energy < early_energy else 0.0
        stability_score = max(0.0, 1.0 - late_variance)  # Lower variance = higher score
        
        return (energy_decrease_score + stability_score) / 2.0
    
    def test_pattern_recognition(self, brain: BrainFactory, cycles: int = 150) -> float:
        """Test ability to recognize and respond differently to different patterns"""
        pattern_a = [0.8, 0.2, 0.6, 0.4] * 4
        pattern_b = [0.2, 0.8, 0.4, 0.6] * 4
        
        actions_a = []
        actions_b = []
        
        for i in range(cycles):
            if i % 2 == 0:
                action, _ = brain.process_sensory_input(pattern_a)
                actions_a.append(action)
            else:
                action, _ = brain.process_sensory_input(pattern_b)
                actions_b.append(action)
        
        if len(actions_a) < 10 or len(actions_b) < 10:
            return 0.0
        
        # Calculate mean responses to each pattern
        mean_response_a = np.mean(actions_a[-10:], axis=0)
        mean_response_b = np.mean(actions_b[-10:], axis=0)
        
        # Good pattern recognition: different responses to different patterns
        response_difference = np.linalg.norm(mean_response_a - mean_response_b)
        recognition_score = min(1.0, response_difference * 2)  # Scale to 0-1
        
        return recognition_score
    
    def test_goal_seeking(self, brain: BrainFactory, cycles: int = 100) -> float:
        """Test goal-directed behavior (simplified light-seeking)"""
        goal_seeking_scores = []
        
        for i in range(cycles):
            # Simulate distance to light source (closer = higher values in first channels)
            distance_to_light = np.random.uniform(0.1, 1.0)
            light_gradient = [distance_to_light, distance_to_light * 0.8] + [0.1] * 14
            
            action, _ = brain.process_sensory_input(light_gradient)
            
            # Good goal-seeking: stronger actions when closer to goal
            action_strength = np.linalg.norm(action)
            goal_response = action_strength * distance_to_light
            goal_seeking_scores.append(goal_response)
        
        if not goal_seeking_scores:
            return 0.0
        
        # Score based on average goal-directed response
        return min(1.0, np.mean(goal_seeking_scores))
    
    def test_biological_realism(self, brain: BrainFactory, cycles: int = 100) -> float:
        """Test biological realism markers"""
        pattern = [0.6, 0.4, 0.7, 0.3] * 4
        
        evolution_cycles_history = []
        prediction_efficiency_history = []
        
        for i in range(cycles):
            _, brain_state = brain.process_sensory_input(pattern)
            
            evolution_cycles = brain_state.get('field_evolution_cycles', 0)
            prediction_efficiency = brain_state.get('prediction_efficiency', 0.0)
            
            evolution_cycles_history.append(evolution_cycles)
            prediction_efficiency_history.append(prediction_efficiency)
        
        # Biological markers
        evolution_occurred = max(evolution_cycles_history) > 0
        efficiency_improved = (np.mean(prediction_efficiency_history[-10:]) > 
                              np.mean(prediction_efficiency_history[:10]))
        
        biological_score = 0.0
        if evolution_occurred:
            biological_score += 0.5
        if efficiency_improved:
            biological_score += 0.5
        
        return biological_score
    
    def test_computational_efficiency(self, brain: BrainFactory, cycles: int = 50) -> float:
        """Test intelligence per compute - measures intelligence achievement relative to computational cost"""
        test_input = [0.5] * 16
        
        # Warm-up to get stable performance
        for i in range(5):
            brain.process_sensory_input(test_input)
        
        start_time = time.time()
        
        for i in range(cycles):
            brain.process_sensory_input(test_input)
        
        elapsed_time = time.time() - start_time
        avg_cycle_time = elapsed_time / cycles
        throughput = 1.0 / avg_cycle_time  # cycles per second
        
        # Calculate current intelligence achievement from prediction learning and exploration-exploitation
        # (simplified version of full assessment for efficiency measurement)
        quick_prediction_score = self._quick_prediction_assessment(brain, 20)
        quick_exploration_score = self._quick_exploration_assessment(brain, 50) 
        intelligence_level = (quick_prediction_score + quick_exploration_score) / 2.0
        
        # Intelligence per compute: intelligence achievement per unit of computational cost
        # Higher intelligence + higher throughput = better efficiency
        # Lower intelligence + lower throughput = worse efficiency
        
        # Base throughput score (0.0 to 1.0)
        if throughput >= 5.0:  # 200ms or better
            throughput_score = 1.0
        elif throughput >= 2.0:  # 200-500ms range
            throughput_score = 0.4 + (throughput - 2.0) / 3.0 * 0.6
        elif throughput >= 1.0:  # 500ms-1s range  
            throughput_score = 0.2 + (throughput - 1.0) / 1.0 * 0.2
        else:  # Slower than 1 cycle/sec
            throughput_score = max(0.0, throughput / 1.0 * 0.2)
        
        # Intelligence per compute formula:
        # Reward high intelligence even with moderate throughput
        # Penalize low intelligence even with high throughput
        intelligence_per_compute = (intelligence_level * 0.7) + (throughput_score * 0.3)
        
        # Bonus for achieving both high intelligence AND high throughput
        if intelligence_level > 0.5 and throughput_score > 0.7:
            intelligence_per_compute *= 1.2  # 20% bonus for high performance on both
        
        return min(1.0, max(0.0, intelligence_per_compute))
    
    def _quick_prediction_assessment(self, brain: BrainFactory, cycles: int) -> float:
        """Quick prediction learning assessment for efficiency calculation"""
        pattern = [0.5, 0.3, 0.8, 0.2, 0.6, 0.1, 0.9, 0.4] * 2
        prediction_errors = []
        
        for i in range(cycles):
            _, brain_state = brain.process_sensory_input(pattern)
            prediction_confidence = brain_state.get('prediction_confidence', 0.0)
            prediction_errors.append(1.0 - prediction_confidence)
        
        if len(prediction_errors) < 10:
            return 0.0
            
        early_error = np.mean(prediction_errors[:5])
        late_error = np.mean(prediction_errors[-5:])
        
        if early_error == 0:
            return 0.0
        
        improvement = max(0.0, (early_error - late_error) / early_error)
        return min(1.0, improvement)
    
    def _quick_exploration_assessment(self, brain: BrainFactory, cycles: int) -> float:
        """Quick exploration-exploitation assessment for efficiency calculation"""
        consistent_input = [0.5] * 16
        early_actions = []
        late_actions = []
        
        for i in range(cycles):
            action, _ = brain.process_sensory_input(consistent_input)
            
            if i < 15:  # Early phase
                early_actions.append(action)
            elif i >= 35:  # Late phase
                late_actions.append(action)
        
        if not early_actions or not late_actions:
            return 0.0
        
        early_variance = np.var([np.var(action) for action in early_actions])
        late_variance = np.var([np.var(action) for action in late_actions])
        
        exploration_score = min(1.0, early_variance * 10)
        convergence_score = 1.0 if late_variance < early_variance * 0.5 else 0.5
        
        return (exploration_score + convergence_score) / 2.0
    
    def run_intelligence_assessment(self, brain: BrainFactory, 
                                  profile: IntelligenceProfile) -> Dict[str, Any]:
        """Run complete intelligence assessment against a profile"""
        print(f"\n🧠 Running Intelligence Assessment: {profile.name}")
        print("=" * 60)
        
        results = {}
        detailed_results = {}
        
        for target in profile.targets:
            print(f"📊 Testing {target.metric.value}...")
            
            # Run the appropriate test
            if target.metric == IntelligenceMetric.PREDICTION_LEARNING:
                score = self.test_prediction_learning(brain, target.test_duration_cycles)
            elif target.metric == IntelligenceMetric.EXPLORATION_EXPLOITATION:
                score = self.test_exploration_exploitation_balance(brain, target.test_duration_cycles)
            elif target.metric == IntelligenceMetric.FIELD_STABILIZATION:
                score = self.test_field_stabilization(brain, target.test_duration_cycles)
            elif target.metric == IntelligenceMetric.PATTERN_RECOGNITION:
                score = self.test_pattern_recognition(brain, target.test_duration_cycles)
            elif target.metric == IntelligenceMetric.GOAL_SEEKING:
                score = self.test_goal_seeking(brain, target.test_duration_cycles)
            elif target.metric == IntelligenceMetric.BIOLOGICAL_REALISM:
                score = self.test_biological_realism(brain, target.test_duration_cycles)
            elif target.metric == IntelligenceMetric.COMPUTATIONAL_EFFICIENCY:
                score = self.test_computational_efficiency(brain, target.test_duration_cycles)
            else:
                score = 0.0
            
            results[target.metric] = score
            achieved = target.is_achieved(score)
            
            detailed_results[target.metric.value] = {
                'score': score,
                'target': target.target_value,
                'achieved': achieved,
                'description': target.description
            }
            
            status = "✅ PASS" if achieved else "❌ FAIL"
            print(f"   Score: {score:.3f} / Target: {target.target_value:.3f} {status}")
        
        overall_achievement = profile.overall_achievement(results)
        
        print(f"\n🎯 Overall Intelligence Achievement: {overall_achievement:.1%}")
        
        assessment_result = {
            'profile_name': profile.name,
            'overall_achievement': overall_achievement,
            'detailed_results': detailed_results,
            'timestamp': time.time()
        }
        
        self.test_results_history.append(assessment_result)
        return assessment_result
    
    def compare_brain_implementations(self, config_a: Dict, config_b: Dict, 
                                    profile: IntelligenceProfile) -> Dict[str, Any]:
        """Compare two brain implementations for intelligence parity"""
        print(f"\n🔬 Brain Implementation Comparison")
        print("=" * 60)
        
        brain_a = self.create_brain(config_a)
        brain_b = self.create_brain(config_b)
        
        results_a = self.run_intelligence_assessment(brain_a, profile)
        results_b = self.run_intelligence_assessment(brain_b, profile)
        
        # Compare achievements
        comparison = {
            'brain_a': results_a,
            'brain_b': results_b,
            'parity_analysis': {}
        }
        
        print(f"\n📊 Parity Analysis:")
        for target in profile.targets:
            score_a = results_a['detailed_results'][target.metric.value]['score']
            score_b = results_b['detailed_results'][target.metric.value]['score']
            difference = abs(score_a - score_b)
            parity_ok = difference < 0.1  # 10% tolerance
            
            comparison['parity_analysis'][target.metric.value] = {
                'score_a': score_a,
                'score_b': score_b,
                'difference': difference,
                'parity_ok': parity_ok
            }
            
            status = "✅ PARITY" if parity_ok else "⚠️ DIVERGENT"
            print(f"   {target.metric.value}: {score_a:.3f} vs {score_b:.3f} (Δ{difference:.3f}) {status}")
        
        return comparison


# Predefined Intelligence Profiles

BASIC_INTELLIGENCE_PROFILE = IntelligenceProfile(
    name="Basic Intelligence",
    targets=[
        BehavioralTarget(IntelligenceMetric.PREDICTION_LEARNING, 0.3, 0.05, 
                        "Brain should improve predictions over time"),
        BehavioralTarget(IntelligenceMetric.EXPLORATION_EXPLOITATION, 0.5, 0.1,
                        "Brain should balance exploration and exploitation", 200),
        BehavioralTarget(IntelligenceMetric.COMPUTATIONAL_EFFICIENCY, 0.7, 0.1,
                        "Brain should process efficiently")
    ]
)

BIOLOGICAL_REALISM_PROFILE = IntelligenceProfile(
    name="Biological Realism",
    targets=[
        BehavioralTarget(IntelligenceMetric.FIELD_STABILIZATION, 0.6, 0.1,
                        "Field energy should stabilize with learning"),
        BehavioralTarget(IntelligenceMetric.BIOLOGICAL_REALISM, 0.7, 0.1,
                        "Should show biological learning markers"),
        BehavioralTarget(IntelligenceMetric.PREDICTION_LEARNING, 0.4, 0.05,
                        "Should learn patterns biologically")
    ]
)

ADVANCED_INTELLIGENCE_PROFILE = IntelligenceProfile(
    name="Advanced Intelligence",
    targets=[
        BehavioralTarget(IntelligenceMetric.PREDICTION_LEARNING, 0.7, 0.05,
                        "Strong pattern learning"),
        BehavioralTarget(IntelligenceMetric.PATTERN_RECOGNITION, 0.6, 0.1,
                        "Distinguish between different patterns"),
        BehavioralTarget(IntelligenceMetric.GOAL_SEEKING, 0.5, 0.1,
                        "Show goal-directed behavior"),
        BehavioralTarget(IntelligenceMetric.EXPLORATION_EXPLOITATION, 0.7, 0.1,
                        "Sophisticated exploration-exploitation balance"),
        BehavioralTarget(IntelligenceMetric.COMPUTATIONAL_EFFICIENCY, 0.8, 0.1,
                        "High computational efficiency")
    ]
)


def automated_development_cycle(framework: BehavioralTestFramework,
                              profile: IntelligenceProfile,
                              max_iterations: int = 10,
                              target_achievement: float = 0.8) -> Dict[str, Any]:
    """
    Automated development cycle: test → analyze → improve → repeat
    
    This is the holy grail: "develop and evaluate until X intelligence is reached"
    """
    print(f"\n🚀 Automated Development Cycle")
    print(f"Target: {target_achievement:.1%} achievement on {profile.name}")
    print("=" * 60)
    
    iteration_results = []
    
    for iteration in range(max_iterations):
        print(f"\n🔄 Development Iteration {iteration + 1}/{max_iterations}")
        
        # Create brain with current best configuration
        brain = framework.create_brain()
        
        # Test current intelligence level
        assessment = framework.run_intelligence_assessment(brain, profile)
        achievement = assessment['overall_achievement']
        
        iteration_results.append({
            'iteration': iteration + 1,
            'achievement': achievement,
            'assessment': assessment
        })
        
        print(f"Current achievement: {achievement:.1%}")
        
        # Check if target reached
        if achievement >= target_achievement:
            print(f"🎉 TARGET ACHIEVED! Reached {achievement:.1%} intelligence")
            break
        
        # Analyze weakest areas for improvement
        weak_areas = []
        for target in profile.targets:
            metric_result = assessment['detailed_results'][target.metric.value]
            if not metric_result['achieved']:
                weak_areas.append((target.metric, metric_result['score'], target.target_value))
        
        if weak_areas:
            weakest_metric, current_score, target_score = min(weak_areas, key=lambda x: x[1])
            improvement_needed = target_score - current_score
            print(f"🎯 Focus area: {weakest_metric.value} (needs +{improvement_needed:.3f})")
        
        # Brain cleanup
        try:
            brain.finalize_session()
        except:
            pass
    
    return {
        'target_achievement': target_achievement,
        'final_achievement': achievement if 'achievement' in locals() else 0.0,
        'iterations_completed': len(iteration_results),
        'target_reached': achievement >= target_achievement if 'achievement' in locals() else False,
        'iteration_history': iteration_results
    }


def test_paradigm_shifting_experiment():
    """Experimental test to validate paradigm shifting with divergent patterns"""
    framework = BehavioralTestFramework(quiet_mode=True)
    
    print("🧪 Paradigm Shifting Experiment")
    print("Testing if truly divergent patterns trigger different learning behavior")
    print("=" * 60)
    
    # Test 1: Similar patterns (current system)
    brain1 = framework.create_brain()
    similar_score = framework.test_prediction_learning(brain1, cycles=100, divergent_test=False)
    
    # Test 2: Divergent patterns (should trigger paradigm shift?)
    brain2 = framework.create_brain()
    divergent_score = framework.test_prediction_learning(brain2, cycles=100, divergent_test=True)
    
    print(f"\n📊 Results:")
    print(f"   Similar Pattern Learning:  {similar_score:.3f} (7% expected)")
    print(f"   Divergent Pattern Learning: {divergent_score:.3f} (higher expected if paradigm shifting works)")
    
    ratio = divergent_score / similar_score if similar_score > 0 else float('inf')
    print(f"   Divergent/Similar Ratio: {ratio:.2f}x")
    
    if ratio > 2.0:
        print(f"✅ PARADIGM SHIFTING DETECTED! Divergent patterns trigger different learning")
    elif ratio > 1.2:
        print(f"⚠️  WEAK PARADIGM RESPONSE: Some differentiation detected")
    else:
        print(f"❌ NO PARADIGM SHIFTING: Similar learning behavior regardless of pattern divergence")
    
    return similar_score, divergent_score

if __name__ == "__main__":
    """Example usage of the behavioral test framework"""
    
    # Run the paradigm shifting experiment first
    test_paradigm_shifting_experiment()
    
    print("\n" + "="*60)
    
    framework = BehavioralTestFramework(quiet_mode=True)
    
    print("🧠 Behavioral Test-Driven Development Framework")
    print("Testing actual intelligence behaviors, not just technical functionality")
    
    # Test current brain intelligence
    brain = framework.create_brain()
    
    # Run basic intelligence assessment
    results = framework.run_intelligence_assessment(brain, BASIC_INTELLIGENCE_PROFILE)
    
    print(f"\n✅ Behavioral testing framework ready!")
    print(f"Use this for test-driven brain development!")