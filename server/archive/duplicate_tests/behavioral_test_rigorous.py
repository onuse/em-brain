#!/usr/bin/env python3
"""
Rigorous Behavioral Test Framework

Comprehensive intelligence testing with proper brain initialization.
Includes advanced tests for pattern recognition, goal seeking, and biological realism.
"""

import sys
import os
from pathlib import Path
import time
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from enum import Enum

# Add brain server to path
brain_server_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(brain_server_path))

from src.core.robot_registry import RobotRegistry
from src.core.brain_pool import BrainPool
from src.core.brain_service import BrainService
from src.core.adapters import AdapterFactory
from src.core.connection_handler import ConnectionHandler
from src.core.dynamic_brain_factory import DynamicBrainFactory
from src.core.direct_telemetry import DirectTelemetry
from src.core.monitoring_server import DynamicMonitoringServer


class IntelligenceMetric(Enum):
    """Core intelligence metrics"""
    PREDICTION_LEARNING = "prediction_learning"
    EXPLORATION_EXPLOITATION = "exploration_exploitation" 
    FIELD_STABILIZATION = "field_stabilization"
    PATTERN_RECOGNITION = "pattern_recognition"
    GOAL_SEEKING = "goal_seeking"
    BIOLOGICAL_REALISM = "biological_realism"
    COMPUTATIONAL_EFFICIENCY = "computational_efficiency"
    PARADIGM_SHIFTING = "paradigm_shifting"


@dataclass
class BehavioralTarget:
    """Behavioral target for testing"""
    metric: IntelligenceMetric
    target_value: float
    tolerance: float
    description: str
    test_duration_cycles: int = 100
    
    def is_achieved(self, measured_value: float) -> bool:
        return measured_value >= (self.target_value - self.tolerance)


@dataclass 
class IntelligenceProfile:
    """Intelligence profile for testing"""
    name: str
    targets: List[BehavioralTarget]
    
    def overall_achievement(self, results: Dict[IntelligenceMetric, float]) -> float:
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


class RigorousBehavioralTestFramework:
    """Rigorous behavioral testing with proper initialization"""
    
    def __init__(self, quiet_mode: bool = False):
        self.quiet_mode = quiet_mode
        
        print("🧪 Initializing Rigorous Behavioral Test Framework")
        print("=" * 60)
        
        # Initialize with proper architecture
        self.robot_registry = RobotRegistry()
        
        brain_config = {
            'quiet_mode': quiet_mode,
            'spatial_resolution': 4  # Adaptive based on hardware
        }
        self.brain_factory = DynamicBrainFactory(brain_config)
        self.brain_pool = BrainPool(self.brain_factory)
        self.adapter_factory = AdapterFactory()
        self.brain_service = BrainService(self.brain_pool, self.adapter_factory)
        self.connection_handler = ConnectionHandler(self.robot_registry, self.brain_service)
        
        # Monitoring for telemetry
        self.monitoring_server = DynamicMonitoringServer(
            brain_service=self.brain_service,
            connection_handler=self.connection_handler,
            host='localhost',
            port=9998
        )
        self.monitoring_server.start()
        
        # Use direct telemetry instead of socket-based
        self.telemetry_client = DirectTelemetry(
            brain_service=self.brain_service,
            connection_handler=self.connection_handler
        )
        
        self.client_id = "rigorous_test_robot"
        self.session_id = None
        
        print("✅ Framework initialized with full architecture")
    
    def setup_robot(self, sensory_dim: int = 16, motor_dim: int = 4):
        """Setup robot with specified dimensions"""
        capabilities = [1.0, float(sensory_dim), float(motor_dim), 0.0, 0.0]
        response = self.connection_handler.handle_handshake(self.client_id, capabilities)
        
        self.session_id = self.telemetry_client.wait_for_session(max_wait=2.0, client_id=self.client_id)
        
        print(f"🤖 Robot: {sensory_dim}D sensors, {motor_dim}D motors")
        return response
    
    def process_cycle(self, sensory_input: List[float]) -> Tuple[List[float], Optional[Any]]:
        """Process one cycle and return motor output and telemetry"""
        motor_output = self.connection_handler.handle_sensory_input(
            self.client_id, sensory_input
        )
        
        telemetry = None
        if self.session_id:
            telemetry = self.telemetry_client.get_session_telemetry(self.session_id)
        
        return motor_output, telemetry
    
    def test_prediction_learning(self, cycles: int = 100, divergent: bool = False) -> float:
        """Test prediction learning capability"""
        if not self.quiet_mode:
            print(f"  Testing prediction learning ({cycles} cycles)...")
        
        # Pattern selection
        if divergent:
            # Extreme contrast pattern for paradigm shifting
            pattern = [0.01, 0.99, 0.02, 0.98] * 4
        else:
            # Regular pattern
            pattern = [0.5, 0.3, 0.8, 0.2] * 4
        
        confidence_history = []
        
        for i in range(cycles):
            _, telemetry = self.process_cycle(pattern)
            
            if telemetry:
                confidence_history.append(telemetry.confidence)
        
        if len(confidence_history) >= 20:
            early = np.mean(confidence_history[:10])
            late = np.mean(confidence_history[-10:])
            improvement = late - early
            
            # Score based on improvement and final confidence
            improvement_score = max(0, min(1.0, improvement * 5))
            final_score = max(0, min(1.0, (confidence_history[-1] - 0.5) * 2))
            
            return (improvement_score + final_score) / 2
        
        return 0.0
    
    def test_pattern_recognition(self, cycles: int = 150) -> float:
        """Test ability to distinguish between different patterns"""
        if not self.quiet_mode:
            print(f"  Testing pattern recognition ({cycles} cycles)...")
        
        # Three distinct patterns
        patterns = {
            'sine': lambda t: [np.sin(t * 0.1 + i * 0.5) for i in range(16)],
            'square': lambda t: [1.0 if (t + i) % 10 < 5 else 0.0 for i in range(16)],
            'random': lambda t: list(np.random.rand(16))
        }
        
        responses_by_pattern = {name: [] for name in patterns}
        
        # Present patterns in sequence
        for i in range(cycles):
            pattern_name = list(patterns.keys())[i % len(patterns)]
            pattern = patterns[pattern_name](i)
            
            motor_output, _ = self.process_cycle(pattern)
            
            # Store response characteristics
            response_energy = np.mean(np.abs(motor_output))
            responses_by_pattern[pattern_name].append(response_energy)
        
        # Calculate how distinct the responses are
        mean_responses = {name: np.mean(responses) 
                         for name, responses in responses_by_pattern.items()
                         if len(responses) > 5}
        
        if len(mean_responses) >= 2:
            # Calculate coefficient of variation between pattern responses
            response_values = list(mean_responses.values())
            cv = np.std(response_values) / (np.mean(response_values) + 1e-6)
            
            # Good pattern recognition shows different responses
            return min(1.0, cv * 2)
        
        return 0.0
    
    def test_goal_seeking(self, cycles: int = 100) -> float:
        """Test goal-directed behavior"""
        if not self.quiet_mode:
            print(f"  Testing goal seeking ({cycles} cycles)...")
        
        goal_achievements = []
        
        for goal_position in [0, 8, 15]:  # Different goal positions
            # Create goal signal (high value at goal position)
            goal_pattern = [0.1] * 16
            goal_pattern[goal_position] = 1.0
            
            motor_convergence = []
            
            for i in range(cycles // 3):
                motor_output, _ = self.process_cycle(goal_pattern)
                
                # Check if motor output shows directed behavior
                # (e.g., highest activation near goal dimension)
                if len(motor_output) >= 4:
                    # Map goal position to motor dimension
                    motor_goal_dim = min(3, goal_position // 6)
                    motor_response = abs(motor_output[motor_goal_dim])
                    motor_convergence.append(motor_response)
            
            # Check if response strengthens over time (learning the goal)
            if len(motor_convergence) >= 10:
                early = np.mean(motor_convergence[:5])
                late = np.mean(motor_convergence[-5:])
                achievement = (late - early) / (early + 1e-6)
                goal_achievements.append(max(0, min(1.0, achievement)))
        
        return np.mean(goal_achievements) if goal_achievements else 0.0
    
    def test_biological_realism(self, cycles: int = 100) -> float:
        """Test biological constraints and energy conservation"""
        if not self.quiet_mode:
            print(f"  Testing biological realism ({cycles} cycles)...")
        
        energy_levels = []
        phase_transitions = []
        cycle_times = []
        
        # Natural input pattern (not too extreme)
        for i in range(cycles):
            pattern = [0.5 + 0.3 * np.sin(i * 0.1 + j * 0.2) for j in range(16)]
            
            start_time = time.time()
            _, telemetry = self.process_cycle(pattern)
            cycle_time = (time.time() - start_time) * 1000
            
            if telemetry:
                energy_levels.append(telemetry.energy)
                phase_transitions.append(telemetry.phase)
                cycle_times.append(cycle_time)
        
        scores = []
        
        # Energy conservation (should be stable, not exploding)
        if energy_levels:
            energy_var = np.var(energy_levels)
            energy_mean = np.mean(energy_levels)
            if energy_mean > 0:
                energy_stability = 1.0 / (1.0 + energy_var / energy_mean)
                scores.append(energy_stability)
        
        # Phase diversity (biological systems show multiple states)
        if phase_transitions:
            unique_phases = len(set(phase_transitions))
            phase_diversity = min(1.0, unique_phases / 3.0)  # Expect at least 3 phases
            scores.append(phase_diversity)
        
        # Timing realism (150ms target)
        if cycle_times:
            avg_time = np.mean(cycle_times)
            timing_score = 1.0 if avg_time <= 150 else max(0, 1 - (avg_time - 150) / 500)
            scores.append(timing_score)
        
        return np.mean(scores) if scores else 0.0
    
    def test_exploration_exploitation(self, cycles: int = 200) -> float:
        """Test exploration vs exploitation balance"""
        if not self.quiet_mode:
            print(f"  Testing exploration/exploitation ({cycles} cycles)...")
        
        # Use telemetry for accurate measurement
        cognitive_modes = []
        motor_diversity = []
        
        # Varying reward landscape
        for i in range(cycles):
            if i < cycles // 3:
                # Exploration phase - novel inputs
                pattern = list(np.random.rand(16))
            elif i < 2 * cycles // 3:
                # Mixed phase - some patterns repeat
                pattern = [np.sin(i * 0.1 + j) for j in range(16)] if i % 3 == 0 else list(np.random.rand(16))
            else:
                # Exploitation phase - consistent reward
                pattern = [0.8, 0.2] * 8
            
            motor_output, telemetry = self.process_cycle(pattern)
            motor_diversity.append(np.std(motor_output))
            
            if telemetry:
                cognitive_modes.append(telemetry.mode)
        
        # Calculate metrics
        early_diversity = np.mean(motor_diversity[:cycles//3])
        late_diversity = np.mean(motor_diversity[-cycles//3:])
        
        # Good balance: high early diversity, lower late diversity
        exploration_score = min(1.0, early_diversity * 10)
        exploitation_score = 1.0 if late_diversity < early_diversity * 0.7 else 0.5
        
        # Mode switching as indicator
        if cognitive_modes:
            mode_switches = sum(1 for i in range(1, len(cognitive_modes)) 
                               if cognitive_modes[i] != cognitive_modes[i-1])
            switch_rate = mode_switches / len(cognitive_modes)
            mode_score = min(1.0, switch_rate * 10)  # Some switching is good
        else:
            mode_score = 0.5
        
        return (exploration_score + exploitation_score + mode_score) / 3
    
    def test_field_stabilization(self, cycles: int = 100) -> float:
        """Test field stabilization - measures absolute stability"""
        if not self.quiet_mode:
            print(f"  Testing field stabilization ({cycles} cycles)...")
        
        # Constant input
        stable_pattern = [0.5] * 16
        
        energy_history = []
        
        for i in range(cycles):
            _, telemetry = self.process_cycle(stable_pattern)
            
            if telemetry:
                energy_history.append(telemetry.energy)
        
        if len(energy_history) < 20:
            return 0.5
        
        # Convert to numpy array
        energy = np.array(energy_history)
        
        # 1. Absolute stability in the last quarter
        last_quarter = energy[-len(energy)//4:]
        mean_energy = np.mean(last_quarter)
        if mean_energy > 0:
            cv_last = np.std(last_quarter) / mean_energy
            stability_score = 1.0 / (1.0 + cv_last * 10)
        else:
            stability_score = 0.5
        
        # 2. Convergence trend
        quarters = np.array_split(energy, 4)
        quarter_stds = [np.std(q) for q in quarters]
        
        if len(quarter_stds) >= 4:
            early_volatility = np.mean(quarter_stds[:2])
            late_volatility = np.mean(quarter_stds[-2:])
            
            if early_volatility > 0:
                convergence_score = max(0, 1.0 - late_volatility / early_volatility)
            else:
                convergence_score = 1.0 if late_volatility < 0.001 else 0.5
        else:
            convergence_score = 0.5
        
        # 3. Overall trend
        window = max(5, len(energy) // 10)
        rolling_std = [np.std(energy[i:i+window]) 
                       for i in range(0, len(energy) - window, window//2)]
        
        if len(rolling_std) >= 2:
            trend = np.polyfit(range(len(rolling_std)), rolling_std, 1)[0]
            trend_score = 1.0 / (1.0 + np.exp(trend * 100))
        else:
            trend_score = 0.5
        
        # Combine scores
        final_score = (
            0.4 * stability_score +
            0.3 * convergence_score +
            0.3 * trend_score
        )
        
        return final_score
    
    def test_computational_efficiency(self, cycles: int = 50) -> float:
        """Test computational efficiency"""
        if not self.quiet_mode:
            print(f"  Testing computational efficiency ({cycles} cycles)...")
        
        cycle_times = []
        
        for i in range(cycles):
            pattern = list(np.random.randn(16) * 0.5)
            
            start_time = time.time()
            self.process_cycle(pattern)
            cycle_time = (time.time() - start_time) * 1000
            
            if i >= 5:  # Skip warmup
                cycle_times.append(cycle_time)
        
        avg_time = np.mean(cycle_times)
        
        # Biological target: 150ms
        if avg_time <= 150:
            return 1.0
        elif avg_time <= 300:
            return 0.5 + (300 - avg_time) / 300
        else:
            return max(0, 1 - (avg_time - 300) / 1000)
    
    def test_paradigm_shifting(self, cycles: int = 200) -> float:
        """Test ability to shift between radically different paradigms"""
        if not self.quiet_mode:
            print(f"  Testing paradigm shifting ({cycles} cycles)...")
        
        # First paradigm: low-energy patterns
        paradigm1_score = self.test_prediction_learning(cycles // 2, divergent=False)
        
        # Second paradigm: high-contrast patterns  
        paradigm2_score = self.test_prediction_learning(cycles // 2, divergent=True)
        
        # Good paradigm shifting: performs well on both
        return (paradigm1_score + paradigm2_score) / 2
    
    def run_assessment(self, profile: IntelligenceProfile) -> Dict[str, Any]:
        """Run comprehensive intelligence assessment"""
        print(f"\n🚀 Rigorous Assessment: {profile.name}")
        print("=" * 60)
        
        # Setup robot
        self.setup_robot()
        
        results = {}
        detailed_results = {}
        
        total_start = time.time()
        
        for target in profile.targets:
            print(f"\n📊 Testing {target.metric.value}...")
            
            test_start = time.time()
            
            # Run appropriate test
            if target.metric == IntelligenceMetric.PREDICTION_LEARNING:
                score = self.test_prediction_learning(target.test_duration_cycles)
            elif target.metric == IntelligenceMetric.EXPLORATION_EXPLOITATION:
                score = self.test_exploration_exploitation(target.test_duration_cycles)
            elif target.metric == IntelligenceMetric.FIELD_STABILIZATION:
                score = self.test_field_stabilization(target.test_duration_cycles)
            elif target.metric == IntelligenceMetric.PATTERN_RECOGNITION:
                score = self.test_pattern_recognition(target.test_duration_cycles)
            elif target.metric == IntelligenceMetric.GOAL_SEEKING:
                score = self.test_goal_seeking(target.test_duration_cycles)
            elif target.metric == IntelligenceMetric.BIOLOGICAL_REALISM:
                score = self.test_biological_realism(target.test_duration_cycles)
            elif target.metric == IntelligenceMetric.COMPUTATIONAL_EFFICIENCY:
                score = self.test_computational_efficiency(target.test_duration_cycles)
            elif target.metric == IntelligenceMetric.PARADIGM_SHIFTING:
                score = self.test_paradigm_shifting(target.test_duration_cycles)
            else:
                score = 0.0
            
            test_time = time.time() - test_start
            
            results[target.metric] = score
            detailed_results[target.metric.value] = {
                'score': score,
                'target': target.target_value,
                'achieved': target.is_achieved(score),
                'test_time_s': test_time
            }
            
            status = "✅ PASS" if target.is_achieved(score) else "❌ FAIL"
            print(f"   Score: {score:.3f} / Target: {target.target_value:.3f} {status}")
            print(f"   Time: {test_time:.1f}s")
        
        total_time = time.time() - total_start
        overall = profile.overall_achievement(results)
        
        print(f"\n🎯 Overall Achievement: {overall:.1%}")
        print(f"⏱️  Total time: {total_time:.1f}s")
        
        return {
            'profile_name': profile.name,
            'overall_achievement': overall,
            'results': results,
            'detailed_results': detailed_results,
            'total_test_time': total_time
        }
    
    def cleanup(self):
        """Clean shutdown"""
        if self.connection_handler and self.client_id:
            self.connection_handler.handle_disconnect(self.client_id)
        if self.telemetry_client:
            self.telemetry_client.disconnect()
        if self.monitoring_server:
            self.monitoring_server.stop()
        print("\n✅ Cleanup complete")


# Test profiles
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

COMPREHENSIVE_PROFILE = IntelligenceProfile(
    name="Comprehensive Intelligence",
    targets=[
        BehavioralTarget(IntelligenceMetric.PREDICTION_LEARNING, 0.6, 0.05,
                        "Strong prediction", 150),
        BehavioralTarget(IntelligenceMetric.PATTERN_RECOGNITION, 0.6, 0.1,
                        "Pattern discrimination", 150),
        BehavioralTarget(IntelligenceMetric.GOAL_SEEKING, 0.5, 0.1,
                        "Goal-directed behavior", 100),
        BehavioralTarget(IntelligenceMetric.EXPLORATION_EXPLOITATION, 0.6, 0.1,
                        "E/E balance", 200),
        BehavioralTarget(IntelligenceMetric.FIELD_STABILIZATION, 0.6, 0.1,
                        "Field stability", 100),
        BehavioralTarget(IntelligenceMetric.BIOLOGICAL_REALISM, 0.7, 0.1,
                        "Biological constraints", 100),
        BehavioralTarget(IntelligenceMetric.PARADIGM_SHIFTING, 0.5, 0.1,
                        "Paradigm adaptation", 200),
        BehavioralTarget(IntelligenceMetric.COMPUTATIONAL_EFFICIENCY, 0.7, 0.1,
                        "Efficiency", 50)
    ]
)


def main():
    """Run rigorous behavioral tests"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Rigorous Behavioral Test")
    parser.add_argument('--profile', choices=['basic', 'biological', 'advanced', 'comprehensive'], 
                        default='basic', help='Test profile to run')
    parser.add_argument('--quiet', action='store_true', help='Quiet mode')
    args = parser.parse_args()
    
    print("🧪 Rigorous Behavioral Intelligence Test")
    print("=" * 50)
    
    framework = RigorousBehavioralTestFramework(quiet_mode=args.quiet)
    
    # Select profile
    profiles = {
        'basic': BASIC_INTELLIGENCE_PROFILE,
        'biological': BIOLOGICAL_REALISM_PROFILE,
        'advanced': ADVANCED_INTELLIGENCE_PROFILE,
        'comprehensive': COMPREHENSIVE_PROFILE
    }
    profile = profiles[args.profile]
    
    try:
        results = framework.run_assessment(profile)
        
        print("\n" + "=" * 50)
        print("📊 FINAL RESULTS")
        print("=" * 50)
        
        for metric, details in results['detailed_results'].items():
            status = "✅" if details['achieved'] else "❌"
            print(f"{status} {metric}: {details['score']:.3f} / {details['target']:.3f}")
        
        print(f"\n🏆 Overall Achievement: {results['overall_achievement']:.1%}")
        
        if results['overall_achievement'] >= 0.8:
            print("🎉 Brain demonstrates excellent intelligence!")
        elif results['overall_achievement'] >= 0.6:
            print("👍 Brain shows good intelligence")
        else:
            print("🔧 Brain needs optimization")
            
    finally:
        framework.cleanup()


if __name__ == "__main__":
    main()