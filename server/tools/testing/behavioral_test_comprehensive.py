#!/usr/bin/env python3
"""
Comprehensive Behavioral Test - All Intelligence Metrics

A working version of the comprehensive test that includes all intelligence metrics.
Based on behavioral_test_dynamic.py but with additional tests from the rigorous framework.
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
from src.core.telemetry_client import TelemetryClient
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
    test_duration_cycles: int = 50
    
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


class ComprehensiveBehavioralTest:
    """Comprehensive behavioral test framework"""
    
    def __init__(self, quiet_mode: bool = False):
        self.quiet_mode = quiet_mode
        
        # Initialize components
        print("🧠 Initializing Comprehensive Behavioral Test Framework")
        print("=" * 60)
        
        self.robot_registry = RobotRegistry()
        
        brain_config = {
            'quiet_mode': quiet_mode,
            'spatial_resolution': 4
        }
        self.brain_factory = DynamicBrainFactory(brain_config)
        self.brain_pool = BrainPool(self.brain_factory)
        self.adapter_factory = AdapterFactory()
        self.brain_service = BrainService(self.brain_pool, self.adapter_factory)
        self.connection_handler = ConnectionHandler(self.robot_registry, self.brain_service)
        
        # Start monitoring server
        self.monitoring_server = DynamicMonitoringServer(
            brain_service=self.brain_service,
            connection_handler=self.connection_handler,
            host='localhost',
            port=9998
        )
        self.monitoring_server.start()
        
        # Telemetry client
        self.telemetry_client = TelemetryClient()
        
        self.client_id = "comprehensive_test_robot"
        self.session_id = None
        
        print("✅ Framework initialized")
    
    def setup_robot(self, sensory_dim: int = 16, motor_dim: int = 4):
        """Setup virtual robot"""
        capabilities = [1.0, float(sensory_dim), float(motor_dim), 0.0, 0.0]
        response = self.connection_handler.handle_handshake(self.client_id, capabilities)
        
        # Wait for session
        self.session_id = self.telemetry_client.wait_for_session(max_wait=2.0)
        
        print(f"\n🤖 Robot configured: {sensory_dim}D sensors, {motor_dim}D motors")
        return response
    
    def test_prediction_learning(self, cycles: int = 50) -> float:
        """Test prediction learning"""
        if not self.quiet_mode:
            print(f"   Running {cycles} cycles...")
        
        pattern = [0.5, 0.3, 0.8, 0.2] * 4
        confidence_values = []
        
        for i in range(cycles):
            self.connection_handler.handle_sensory_input(self.client_id, pattern)
            
            if self.session_id:
                telemetry = self.telemetry_client.get_session_telemetry(self.session_id)
                if telemetry:
                    confidence_values.append(telemetry.confidence)
        
        if len(confidence_values) >= 10:
            early = np.mean(confidence_values[:5])
            late = np.mean(confidence_values[-5:])
            improvement = late - early
            
            improvement_score = max(0, min(1.0, improvement * 5))
            final_score = max(0, min(1.0, (confidence_values[-1] - 0.5) * 2))
            
            return (improvement_score + final_score) / 2
        
        return 0.0
    
    def test_pattern_recognition(self, cycles: int = 60) -> float:
        """Test ability to distinguish patterns"""
        if not self.quiet_mode:
            print(f"   Running {cycles} cycles...")
        
        patterns = {
            'sine': lambda t: [np.sin(t * 0.1 + i * 0.5) for i in range(16)],
            'square': lambda t: [1.0 if (t + i) % 10 < 5 else 0.0 for i in range(16)],
            'random': lambda t: list(np.random.rand(16))
        }
        
        responses_by_pattern = {name: [] for name in patterns}
        
        for i in range(cycles):
            pattern_name = list(patterns.keys())[i % len(patterns)]
            pattern = patterns[pattern_name](i)
            
            motor_output = self.connection_handler.handle_sensory_input(
                self.client_id, pattern
            )
            
            response_energy = np.mean(np.abs(motor_output))
            responses_by_pattern[pattern_name].append(response_energy)
        
        mean_responses = {name: np.mean(responses) 
                         for name, responses in responses_by_pattern.items()
                         if len(responses) > 5}
        
        if len(mean_responses) >= 2:
            response_values = list(mean_responses.values())
            cv = np.std(response_values) / (np.mean(response_values) + 1e-6)
            return min(1.0, cv * 2)
        
        return 0.0
    
    def test_goal_seeking(self, cycles: int = 60) -> float:
        """Test goal-directed behavior"""
        if not self.quiet_mode:
            print(f"   Running {cycles} cycles...")
        
        goal_achievements = []
        
        for goal_position in [0, 8, 15]:
            goal_pattern = [0.1] * 16
            goal_pattern[goal_position] = 1.0
            
            motor_convergence = []
            
            for i in range(cycles // 3):
                motor_output = self.connection_handler.handle_sensory_input(
                    self.client_id, goal_pattern
                )
                
                if len(motor_output) >= 4:
                    motor_goal_dim = min(3, goal_position // 4)
                    motor_response = abs(motor_output[motor_goal_dim])
                    motor_convergence.append(motor_response)
            
            if len(motor_convergence) >= 10:
                early = np.mean(motor_convergence[:5])
                late = np.mean(motor_convergence[-5:])
                achievement = (late - early) / (early + 1e-6)
                goal_achievements.append(max(0, min(1.0, achievement)))
        
        return np.mean(goal_achievements) if goal_achievements else 0.0
    
    def test_exploration_exploitation(self, cycles: int = 50) -> float:
        """Test exploration vs exploitation"""
        if not self.quiet_mode:
            print(f"   Running {cycles} cycles...")
        
        if not self.session_id:
            return 0.0
        
        cognitive_modes = []
        motor_diversity = []
        
        for i in range(cycles):
            if i % 3 == 0:
                pattern = [1.0] + [0.0] * 15
            elif i % 3 == 1:
                pattern = [0.0, 1.0] + [0.0] * 14
            else:
                pattern = [0.0, 0.0, 1.0] + [0.0] * 13
            
            motor_output = self.connection_handler.handle_sensory_input(
                self.client_id, pattern
            )
            motor_diversity.append(motor_output[:4])
            
            telemetry = self.telemetry_client.get_session_telemetry(self.session_id)
            if telemetry:
                cognitive_modes.append(telemetry.mode)
        
        mode_switches = sum(1 for i in range(1, len(cognitive_modes)) 
                           if cognitive_modes[i] != cognitive_modes[i-1])
        
        motor_array = np.array(motor_diversity)
        motor_variance = np.mean(np.std(motor_array, axis=0))
        
        switch_score = min(1.0, mode_switches / (cycles * 0.1))
        diversity_score = min(1.0, motor_variance * 10)
        
        return (switch_score + diversity_score) / 2
    
    def test_field_stabilization(self, cycles: int = 30) -> float:
        """Test field stabilization"""
        if not self.quiet_mode:
            print(f"   Running {cycles} cycles...")
        
        if not self.session_id:
            return 0.0
        
        stable_input = [0.5] * 16
        energy_levels = []
        
        for i in range(cycles):
            self.connection_handler.handle_sensory_input(self.client_id, stable_input)
            
            telemetry = self.telemetry_client.get_session_telemetry(self.session_id)
            if telemetry:
                energy_levels.append(telemetry.energy)
        
        if len(energy_levels) >= 10:
            early_variance = np.var(energy_levels[:10])
            late_variance = np.var(energy_levels[-10:])
            
            if early_variance > 0:
                stability_improvement = 1.0 - (late_variance / early_variance)
                return max(0, min(1.0, stability_improvement))
        
        return 0.5
    
    def test_biological_realism(self, cycles: int = 50) -> float:
        """Test biological constraints"""
        if not self.quiet_mode:
            print(f"   Running {cycles} cycles...")
        
        if not self.session_id:
            return 0.0
        
        energy_levels = []
        phase_transitions = []
        cycle_times = []
        
        for i in range(cycles):
            pattern = [0.5 + 0.3 * np.sin(i * 0.1 + j * 0.2) for j in range(16)]
            
            start_time = time.time()
            self.connection_handler.handle_sensory_input(self.client_id, pattern)
            cycle_time = (time.time() - start_time) * 1000
            
            telemetry = self.telemetry_client.get_session_telemetry(self.session_id)
            if telemetry:
                energy_levels.append(telemetry.energy)
                phase_transitions.append(telemetry.phase)
                cycle_times.append(cycle_time)
        
        scores = []
        
        # Energy conservation
        if energy_levels:
            energy_var = np.var(energy_levels)
            energy_mean = np.mean(energy_levels)
            if energy_mean > 0:
                energy_stability = 1.0 / (1.0 + energy_var / energy_mean)
                scores.append(energy_stability)
        
        # Phase diversity
        if phase_transitions:
            unique_phases = len(set(phase_transitions))
            phase_diversity = min(1.0, unique_phases / 3.0)
            scores.append(phase_diversity)
        
        # Timing realism
        if cycle_times:
            avg_time = np.mean(cycle_times)
            timing_score = 1.0 if avg_time <= 150 else max(0, 1 - (avg_time - 150) / 500)
            scores.append(timing_score)
        
        return np.mean(scores) if scores else 0.0
    
    def test_computational_efficiency(self, cycles: int = 20) -> float:
        """Test computational efficiency"""
        if not self.quiet_mode:
            print(f"   Running {cycles} cycles...")
        
        cycle_times = []
        
        for i in range(cycles):
            sensory_input = list(np.random.randn(16) * 0.5)
            
            start_time = time.time()
            self.connection_handler.handle_sensory_input(self.client_id, sensory_input)
            cycle_time = (time.time() - start_time) * 1000
            
            if i >= 5:  # Skip warmup
                cycle_times.append(cycle_time)
        
        avg_time = np.mean(cycle_times)
        
        if avg_time <= 150:
            return 1.0
        elif avg_time <= 300:
            return 0.5 + (300 - avg_time) / 300
        else:
            return max(0, 1 - (avg_time - 300) / 1000)
    
    def test_paradigm_shifting(self, cycles: int = 100) -> float:
        """Test paradigm shifting ability"""
        if not self.quiet_mode:
            print(f"   Running {cycles} cycles...")
        
        # First paradigm: regular pattern
        pattern1 = [0.5, 0.3, 0.8, 0.2] * 4
        score1 = self.test_prediction_learning(cycles // 2)
        
        # Second paradigm: extreme pattern
        pattern2 = [0.01, 0.99, 0.02, 0.98] * 4
        confidence_values = []
        
        for i in range(cycles // 2):
            self.connection_handler.handle_sensory_input(self.client_id, pattern2)
            
            if self.session_id:
                telemetry = self.telemetry_client.get_session_telemetry(self.session_id)
                if telemetry:
                    confidence_values.append(telemetry.confidence)
        
        score2 = 0.0
        if len(confidence_values) >= 10:
            early = np.mean(confidence_values[:5])
            late = np.mean(confidence_values[-5:])
            improvement = late - early
            score2 = max(0, min(1.0, improvement * 5))
        
        return (score1 + score2) / 2
    
    def run_assessment(self, profile: IntelligenceProfile) -> Dict[str, Any]:
        """Run comprehensive assessment"""
        print(f"\n🚀 Comprehensive Assessment: {profile.name}")
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
        
        # Get stats
        stats = self.connection_handler.get_stats()
        print(f"\n📊 Session Statistics:")
        print(f"   Total messages: {stats['total_messages']}")
        
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
        print("\n✅ Test framework cleanup complete")


# Test profile
COMPREHENSIVE_PROFILE = IntelligenceProfile(
    name="Comprehensive Intelligence",
    targets=[
        BehavioralTarget(IntelligenceMetric.PREDICTION_LEARNING, 0.6, 0.05,
                        "Strong prediction learning", 100),
        BehavioralTarget(IntelligenceMetric.PATTERN_RECOGNITION, 0.5, 0.1,
                        "Pattern discrimination", 60),
        BehavioralTarget(IntelligenceMetric.GOAL_SEEKING, 0.4, 0.1,
                        "Goal-directed behavior", 60),
        BehavioralTarget(IntelligenceMetric.EXPLORATION_EXPLOITATION, 0.5, 0.1,
                        "E/E balance", 100),
        BehavioralTarget(IntelligenceMetric.FIELD_STABILIZATION, 0.5, 0.1,
                        "Field stability", 50),
        BehavioralTarget(IntelligenceMetric.BIOLOGICAL_REALISM, 0.6, 0.1,
                        "Biological constraints", 50),
        BehavioralTarget(IntelligenceMetric.PARADIGM_SHIFTING, 0.4, 0.1,
                        "Paradigm adaptation", 100),
        BehavioralTarget(IntelligenceMetric.COMPUTATIONAL_EFFICIENCY, 0.7, 0.1,
                        "Processing efficiency", 20)
    ]
)


def main():
    """Run comprehensive behavioral test"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Comprehensive Behavioral Test")
    parser.add_argument('--quiet', action='store_true', help='Quiet mode')
    args = parser.parse_args()
    
    print("🧠 Comprehensive Behavioral Intelligence Test")
    print("=" * 50)
    
    framework = ComprehensiveBehavioralTest(quiet_mode=args.quiet)
    
    try:
        results = framework.run_assessment(COMPREHENSIVE_PROFILE)
        
        print("\n" + "=" * 50)
        print("📊 FINAL RESULTS")
        print("=" * 50)
        
        for metric, details in results['detailed_results'].items():
            status = "✅" if details['achieved'] else "❌"
            print(f"{status} {metric}: {details['score']:.3f} / {details['target']:.3f}")
        
        print(f"\n🏆 Overall Achievement: {results['overall_achievement']:.1%}")
        
        if results['overall_achievement'] >= 0.8:
            print("🎉 Brain demonstrates excellent comprehensive intelligence!")
        elif results['overall_achievement'] >= 0.6:
            print("👍 Brain shows good intelligence across all metrics")
        else:
            print("🔧 Brain needs optimization in some areas")
            
    finally:
        framework.cleanup()


if __name__ == "__main__":
    main()