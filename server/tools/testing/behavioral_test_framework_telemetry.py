#!/usr/bin/env python3
"""
Behavioral Test Framework with Telemetry Support

Updated version of behavioral_test_framework.py that uses the telemetry system
for accurate brain state monitoring and improved performance.
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

# Import new architecture components for telemetry-based testing
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
    COMPUTATIONAL_EFFICIENCY = "computational_efficiency"


@dataclass
class BehavioralTarget:
    """Behavioral target for testing"""
    metric: IntelligenceMetric
    target_value: float
    tolerance: float
    description: str
    test_duration_cycles: int = 50  # Reduced from 100 for efficiency
    
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


class TelemetryBehavioralTestFramework:
    """Behavioral test framework using telemetry for accurate brain monitoring"""
    
    def __init__(self, quiet_mode: bool = True):
        self.quiet_mode = quiet_mode
        
        # Initialize components
        print("🧠 Initializing Telemetry-Based Behavioral Test Framework")
        print("=" * 60)
        
        # Create brain infrastructure
        self.robot_registry = RobotRegistry()
        
        brain_config = {
            'quiet_mode': quiet_mode,
            'use_simple_brain': False,
            'spatial_resolution': 4  # Keep low for performance
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
        
        # Virtual robot
        self.client_id = "behavioral_test_robot"
        self.session_id = None
        
        print("✅ Framework initialized with telemetry support")
    
    def setup_robot(self, sensory_dim: int = 16, motor_dim: int = 4):
        """Setup virtual robot for testing"""
        capabilities = [1.0, float(sensory_dim), float(motor_dim), 0.0, 0.0]
        response = self.connection_handler.handle_handshake(self.client_id, capabilities)
        
        # Wait for session
        self.session_id = self.telemetry_client.wait_for_session(max_wait=2.0)
        
        print(f"🤖 Robot configured: {sensory_dim}D sensors, {motor_dim}D motors")
        return response
    
    def test_prediction_learning(self, cycles: int = 50) -> float:
        """Test prediction learning using telemetry"""
        
        if not self.session_id:
            print("⚠️  No session available")
            return 0.0
        
        # Create repeating pattern
        pattern = [0.5, 0.3, 0.8, 0.2] * 4  # 16D pattern
        
        confidence_history = []
        
        for i in range(cycles):
            # Process pattern
            motor_output = self.connection_handler.handle_sensory_input(
                self.client_id, pattern
            )
            
            # Get telemetry
            telemetry = self.telemetry_client.get_session_telemetry(self.session_id)
            if telemetry:
                confidence_history.append(telemetry.confidence)
        
        # Calculate improvement
        if len(confidence_history) >= 10:
            early = np.mean(confidence_history[:5])
            late = np.mean(confidence_history[-5:])
            improvement = late - early
            
            # Score based on improvement and final confidence
            improvement_score = max(0, min(1.0, improvement * 5))
            final_score = max(0, min(1.0, (confidence_history[-1] - 0.5) * 2))
            
            return (improvement_score + final_score) / 2
        
        return 0.0
    
    def test_exploration_exploitation(self, cycles: int = 50) -> float:
        """Test exploration vs exploitation using telemetry"""
        
        if not self.session_id:
            return 0.0
        
        cognitive_modes = []
        energy_levels = []
        motor_diversity = []
        
        # Present varied stimuli
        for i in range(cycles):
            # Alternate between different patterns
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
            
            # Get telemetry
            telemetry = self.telemetry_client.get_session_telemetry(self.session_id)
            if telemetry:
                cognitive_modes.append(telemetry.mode)
                energy_levels.append(telemetry.energy)
        
        # Calculate metrics
        mode_switches = sum(1 for i in range(1, len(cognitive_modes)) 
                           if cognitive_modes[i] != cognitive_modes[i-1])
        
        motor_array = np.array(motor_diversity)
        motor_variance = np.mean(np.std(motor_array, axis=0))
        
        # Score
        switch_score = min(1.0, mode_switches / (cycles * 0.1))  # Expect ~10% switches
        diversity_score = min(1.0, motor_variance * 10)
        
        return (switch_score + diversity_score) / 2
    
    def test_field_stabilization(self, cycles: int = 30) -> float:
        """Test field stabilization using telemetry"""
        
        if not self.session_id:
            return 0.0
        
        energy_levels = []
        
        # Constant input for stabilization test
        stable_input = [0.5] * 16
        
        for i in range(cycles):
            self.connection_handler.handle_sensory_input(self.client_id, stable_input)
            
            telemetry = self.telemetry_client.get_session_telemetry(self.session_id)
            if telemetry:
                energy_levels.append(telemetry.energy)
        
        if len(energy_levels) >= 10:
            # Check if energy stabilizes
            early_variance = np.var(energy_levels[:10])
            late_variance = np.var(energy_levels[-10:])
            
            # Good stabilization: decreasing variance
            if early_variance > 0:
                stability_improvement = 1.0 - (late_variance / early_variance)
                return max(0, min(1.0, stability_improvement))
        
        return 0.5
    
    def test_computational_efficiency(self, cycles: int = 20) -> float:
        """Test computational efficiency"""
        
        cycle_times = []
        
        for i in range(cycles):
            start_time = time.time()
            
            sensory_input = list(np.random.randn(16) * 0.5)
            self.connection_handler.handle_sensory_input(self.client_id, sensory_input)
            
            cycle_time = (time.time() - start_time) * 1000
            
            if i >= 5:  # Skip warmup
                cycle_times.append(cycle_time)
        
        avg_time = np.mean(cycle_times)
        
        # Score based on biological timing (150ms target)
        if avg_time <= 150:
            return 1.0
        elif avg_time <= 300:
            return 0.5 + (300 - avg_time) / 300
        else:
            return max(0, 1 - (avg_time - 300) / 1000)
    
    def run_assessment(self, profile: IntelligenceProfile) -> Dict[str, Any]:
        """Run intelligence assessment"""
        print(f"\n🚀 Intelligence Assessment: {profile.name}")
        print("=" * 60)
        
        # Setup robot
        self.setup_robot()
        
        results = {}
        detailed_results = {}
        
        total_start = time.time()
        
        for target in profile.targets:
            print(f"\n📊 Testing {target.metric.value}...")
            
            test_start = time.time()
            
            if target.metric == IntelligenceMetric.PREDICTION_LEARNING:
                score = self.test_prediction_learning(target.test_duration_cycles)
            elif target.metric == IntelligenceMetric.EXPLORATION_EXPLOITATION:
                score = self.test_exploration_exploitation(target.test_duration_cycles)
            elif target.metric == IntelligenceMetric.FIELD_STABILIZATION:
                score = self.test_field_stabilization(target.test_duration_cycles)
            elif target.metric == IntelligenceMetric.COMPUTATIONAL_EFFICIENCY:
                score = self.test_computational_efficiency(target.test_duration_cycles)
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
EFFICIENT_INTELLIGENCE_PROFILE = IntelligenceProfile(
    name="Efficient Intelligence",
    targets=[
        BehavioralTarget(
            IntelligenceMetric.PREDICTION_LEARNING, 0.3, 0.05,
            "Basic prediction capability", 30
        ),
        BehavioralTarget(
            IntelligenceMetric.EXPLORATION_EXPLOITATION, 0.3, 0.1,
            "Balanced behavior", 30
        ),
        BehavioralTarget(
            IntelligenceMetric.FIELD_STABILIZATION, 0.5, 0.1,
            "Stable field dynamics", 20
        ),
        BehavioralTarget(
            IntelligenceMetric.COMPUTATIONAL_EFFICIENCY, 0.7, 0.1,
            "Fast processing", 20
        )
    ]
)


def main():
    """Run telemetry-based behavioral test"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Telemetry Behavioral Test")
    parser.add_argument('--quiet', action='store_true', help='Quiet mode')
    args = parser.parse_args()
    
    print("🧠 Telemetry-Based Behavioral Intelligence Test")
    print("=" * 50)
    
    framework = TelemetryBehavioralTestFramework(quiet_mode=args.quiet)
    
    try:
        results = framework.run_assessment(EFFICIENT_INTELLIGENCE_PROFILE)
        
        print("\n" + "=" * 50)
        print("📊 FINAL RESULTS")
        print("=" * 50)
        
        for metric, details in results['detailed_results'].items():
            status = "✅" if details['achieved'] else "❌"
            print(f"{status} {metric}: {details['score']:.3f} / {details['target']:.3f}")
        
        print(f"\n🏆 Overall Achievement: {results['overall_achievement']:.1%}")
        
        if results['overall_achievement'] >= 0.8:
            print("🎉 Brain demonstrates strong intelligence!")
        elif results['overall_achievement'] >= 0.6:
            print("👍 Brain shows good intelligence")
        else:
            print("🔧 Brain needs optimization")
            
    finally:
        framework.cleanup()


if __name__ == "__main__":
    main()