#!/usr/bin/env python3
"""
Dynamic Behavioral Test Framework

Tests behavioral intelligence using the new dynamic brain architecture.
Creates a virtual robot client and tests the brain's learning capabilities.
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

# Import new architecture components
from src.core.robot_registry import RobotRegistry
from src.core.brain_pool import BrainPool
from src.core.brain_service import BrainService
from src.core.adapters import AdapterFactory
from src.core.connection_handler import ConnectionHandler
from src.core.dynamic_brain_factory import DynamicBrainFactory


class IntelligenceMetric(Enum):
    """Core intelligence metrics"""
    PREDICTION_LEARNING = "prediction_learning"
    EXPLORATION_EXPLOITATION = "exploration_exploitation" 
    COMPUTATIONAL_EFFICIENCY = "computational_efficiency"


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


class DynamicBehavioralTestFramework:
    """Behavioral test framework using dynamic brain architecture"""
    
    def __init__(self, use_simple_brain: bool = False, quiet_mode: bool = False):
        self.quiet_mode = quiet_mode
        self.use_simple_brain = use_simple_brain
        
        # Initialize dynamic architecture components
        print("🧠 Initializing Dynamic Behavioral Test Framework")
        print("=" * 60)
        
        # Robot registry
        self.robot_registry = RobotRegistry()
        
        # Brain components
        brain_config = {
            'quiet_mode': quiet_mode,
            'use_simple_brain': use_simple_brain,
            'spatial_resolution': 4  # Keep low for fast tests
        }
        self.brain_factory = DynamicBrainFactory(brain_config)
        self.brain_pool = BrainPool(self.brain_factory)
        self.adapter_factory = AdapterFactory()
        self.brain_service = BrainService(self.brain_pool, self.adapter_factory)
        self.connection_handler = ConnectionHandler(self.robot_registry, self.brain_service)
        
        # Virtual robot client ID
        self.client_id = "test_behavioral_robot"
        self.session = None
        
        print(f"✅ Framework initialized with {'simple' if use_simple_brain else 'unified'} brain")
    
    def setup_virtual_robot(self, sensory_dim: int = 16, motor_dim: int = 4):
        """Setup a virtual robot for testing"""
        # Create capabilities vector (extended handshake format)
        capabilities = [
            1.0,  # robot_version
            float(sensory_dim),  # sensory dimensions
            float(motor_dim),    # motor dimensions  
            0.0,  # hardware_type (0 = generic)
            0.0   # capabilities_mask
        ]
        
        # Perform handshake
        response = self.connection_handler.handle_handshake(self.client_id, capabilities)
        
        print(f"\n🤖 Virtual robot configured:")
        print(f"   Sensory: {sensory_dim}D")
        print(f"   Motor: {motor_dim}D")
        print(f"   Brain response: {response}")
        
        return response
    
    def test_prediction_learning(self, cycles: int = 50) -> float:
        """Test prediction learning capability"""
        prediction_errors = []
        
        # Create predictable sine wave pattern
        for i in range(cycles):
            # Generate sensory input with pattern
            phase = i * 0.1
            sensory_input = [
                np.sin(phase),
                np.cos(phase),
                np.sin(phase * 2),
                np.cos(phase * 2)
            ] + [0.1] * 12  # Pad to 16D
            
            # Process through brain
            motor_output = self.connection_handler.handle_sensory_input(
                self.client_id, sensory_input
            )
            
            # Simple heuristic: if motor output follows a pattern, prediction is working
            if i > cycles // 2:
                # Check if motor output is becoming more consistent
                if i > 0 and len(prediction_errors) > 0:
                    variation = np.std(motor_output)
                    prediction_errors.append(variation)
        
        # Score based on decreasing variation (learning the pattern)
        if len(prediction_errors) > 2:
            early_error = np.mean(prediction_errors[:5])
            late_error = np.mean(prediction_errors[-5:])
            improvement = max(0, (early_error - late_error) / (early_error + 0.001))
            return min(1.0, improvement * 2)  # Scale up
        return 0.0
    
    def test_exploration_exploitation(self, cycles: int = 50) -> float:
        """Test exploration vs exploitation balance"""
        motor_outputs = []
        
        # Present three different stimulus types
        for i in range(cycles):
            if i % 3 == 0:
                sensory_input = [1.0, 0.0, 0.0] + [0.0] * 13
            elif i % 3 == 1:
                sensory_input = [0.0, 1.0, 0.0] + [0.0] * 13
            else:
                sensory_input = [0.0, 0.0, 1.0] + [0.0] * 13
                
            motor_output = self.connection_handler.handle_sensory_input(
                self.client_id, sensory_input
            )
            motor_outputs.append(motor_output[:3])  # First 3 motors
        
        # Measure behavioral diversity
        motor_outputs = np.array(motor_outputs)
        diversity = np.mean(np.std(motor_outputs, axis=0))
        
        # Also check for switching behavior (not stuck)
        switches = 0
        for i in range(1, len(motor_outputs)):
            if np.linalg.norm(motor_outputs[i] - motor_outputs[i-1]) > 0.1:
                switches += 1
        
        switch_rate = switches / len(motor_outputs)
        
        # Combined score
        return min(1.0, diversity * 5 + switch_rate)
    
    def test_computational_efficiency(self, cycles: int = 20) -> float:
        """Test computational efficiency"""
        cycle_times = []
        
        for i in range(cycles):
            # Random sensory input
            sensory_input = [np.random.randn() * 0.5 for _ in range(16)]
            
            start_time = time.time()
            motor_output = self.connection_handler.handle_sensory_input(
                self.client_id, sensory_input
            )
            cycle_time = (time.time() - start_time) * 1000
            
            if i >= 5:  # Skip warmup cycles
                cycle_times.append(cycle_time)
        
        avg_cycle_time = np.mean(cycle_times)
        
        # Score based on meeting biological timing constraint
        if avg_cycle_time <= 150:
            return 1.0
        elif avg_cycle_time <= 300:
            return 0.5 + (300 - avg_cycle_time) / 300
        else:
            return max(0, 1 - (avg_cycle_time - 300) / 1000)
    
    def run_assessment(self, profile: IntelligenceProfile) -> Dict[str, Any]:
        """Run intelligence assessment"""
        print(f"\n🚀 Intelligence Assessment: {profile.name}")
        print("=" * 60)
        
        # Setup virtual robot
        self.setup_virtual_robot()
        
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
        overall_achievement = profile.overall_achievement(results)
        
        print(f"\n🎯 Overall Intelligence Achievement: {overall_achievement:.1%}")
        print(f"⏱️  Total test time: {total_time:.1f}s")
        
        # Show brain configuration
        stats = self.connection_handler.get_stats()
        print(f"\n📊 Session Statistics:")
        print(f"   Total messages: {stats['total_messages']}")
        
        return {
            'profile_name': profile.name,
            'overall_achievement': overall_achievement,
            'results': results,
            'detailed_results': detailed_results,
            'total_test_time': total_time
        }
    
    def cleanup(self):
        """Clean shutdown"""
        if self.connection_handler and self.client_id:
            self.connection_handler.handle_disconnect(self.client_id)
        print("\n✅ Test framework cleanup complete")


# Test profiles
BASIC_INTELLIGENCE_PROFILE = IntelligenceProfile(
    name="Basic Intelligence",
    targets=[
        BehavioralTarget(
            IntelligenceMetric.PREDICTION_LEARNING, 0.3, 0.05, 
            "Brain should show learning improvement", 50
        ),
        BehavioralTarget(
            IntelligenceMetric.EXPLORATION_EXPLOITATION, 0.4, 0.1,
            "Brain should balance exploration/exploitation", 50
        ),
        BehavioralTarget(
            IntelligenceMetric.COMPUTATIONAL_EFFICIENCY, 0.7, 0.1,
            "Brain should meet biological timing", 20
        )
    ]
)

FAST_INTELLIGENCE_PROFILE = IntelligenceProfile(
    name="Fast Intelligence Check",
    targets=[
        BehavioralTarget(
            IntelligenceMetric.PREDICTION_LEARNING, 0.2, 0.05, 
            "Basic prediction capability", 20
        ),
        BehavioralTarget(
            IntelligenceMetric.EXPLORATION_EXPLOITATION, 0.3, 0.1,
            "Some behavioral variety", 30
        ),
        BehavioralTarget(
            IntelligenceMetric.COMPUTATIONAL_EFFICIENCY, 0.7, 0.1,
            "Fast processing", 10
        )
    ]
)


def main():
    """Run behavioral test with dynamic brain"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Dynamic Behavioral Test")
    parser.add_argument('--simple', action='store_true', help='Use simple brain implementation')
    parser.add_argument('--fast', action='store_true', help='Run fast test profile')
    parser.add_argument('--quiet', action='store_true', help='Quiet mode')
    args = parser.parse_args()
    
    print("🧠 Dynamic Behavioral Intelligence Test")
    print("=" * 50)
    
    framework = DynamicBehavioralTestFramework(
        use_simple_brain=args.simple,
        quiet_mode=args.quiet
    )
    
    try:
        # Choose profile
        profile = FAST_INTELLIGENCE_PROFILE if args.fast else BASIC_INTELLIGENCE_PROFILE
        
        # Run assessment
        results = framework.run_assessment(profile)
        
        # Summary
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
            print("👍 Brain shows good intelligence capabilities")
        else:
            print("🔧 Brain needs further optimization")
            
    finally:
        framework.cleanup()


if __name__ == "__main__":
    main()