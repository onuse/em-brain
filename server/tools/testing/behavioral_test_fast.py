#!/usr/bin/env python3
"""
Fast Behavioral Test Framework

A streamlined version that runs quickly while still testing core intelligence.
Reduces cycle counts and focuses on essential metrics.
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

from src.brain_factory import BrainFactory


class IntelligenceMetric(Enum):
    """Core intelligence metrics"""
    PREDICTION_LEARNING = "prediction_learning"
    EXPLORATION_EXPLOITATION = "exploration_exploitation" 
    COMPUTATIONAL_EFFICIENCY = "computational_efficiency"


@dataclass
class FastBehavioralTarget:
    """Fast behavioral target with reduced cycle counts"""
    metric: IntelligenceMetric
    target_value: float
    tolerance: float
    description: str
    test_duration_cycles: int = 20  # Much faster!
    
    def is_achieved(self, measured_value: float) -> bool:
        return measured_value >= (self.target_value - self.tolerance)


@dataclass 
class FastIntelligenceProfile:
    """Fast intelligence profile"""
    name: str
    targets: List[FastBehavioralTarget]
    
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


class FastBehavioralTestFramework:
    """Fast behavioral test framework"""
    
    def __init__(self, quiet_mode: bool = False):
        self.quiet_mode = quiet_mode
        self.shared_brain = None  # Reuse brain across tests
        self.warmup_done = False
        
    def create_brain(self, config: Dict[str, Any] = None) -> BrainFactory:
        """Create or reuse a brain for testing"""
        if self.shared_brain is not None:
            return self.shared_brain
            
        default_config = {
            'memory': {'enable_persistence': True},
            'brain': {
                'field_spatial_resolution': 4,
                'target_cycle_time_ms': 150,
                'field_evolution_rate': 0.1,
                'constraint_discovery_rate': 0.15
            }
        }
        
        if config:
            default_config['brain'].update(config.get('brain', {}))
        
        self.shared_brain = BrainFactory(
            config=default_config,
            enable_logging=False,
            quiet_mode=self.quiet_mode
        )
        
        # Shared warmup
        if not self.warmup_done:
            print("🔥 Warming up brain...")
            for i in range(10):
                sensory_input = [np.sin(i * 0.1), np.cos(i * 0.1), 0.0] + [0.1] * 13
                self.shared_brain.process_sensory_input(sensory_input)
            self.warmup_done = True
            print("✅ Warmup complete")
        
        return self.shared_brain
    
    def test_prediction_learning(self, brain: BrainFactory, cycles: int = 20) -> float:
        """Fast prediction learning test"""
        # Simple pattern: sine wave
        prediction_errors = []
        
        for i in range(cycles):
            # Create predictable pattern
            sensory_input = [np.sin(i * 0.2), np.cos(i * 0.2), 0.0] + [0.1] * 13
            
            action, state = brain.process_sensory_input(sensory_input)
            confidence = state.get('prediction_confidence', 0.5)
            
            # Track improvement
            if i > cycles // 2:
                prediction_errors.append(1.0 - confidence)
        
        # Score based on final vs initial error
        if prediction_errors:
            improvement = 1.0 - np.mean(prediction_errors[-3:])
            return max(0.0, improvement)
        return 0.0
    
    def test_exploration_exploitation(self, brain: BrainFactory, cycles: int = 30) -> float:
        """Fast exploration vs exploitation test"""
        actions = []
        
        # Present varied stimuli
        for i in range(cycles):
            if i % 3 == 0:
                sensory_input = [1.0, 0.0, 0.0] + [0.0] * 13
            elif i % 3 == 1:
                sensory_input = [0.0, 1.0, 0.0] + [0.0] * 13
            else:
                sensory_input = [0.0, 0.0, 1.0] + [0.0] * 13
                
            action, _ = brain.process_sensory_input(sensory_input)
            actions.append(action[:3])
        
        # Measure behavioral diversity
        actions = np.array(actions)
        diversity = np.mean(np.std(actions, axis=0))
        
        # Score based on appropriate diversity
        return min(1.0, diversity * 10)  # Scale up small values
    
    def test_computational_efficiency(self, brain: BrainFactory, cycles: int = 10) -> float:
        """Fast computational efficiency test"""
        cycle_times = []
        
        for i in range(cycles):
            sensory_input = [np.random.randn() * 0.1 for _ in range(16)]
            
            start_time = time.time()
            action, state = brain.process_sensory_input(sensory_input)
            cycle_time = (time.time() - start_time) * 1000
            
            cycle_times.append(cycle_time)
        
        avg_cycle_time = np.mean(cycle_times[2:])  # Skip first 2 (warmup)
        
        # Score based on meeting biological constraint
        if avg_cycle_time <= 150:
            return 1.0
        elif avg_cycle_time <= 300:
            return 0.5
        else:
            return 0.0
    
    def run_fast_assessment(self, brain: BrainFactory, profile: FastIntelligenceProfile) -> Dict[str, Any]:
        """Run fast intelligence assessment"""
        print(f"\n🚀 Fast Intelligence Assessment: {profile.name}")
        print("=" * 60)
        
        results = {}
        detailed_results = {}
        
        total_start = time.time()
        
        for target in profile.targets:
            print(f"📊 Testing {target.metric.value}...", end='', flush=True)
            
            test_start = time.time()
            
            if target.metric == IntelligenceMetric.PREDICTION_LEARNING:
                score = self.test_prediction_learning(brain, target.test_duration_cycles)
            elif target.metric == IntelligenceMetric.EXPLORATION_EXPLOITATION:
                score = self.test_exploration_exploitation(brain, target.test_duration_cycles)
            elif target.metric == IntelligenceMetric.COMPUTATIONAL_EFFICIENCY:
                score = self.test_computational_efficiency(brain, target.test_duration_cycles)
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
            print(f"\n   Score: {score:.3f} / Target: {target.target_value:.3f} {status}")
        
        total_time = time.time() - total_start
        overall_achievement = profile.overall_achievement(results)
        
        print(f"\n🎯 Overall Intelligence Achievement: {overall_achievement:.1%}")
        print(f"⏱️  Total test time: {total_time:.1f}s")
        
        return {
            'profile_name': profile.name,
            'overall_achievement': overall_achievement,
            'results': results,
            'detailed_results': detailed_results,
            'total_test_time': total_time
        }
    
    def cleanup(self):
        """Clean shutdown"""
        if self.shared_brain:
            self.shared_brain.shutdown()
            self.shared_brain = None
            self.warmup_done = False


# Fast test profile
FAST_INTELLIGENCE_PROFILE = FastIntelligenceProfile(
    name="Fast Basic Intelligence",
    targets=[
        FastBehavioralTarget(
            IntelligenceMetric.PREDICTION_LEARNING, 0.3, 0.05, 
            "Brain should improve predictions", 20
        ),
        FastBehavioralTarget(
            IntelligenceMetric.EXPLORATION_EXPLOITATION, 0.5, 0.1,
            "Brain should show behavioral variety", 30
        ),
        FastBehavioralTarget(
            IntelligenceMetric.COMPUTATIONAL_EFFICIENCY, 0.7, 0.1,
            "Brain should process efficiently", 10
        )
    ]
)


def main():
    """Run fast behavioral test"""
    print("🧠 Fast Behavioral Intelligence Test")
    print("=" * 50)
    
    framework = FastBehavioralTestFramework(quiet_mode=True)
    
    try:
        # Create brain
        brain = framework.create_brain()
        
        # Run assessment
        results = framework.run_fast_assessment(brain, FAST_INTELLIGENCE_PROFILE)
        
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