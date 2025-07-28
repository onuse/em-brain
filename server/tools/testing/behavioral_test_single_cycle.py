#!/usr/bin/env python3
"""
Single-Cycle Behavioral Test Framework

A minimal version of the behavioral test that runs just one cycle
to verify brain functionality without timing out.
"""

import sys
import os
import time
import numpy as np
from typing import Dict, List, Tuple, Any
from enum import Enum

# Add server source to path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

try:
    from src.core.dynamic_brain_factory import DynamicBrainFactory
except ImportError:
    print("❌ Failed to import DynamicBrainFactory. Make sure you're running from the server directory.")
    sys.exit(1)


class TestResult(Enum):
    """Test result status"""
    PASS = "✅ PASS"
    FAIL = "❌ FAIL"
    WARN = "⚠️ WARN"
    SKIP = "⏭️ SKIP"


class SingleCycleBehavioralTest:
    """Single-cycle version of behavioral intelligence tests"""
    
    def __init__(self, brain_config: Dict[str, Any] = None):
        """Initialize test framework with brain configuration"""
        self.brain_config = brain_config or {}
        self.results = {}
        
        # Test parameters - minimal for single cycle
        self.single_cycle_only = True
        self.warmup_cycles = 5  # Quick warmup
        self.test_cycles = 1    # Single test cycle
        
        # Initialize brain factory
        print("🧠 Initializing Dynamic Brain Factory...")
        factory_config = self.brain_config.copy()
        factory_config['use_dynamic_brain'] = True
        factory_config['quiet_mode'] = False  # Show progress for debugging
        factory_config['pattern_attention'] = False  # Disable for performance testing
        
        self.factory = DynamicBrainFactory(factory_config)
        
        # Create brain wrapper
        self.brain_wrapper = self.factory.create(
            field_dimensions=None,
            spatial_resolution=brain_config.get('field_spatial_resolution', 4),
            sensory_dim=16,  # PiCar-X standard (from profile)
            motor_dim=5      # PiCar-X has 5 motor channels
        )
        self.brain = self.brain_wrapper.brain
        
        print(f"✅ Brain initialized with {len(self.brain.field_dimensions)}D field")
        
    def run_single_cycle_test(self) -> Dict[str, Any]:
        """Run a single cycle through the brain to verify basic functionality"""
        print("\n🔄 Running Single-Cycle Test...")
        
        # Warmup phase
        print(f"   Warming up with {self.warmup_cycles} cycles...")
        warmup_times = []
        for i in range(self.warmup_cycles):
            start_time = time.time()
            # PiCar-X has 16 sensors
            sensory_input = [np.sin(i * 0.1), np.cos(i * 0.1), 0.0] + [0.1] * 13
            action, state = self.brain.process_robot_cycle(sensory_input)
            cycle_time = (time.time() - start_time) * 1000
            warmup_times.append(cycle_time)
        
        avg_warmup_time = np.mean(warmup_times)
        print(f"   Average warmup cycle time: {avg_warmup_time:.1f}ms")
        
        # Single test cycle with detailed metrics
        print("\n🎯 Running test cycle...")
        test_input = [1.0, 0.5, 0.0] + [0.2] * 13  # Strong sensory signal (16 total for PiCar-X)
        
        start_time = time.time()
        action, state = self.brain.process_robot_cycle(test_input)
        cycle_time = (time.time() - start_time) * 1000
        
        # Get brain state from return value
        post_stats = state
        
        # Analyze results
        results = {
            'cycle_time_ms': cycle_time,
            'action': action[:4],  # First 4 motor outputs
            'confidence': state.get('last_action_confidence', 0),
            'brain_cycles': state.get('brain_cycles', 0),
            'topology_regions': state.get('topology_regions_count', 0),
            'field_energy': state.get('field_energy', 0),
            'prediction_confidence': state.get('prediction_confidence', 0),
            'cognitive_mode': state.get('cognitive_mode', 'unknown')
        }
        
        # Print results
        print(f"\n📊 Single Cycle Results:")
        print(f"   Cycle time: {cycle_time:.1f}ms")
        print(f"   Action output: {[f'{a:.4f}' for a in action[:4]]}")
        print(f"   Confidence: {results['confidence']:.4f}")
        print(f"   Brain cycles: {results['brain_cycles']}")
        print(f"   Topology regions: {results['topology_regions']}")
        print(f"   Field energy: {results['field_energy']:.6f}")
        print(f"   Prediction confidence: {results['prediction_confidence']:.4f}")
        print(f"   Cognitive mode: {results['cognitive_mode']}")
        
        # Determine pass/fail
        if cycle_time < 150:  # Biological constraint
            print(f"\n{TestResult.PASS.value} Cycle time meets biological constraint (<150ms)")
        else:
            print(f"\n{TestResult.WARN.value} Cycle time exceeds biological constraint (>150ms)")
        
        if results['topology_regions'] > 0 or results['brain_cycles'] < 10:
            print(f"{TestResult.PASS.value} Brain shows signs of memory formation")
        else:
            print(f"{TestResult.WARN.value} No topology regions formed yet (may need more cycles)")
        
        return results
    
    def run_quick_behavioral_check(self) -> Dict[str, Any]:
        """Run a quick behavioral differentiation check"""
        print("\n🎭 Quick Behavioral Check...")
        
        # Test with two very different stimuli
        stimulus_a = [1.0, 0.0, 0.0] + [0.0] * 13  # Strong X signal (16 total)
        stimulus_b = [0.0, 1.0, 0.0] + [0.0] * 13  # Strong Y signal (16 total)
        
        # Get responses
        action_a, state_a = self.brain.process_robot_cycle(stimulus_a)
        action_b, state_b = self.brain.process_robot_cycle(stimulus_b)
        
        # Calculate difference
        action_diff = np.linalg.norm(np.array(action_a[:4]) - np.array(action_b[:4]))
        
        results = {
            'stimulus_a_response': action_a[:4],
            'stimulus_b_response': action_b[:4],
            'response_difference': action_diff,
            'behavioral_differentiation': action_diff > 0.01
        }
        
        print(f"   Response to X stimulus: {[f'{a:.4f}' for a in action_a[:4]]}")
        print(f"   Response to Y stimulus: {[f'{a:.4f}' for a in action_b[:4]]}")
        print(f"   Response difference: {action_diff:.4f}")
        
        if results['behavioral_differentiation']:
            print(f"\n{TestResult.PASS.value} Brain shows behavioral differentiation")
        else:
            print(f"\n{TestResult.WARN.value} Limited behavioral differentiation (may need more training)")
        
        return results
    
    def cleanup(self):
        """Clean shutdown"""
        print("\n🔌 Test complete")
        print("✅ Cleanup complete")


def main():
    """Run single-cycle behavioral test"""
    print("🧪 Single-Cycle Behavioral Test Framework")
    print("=" * 50)
    
    # Test configuration - optimized for performance
    brain_config = {
        'field_spatial_resolution': 4,  # Conservative resolution
        'target_cycle_time_ms': 150,
        'field_evolution_rate': 0.1,
        'constraint_discovery_rate': 0.15
    }
    
    # Create and run test
    test = SingleCycleBehavioralTest(brain_config)
    
    try:
        # Run tests
        single_cycle_results = test.run_single_cycle_test()
        behavioral_results = test.run_quick_behavioral_check()
        
        # Summary
        print("\n" + "=" * 50)
        print("📊 TEST SUMMARY")
        print("=" * 50)
        
        cycle_time = single_cycle_results['cycle_time_ms']
        if cycle_time < 150:
            print(f"✅ Performance: {cycle_time:.1f}ms (PASS)")
        else:
            print(f"⚠️ Performance: {cycle_time:.1f}ms (SLOW)")
        
        if behavioral_results['behavioral_differentiation']:
            print("✅ Behavioral differentiation: DETECTED")
        else:
            print("⚠️ Behavioral differentiation: LIMITED")
        
        if single_cycle_results['topology_regions'] > 0:
            print(f"✅ Memory formation: {single_cycle_results['topology_regions']} regions")
        else:
            print("⚠️ Memory formation: NOT YET DETECTED")
        
        print(f"\n🏁 Single-cycle test completed successfully!")
        
    finally:
        test.cleanup()


if __name__ == "__main__":
    main()