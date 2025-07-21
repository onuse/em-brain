#!/usr/bin/env python3
"""
Field Brain Learning Stability Test

Quick test (2-3 minutes) to detect if the field brain maintains learning capability
or degrades over time. This replicates the biological embodied learning degradation
pattern but in an accelerated timeframe.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import time
import random
import numpy as np
from typing import List, Dict, Tuple

# Add server path for imports
server_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src')
sys.path.insert(0, server_path)
sys.path.insert(0, os.path.join(server_path, '..'))

from src.brain_factory import BrainFactory


class LearningStabilityTest:
    """Test field brain learning stability over time."""
    
    def __init__(self):
        self.config = {
            'brain': {
                'type': 'field',
                'sensory_dim': 8,
                'motor_dim': 4,
                'enable_enhanced_dynamics': False,
                'enable_attention_guidance': False,
                'enable_hierarchical_processing': False
            }
        }
        self.brain = BrainFactory(config=self.config, quiet_mode=True)
        
        # Learning pattern: simple navigation-like task
        self.light_position = [0.8, 0.2]  # Target position
        self.robot_position = [0.1, 0.1]  # Start position
        
        # Tracking metrics
        self.learning_history = []
        self.field_stats_history = []
        
    def generate_sensory_input(self, noise_level: float = 0.1) -> List[float]:
        """Generate navigation-like sensory input with target seeking."""
        # Distance and direction to light
        dx = self.light_position[0] - self.robot_position[0]
        dy = self.light_position[1] - self.robot_position[1]
        distance = np.sqrt(dx*dx + dy*dy)
        
        # Normalize direction
        if distance > 0.01:
            dx_norm = dx / distance
            dy_norm = dy / distance
        else:
            dx_norm = dy_norm = 0.0
        
        # Create 8D sensor input: [light_x, light_y, dist, angle, obstacles...]
        sensors = [
            self.light_position[0] + random.uniform(-noise_level, noise_level),
            self.light_position[1] + random.uniform(-noise_level, noise_level),
            min(1.0, distance) + random.uniform(-noise_level, noise_level),
            dx_norm + random.uniform(-noise_level, noise_level),
            dy_norm + random.uniform(-noise_level, noise_level),
            random.uniform(0.0, 0.2),  # Obstacle sensors
            random.uniform(0.0, 0.2),
            random.uniform(0.0, 0.2)
        ]
        
        return sensors
    
    def update_robot_position(self, action: List[float]):
        """Update robot position based on action (simulate movement)."""
        if len(action) >= 2:
            # Simple movement model
            move_x = np.clip(action[0], -0.1, 0.1)
            move_y = np.clip(action[1], -0.1, 0.1)
            
            self.robot_position[0] = np.clip(self.robot_position[0] + move_x, 0.0, 1.0)
            self.robot_position[1] = np.clip(self.robot_position[1] + move_y, 0.0, 1.0)
    
    def calculate_learning_metrics(self, action: List[float], brain_state: Dict) -> Dict:
        """Calculate learning performance metrics."""
        # Distance to target (lower is better)
        dx = self.light_position[0] - self.robot_position[0]
        dy = self.light_position[1] - self.robot_position[1]
        distance_to_target = np.sqrt(dx*dx + dy*dy)
        
        # Action consistency (how stable are the actions)
        action_magnitude = np.sqrt(sum(a*a for a in action))
        
        # Field brain specific metrics
        field_energy = brain_state.get('field_energy', 0.0)
        confidence = brain_state.get('prediction_confidence', 0.0)
        evolution_cycles = brain_state.get('field_evolution_cycles', 0)
        brain_cycles = brain_state.get('brain_cycles', 0)
        
        return {
            'distance_to_target': distance_to_target,
            'action_magnitude': action_magnitude,
            'field_energy': field_energy,
            'prediction_confidence': confidence,
            'field_evolution_cycles': evolution_cycles,
            'brain_cycles': brain_cycles,
            'robot_x': self.robot_position[0],
            'robot_y': self.robot_position[1]
        }
    
    def run_learning_phase(self, phase_name: str, duration_seconds: int, cycles_target: int) -> Dict:
        """Run a learning phase and return performance metrics."""
        print(f"\n🔄 {phase_name} ({duration_seconds}s, target: {cycles_target} cycles)")
        
        start_time = time.time()
        cycle_count = 0
        phase_metrics = []
        
        while time.time() - start_time < duration_seconds and cycle_count < cycles_target:
            # Generate sensory input
            sensors = self.generate_sensory_input()
            
            # Process through brain
            action, brain_state = self.brain.process_sensory_input(sensors)
            
            # Update world state
            self.update_robot_position(action)
            
            # Calculate learning metrics
            metrics = self.calculate_learning_metrics(action, brain_state)
            metrics['cycle'] = cycle_count
            metrics['time'] = time.time() - start_time
            phase_metrics.append(metrics)
            
            cycle_count += 1
            
            # Brief pause to simulate real processing
            time.sleep(0.01)  # 10ms cycle time
        
        # Analyze phase performance
        if phase_metrics:
            avg_distance = np.mean([m['distance_to_target'] for m in phase_metrics])
            final_distance = phase_metrics[-1]['distance_to_target']
            distance_improvement = phase_metrics[0]['distance_to_target'] - final_distance
            
            final_evolution_cycles = phase_metrics[-1]['field_evolution_cycles']
            final_confidence = phase_metrics[-1]['prediction_confidence']
            final_field_energy = phase_metrics[-1]['field_energy']
            
            phase_summary = {
                'phase': phase_name,
                'cycles_completed': cycle_count,
                'avg_distance_to_target': avg_distance,
                'final_distance_to_target': final_distance,
                'distance_improvement': distance_improvement,
                'field_evolution_cycles': final_evolution_cycles,
                'prediction_confidence': final_confidence,
                'field_energy': final_field_energy,
                'learning_trajectory': phase_metrics[-10:]  # Last 10 cycles
            }
            
            print(f"   Cycles: {cycle_count}")
            print(f"   Avg distance to target: {avg_distance:.3f}")
            print(f"   Distance improvement: {distance_improvement:.3f}")
            print(f"   Field evolution cycles: {final_evolution_cycles}")
            print(f"   Prediction confidence: {final_confidence:.4f}")
            print(f"   Field energy: {final_field_energy:.3f}")
            
            return phase_summary
        
        return {'phase': phase_name, 'cycles_completed': 0}
    
    def analyze_learning_stability(self, results: List[Dict]) -> Dict:
        """Analyze learning stability across phases."""
        if len(results) < 2:
            return {'stable': False, 'reason': 'insufficient_data'}
        
        # Check for learning degradation pattern
        phase1 = results[0]
        phase2 = results[1]
        
        # Key indicators of learning degradation
        evolution_decline = phase1['field_evolution_cycles'] > 0 and phase2['field_evolution_cycles'] == phase1['field_evolution_cycles']
        confidence_decline = phase2['prediction_confidence'] < phase1['prediction_confidence'] * 0.5
        performance_decline = phase2['distance_improvement'] < phase1['distance_improvement'] * 0.5
        energy_accumulation = phase2['field_energy'] > phase1['field_energy'] * 1.5
        
        # Learning stability indicators
        evolution_progress = phase2['field_evolution_cycles'] > phase1['field_evolution_cycles']
        maintained_confidence = phase2['prediction_confidence'] >= phase1['prediction_confidence'] * 0.8
        continued_learning = phase2['distance_improvement'] >= phase1['distance_improvement'] * 0.7
        
        stability_score = sum([evolution_progress, maintained_confidence, continued_learning]) / 3.0
        degradation_score = sum([evolution_decline, confidence_decline, performance_decline, energy_accumulation]) / 4.0
        
        analysis = {
            'stable': stability_score > 0.6 and degradation_score < 0.4,
            'stability_score': stability_score,
            'degradation_score': degradation_score,
            'indicators': {
                'evolution_progress': evolution_progress,
                'maintained_confidence': maintained_confidence,
                'continued_learning': continued_learning,
                'evolution_decline': evolution_decline,
                'confidence_decline': confidence_decline,
                'performance_decline': performance_decline,
                'energy_accumulation': energy_accumulation
            },
            'phase_comparison': {
                'phase1_evolution': phase1['field_evolution_cycles'],
                'phase2_evolution': phase2['field_evolution_cycles'],
                'phase1_confidence': phase1['prediction_confidence'],
                'phase2_confidence': phase2['prediction_confidence'],
                'phase1_improvement': phase1['distance_improvement'],
                'phase2_improvement': phase2['distance_improvement']
            }
        }
        
        return analysis
    
    def run_stability_test(self) -> Dict:
        """Run the complete learning stability test."""
        print("🧠 Field Brain Learning Stability Test")
        print("=" * 50)
        print("Quick test to detect learning degradation in ~3 minutes")
        
        # Phase 1: Initial learning (like Session 0 in biological learning)
        phase1_results = self.run_learning_phase("Phase 1: Initial Learning", 60, 150)
        
        # Small break (like consolidation)
        print("\n⏸️ Brief consolidation pause (10s)...")
        time.sleep(10)
        
        # Phase 2: Continued learning (like Session 2 in biological learning)
        phase2_results = self.run_learning_phase("Phase 2: Continued Learning", 60, 150)
        
        # Phase 3: Extended learning (test if degradation continues)
        phase3_results = self.run_learning_phase("Phase 3: Extended Learning", 60, 150)
        
        # Analyze results
        print(f"\n📊 Learning Stability Analysis:")
        all_results = [phase1_results, phase2_results, phase3_results]
        
        stability_analysis = self.analyze_learning_stability(all_results)
        
        if stability_analysis['stable']:
            print(f"✅ STABLE: Field brain maintains learning capability")
            print(f"   Stability score: {stability_analysis['stability_score']:.2f}")
        else:
            print(f"❌ UNSTABLE: Field brain shows learning degradation")
            print(f"   Degradation score: {stability_analysis['degradation_score']:.2f}")
        
        print(f"\n🔍 Detailed indicators:")
        indicators = stability_analysis['indicators']
        for indicator, value in indicators.items():
            status = "✅" if value else "❌"
            print(f"   {status} {indicator}: {value}")
        
        # Cleanup
        self.brain.finalize_session()
        
        return {
            'test_results': all_results,
            'stability_analysis': stability_analysis,
            'recommendation': 'stable' if stability_analysis['stable'] else 'needs_investigation'
        }


def main():
    """Run the learning stability test."""
    test = LearningStabilityTest()
    results = test.run_stability_test()
    
    print(f"\n🎯 Test Summary:")
    print(f"   Learning stability: {'PASS' if results['recommendation'] == 'stable' else 'FAIL'}")
    print(f"   Recommendation: {results['recommendation']}")
    
    if results['recommendation'] != 'stable':
        print(f"\n⚠️ The field brain shows signs of learning degradation.")
        print(f"   This suggests issues with field evolution, confidence calculation,")
        print(f"   or energy accumulation that need further investigation.")
    
    return results


if __name__ == "__main__":
    main()