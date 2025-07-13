#!/usr/bin/env python3
"""
Behavioral Authenticity Analysis for Artificial Life Validation

This tool addresses the fundamental question: How do we judge what's "correct" 
behavior for an artificial life system? 

Key Research Questions:
1. Are we seeing authentic biological-like learning curves?
2. Is cautious behavior evidence of proper intelligence emergence?
3. What timescales are needed to validate genuine learning?
4. How do we distinguish between "working correctly" vs "needing tuning"?

Scientific Approach:
- Run extended learning sessions (30min - 24hrs)
- Track behavioral metrics over biological timescales
- Compare against known learning curve patterns
- Validate emergence claims with statistical rigor
"""

import sys
import os
import time
import json
import numpy as np
import threading
from typing import Dict, List, Any, Tuple
from collections import defaultdict, deque
from datetime import datetime

# Add brain directory to path
brain_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, brain_dir)

from demos.test_demo import PiCarXTextDemo


class BiologicalBehaviorValidator:
    """
    Validates whether robot behavior follows authentic biological learning patterns.
    """
    
    def __init__(self, output_dir: str = "tools/behavioral_analysis"):
        """Initialize the behavior validator."""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Behavioral metrics to track
        self.movement_entropy = deque(maxlen=1000)
        self.exploration_radius = deque(maxlen=1000) 
        self.action_diversity = deque(maxlen=1000)
        self.prediction_confidence = deque(maxlen=1000)
        self.spatial_coverage = deque(maxlen=1000)
        
        # Learning curve indicators
        self.learning_phases = []
        self.breakthrough_events = []
        self.plateau_periods = []
        
        # Position tracking for spatial analysis
        self.position_history = []
        self.unique_positions = set()
        
        print(f"🔬 Behavioral Authenticity Validator initialized")
        print(f"   Output directory: {output_dir}")
        print(f"   Ready to validate biological learning patterns")
    
    def run_extended_learning_session(self, duration_hours: float = 1.0, 
                                    steps_per_hour: int = 3600) -> Dict[str, Any]:
        """
        Run an extended learning session to observe authentic behavior emergence.
        
        Args:
            duration_hours: How long to run (in hours)
            steps_per_hour: Steps per hour (3600 = 1 step/second)
            
        Returns:
            Comprehensive behavioral analysis results
        """
        print(f"\n🧪 Starting Extended Learning Session")
        print(f"   Duration: {duration_hours:.1f} hours")
        print(f"   Steps per hour: {steps_per_hour}")
        print(f"   Total steps: {int(duration_hours * steps_per_hour)}")
        print(f"   This will validate biological authenticity...")
        
        start_time = time.time()
        total_steps = int(duration_hours * steps_per_hour)
        
        # Create demo instance
        demo = PiCarXTextDemo()
        
        # Data collection
        session_data = {
            'start_time': start_time,
            'duration_hours': duration_hours,
            'behavioral_metrics': [],
            'learning_events': [],
            'spatial_analysis': [],
            'phase_transitions': []
        }
        
        print(f"\n🤖 Robot learning session starting...")
        
        # Run extended session
        for step in range(total_steps):
            step_start = time.time()
            
            # Execute robot step
            try:
                cycle_result = demo.robot.control_cycle()
            except Exception as e:
                print(f"   ⚠️  Cycle error at step {step}: {e}")
                continue
            
            # Collect behavioral data
            behavior_data = self._analyze_step_behavior(demo, step, cycle_result)
            session_data['behavioral_metrics'].append(behavior_data)
            
            # Check for learning events
            learning_events = self._detect_learning_events(behavior_data, step)
            session_data['learning_events'].extend(learning_events)
            
            # Progress reporting
            if step % 600 == 0:  # Every 10 minutes at 1 step/second
                elapsed_hours = (time.time() - start_time) / 3600
                self._report_progress(step, total_steps, elapsed_hours, behavior_data)
            
            # Respect timing for biological realism
            step_duration = time.time() - step_start
            target_step_time = 3600.0 / steps_per_hour  # seconds per step
            if step_duration < target_step_time:
                time.sleep(target_step_time - step_duration)
        
        # Final analysis
        final_analysis = self._analyze_complete_session(session_data)
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"{self.output_dir}/behavioral_analysis_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump({
                'session_data': session_data,
                'final_analysis': final_analysis
            }, f, indent=2)
        
        print(f"\n📊 Session complete! Results saved to: {results_file}")
        return final_analysis
    
    def _analyze_step_behavior(self, demo, step: int, cycle_result: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze behavior for a single step."""
        
        # Get robot state
        robot_state = demo.robot.get_robot_status()
        brain_stats = demo.robot.brain.get_brain_stats()
        
        # Current position
        position = robot_state['position']
        self.position_history.append(position)
        
        # Spatial metrics
        pos_key = (round(position[0], 1), round(position[1], 1))
        self.unique_positions.add(pos_key)
        
        # Movement entropy (how unpredictable is movement?)
        if len(self.position_history) >= 5:
            recent_moves = self.position_history[-5:]
            move_vectors = []
            for i in range(1, len(recent_moves)):
                dx = recent_moves[i][0] - recent_moves[i-1][0]
                dy = recent_moves[i][1] - recent_moves[i-1][1]
                move_vectors.append((dx, dy))
            
            # Calculate entropy of movement patterns
            if move_vectors:
                unique_moves = len(set(move_vectors))
                total_moves = len(move_vectors)
                movement_entropy = unique_moves / max(1, total_moves)
                self.movement_entropy.append(movement_entropy)
        
        # Exploration radius
        if self.position_history:
            center_x = np.mean([p[0] for p in self.position_history])
            center_y = np.mean([p[1] for p in self.position_history])
            current_radius = np.sqrt((position[0] - center_x)**2 + (position[1] - center_y)**2)
            self.exploration_radius.append(current_radius)
        
        # Action diversity
        action = cycle_result.get('action_taken', [0, 0, 0, 0])
        if len(action) >= 2:
            action_magnitude = np.sqrt(action[0]**2 + action[1]**2)
            self.action_diversity.append(action_magnitude)
        
        # Prediction confidence
        confidence = cycle_result.get('prediction_confidence', 0.0)
        self.prediction_confidence.append(confidence)
        
        # Spatial coverage
        coverage = len(self.unique_positions)
        self.spatial_coverage.append(coverage)
        
        return {
            'step': step,
            'position': position,
            'movement_entropy': self.movement_entropy[-1] if self.movement_entropy else 0,
            'exploration_radius': self.exploration_radius[-1] if self.exploration_radius else 0,
            'action_diversity': self.action_diversity[-1] if self.action_diversity else 0,
            'prediction_confidence': confidence,
            'spatial_coverage': coverage,
            'brain_experiences': brain_stats.get('total_experiences', 0),
            'working_memory_size': brain_stats.get('working_memory_size', 0)
        }
    
    def _detect_learning_events(self, behavior_data: Dict[str, Any], step: int) -> List[Dict[str, Any]]:
        """Detect significant learning events in the behavioral data."""
        events = []
        
        # Confidence breakthrough (sustained high confidence)
        if (len(self.prediction_confidence) >= 20 and 
            np.mean(list(self.prediction_confidence)[-20:]) > 0.8 and
            step not in [e['step'] for e in self.breakthrough_events]):
            
            events.append({
                'type': 'confidence_breakthrough',
                'step': step,
                'confidence': behavior_data['prediction_confidence'],
                'description': 'Sustained high prediction confidence achieved'
            })
            self.breakthrough_events.append({'step': step, 'type': 'confidence'})
        
        # Spatial breakthrough (significant exploration expansion)
        if (len(self.exploration_radius) >= 50 and
            self.exploration_radius[-1] > np.mean(list(self.exploration_radius)[-50:-1]) * 1.5):
            
            events.append({
                'type': 'spatial_breakthrough',
                'step': step,
                'exploration_radius': behavior_data['exploration_radius'],
                'description': 'Significant expansion in exploration radius'
            })
        
        # Action sophistication (increased action diversity)
        if (len(self.action_diversity) >= 30 and
            np.mean(list(self.action_diversity)[-10:]) > np.mean(list(self.action_diversity)[-30:-10]) * 1.3):
            
            events.append({
                'type': 'action_sophistication',
                'step': step,
                'action_diversity': behavior_data['action_diversity'],
                'description': 'Increased sophistication in action patterns'
            })
        
        return events
    
    def _report_progress(self, step: int, total_steps: int, elapsed_hours: float, 
                        behavior_data: Dict[str, Any]):
        """Report progress during extended session."""
        progress = step / total_steps
        
        print(f"\n📊 Progress Report (Step {step:,}/{total_steps:,}, {progress:.1%})")
        print(f"   Time elapsed: {elapsed_hours:.2f} hours")
        print(f"   Position: ({behavior_data['position'][0]:.1f}, {behavior_data['position'][1]:.1f})")
        print(f"   Prediction confidence: {behavior_data['prediction_confidence']:.3f}")
        print(f"   Unique positions visited: {behavior_data['spatial_coverage']}")
        print(f"   Brain experiences: {behavior_data['brain_experiences']}")
        
        if self.movement_entropy:
            print(f"   Movement entropy: {behavior_data['movement_entropy']:.3f}")
        if self.exploration_radius:
            print(f"   Exploration radius: {behavior_data['exploration_radius']:.2f}")
    
    def _analyze_complete_session(self, session_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the complete session for biological authenticity."""
        
        metrics = session_data['behavioral_metrics']
        if not metrics:
            return {'error': 'No behavioral data collected'}
        
        # Extract time series data
        steps = [m['step'] for m in metrics]
        confidences = [m['prediction_confidence'] for m in metrics]
        positions = [m['position'] for m in metrics]
        spatial_coverage = [m['spatial_coverage'] for m in metrics]
        
        # Analyze learning curve shape
        learning_curve_analysis = self._analyze_learning_curve(confidences, steps)
        
        # Analyze spatial behavior
        spatial_analysis = self._analyze_spatial_behavior(positions)
        
        # Analyze behavioral authenticity
        authenticity_score = self._calculate_authenticity_score(session_data)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(authenticity_score, learning_curve_analysis)
        
        return {
            'session_summary': {
                'total_steps': len(metrics),
                'duration_hours': (time.time() - session_data['start_time']) / 3600,
                'final_confidence': confidences[-1] if confidences else 0,
                'spatial_coverage': spatial_coverage[-1] if spatial_coverage else 0,
                'learning_events': len(session_data['learning_events'])
            },
            'learning_curve_analysis': learning_curve_analysis,
            'spatial_analysis': spatial_analysis,
            'authenticity_score': authenticity_score,
            'recommendations': recommendations,
            'biological_validity': self._assess_biological_validity(learning_curve_analysis, spatial_analysis)
        }
    
    def _analyze_learning_curve(self, confidences: List[float], steps: List[int]) -> Dict[str, Any]:
        """Analyze the shape of the learning curve for biological authenticity."""
        
        if len(confidences) < 50:
            return {'error': 'Insufficient data for learning curve analysis'}
        
        # Divide into phases
        phase_size = len(confidences) // 4
        phases = {
            'early': confidences[:phase_size],
            'early_mid': confidences[phase_size:2*phase_size],
            'late_mid': confidences[2*phase_size:3*phase_size],
            'late': confidences[3*phase_size:]
        }
        
        phase_means = {phase: np.mean(values) for phase, values in phases.items()}
        
        # Check for biological patterns
        has_learning_progression = phase_means['late'] > phase_means['early']
        has_plateaus = any(np.std(phase) < 0.1 for phase in phases.values())
        has_breakthroughs = max(confidences) - min(confidences) > 0.3
        
        # Learning rate analysis
        if len(confidences) >= 100:
            learning_rate = (phase_means['late'] - phase_means['early']) / len(confidences)
        else:
            learning_rate = 0
        
        return {
            'has_learning_progression': has_learning_progression,
            'has_plateaus': has_plateaus,
            'has_breakthroughs': has_breakthroughs,
            'learning_rate': learning_rate,
            'phase_means': phase_means,
            'confidence_range': max(confidences) - min(confidences),
            'curve_shape': 'biological' if (has_learning_progression and has_plateaus) else 'artificial'
        }
    
    def _analyze_spatial_behavior(self, positions: List[Tuple[float, float, float]]) -> Dict[str, Any]:
        """Analyze spatial exploration patterns."""
        
        if len(positions) < 10:
            return {'error': 'Insufficient position data'}
        
        # Calculate total distance traveled
        total_distance = 0
        for i in range(1, len(positions)):
            dx = positions[i][0] - positions[i-1][0]
            dy = positions[i][1] - positions[i-1][1]
            total_distance += np.sqrt(dx**2 + dy**2)
        
        # Calculate exploration efficiency
        start_pos = positions[0]
        end_pos = positions[-1]
        direct_distance = np.sqrt((end_pos[0] - start_pos[0])**2 + (end_pos[1] - start_pos[1])**2)
        exploration_efficiency = direct_distance / max(0.1, total_distance)
        
        # Unique positions
        unique_pos = set()
        for pos in positions:
            rounded_pos = (round(pos[0], 1), round(pos[1], 1))
            unique_pos.add(rounded_pos)
        
        return {
            'total_distance': total_distance,
            'direct_distance': direct_distance,
            'exploration_efficiency': exploration_efficiency,
            'unique_positions': len(unique_pos),
            'area_coverage': len(unique_pos) / len(positions),
            'spatial_pattern': 'exploratory' if exploration_efficiency < 0.3 else 'goal_directed'
        }
    
    def _calculate_authenticity_score(self, session_data: Dict[str, Any]) -> float:
        """Calculate overall biological authenticity score (0-1)."""
        
        score_components = []
        
        # Learning progression component
        metrics = session_data['behavioral_metrics']
        if len(metrics) >= 50:
            early_conf = np.mean([m['prediction_confidence'] for m in metrics[:25]])
            late_conf = np.mean([m['prediction_confidence'] for m in metrics[-25:]])
            learning_component = min(1.0, max(0.0, (late_conf - early_conf) * 2))
            score_components.append(learning_component)
        
        # Exploration component
        if len(metrics) >= 30:
            coverage_growth = metrics[-1]['spatial_coverage'] - metrics[10]['spatial_coverage']
            exploration_component = min(1.0, coverage_growth / 50.0)
            score_components.append(exploration_component)
        
        # Learning events component
        events_component = min(1.0, len(session_data['learning_events']) / 5.0)
        score_components.append(events_component)
        
        # Time consistency component (did behavior sustain over time)
        if len(metrics) >= 100:
            consistency_component = 1.0 - (np.std([m['prediction_confidence'] for m in metrics[-50:]]) / 2.0)
            score_components.append(max(0.0, consistency_component))
        
        return np.mean(score_components) if score_components else 0.0
    
    def _generate_recommendations(self, authenticity_score: float, 
                                learning_analysis: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on the analysis."""
        
        recommendations = []
        
        if authenticity_score < 0.3:
            recommendations.append("🔴 LOW AUTHENTICITY: Consider longer learning sessions or parameter adjustment")
        elif authenticity_score < 0.6:
            recommendations.append("🟡 MODERATE AUTHENTICITY: System shows some biological patterns")
        else:
            recommendations.append("🟢 HIGH AUTHENTICITY: Behavior exhibits biological-like learning patterns")
        
        if not learning_analysis.get('has_learning_progression', False):
            recommendations.append("📈 Increase exploration drive to encourage learning progression")
        
        if not learning_analysis.get('has_plateaus', False):
            recommendations.append("🔄 Learning curve too smooth - may need more realistic challenges")
        
        if learning_analysis.get('curve_shape') == 'artificial':
            recommendations.append("🧬 Learning curve doesn't match biological patterns")
        
        return recommendations
    
    def _assess_biological_validity(self, learning_analysis: Dict[str, Any], 
                                  spatial_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall biological validity of the behavior."""
        
        validity_checks = {
            'learning_progression': learning_analysis.get('has_learning_progression', False),
            'plateau_periods': learning_analysis.get('has_plateaus', False),
            'breakthrough_events': learning_analysis.get('has_breakthroughs', False),
            'spatial_exploration': spatial_analysis.get('spatial_pattern') == 'exploratory',
            'sustained_behavior': learning_analysis.get('curve_shape') == 'biological'
        }
        
        validity_score = sum(validity_checks.values()) / len(validity_checks)
        
        return {
            'validity_checks': validity_checks,
            'validity_score': validity_score,
            'biological_assessment': (
                'HIGHLY_BIOLOGICAL' if validity_score >= 0.8 else
                'MODERATELY_BIOLOGICAL' if validity_score >= 0.6 else
                'SOMEWHAT_BIOLOGICAL' if validity_score >= 0.4 else
                'NON_BIOLOGICAL'
            )
        }


def main():
    """Run behavioral authenticity analysis."""
    
    print("🔬 Behavioral Authenticity Analysis for Artificial Life Validation")
    print("=" * 70)
    
    validator = BiologicalBehaviorValidator()
    
    # Ask user for session parameters
    print("\n🧪 Extended Learning Session Configuration:")
    print("1. Quick validation (15 minutes)")
    print("2. Standard validation (1 hour)")
    print("3. Extended validation (4 hours)")
    print("4. Long-term validation (24 hours)")
    print("5. Custom duration")
    
    choice = input("\nSelect validation type (1-5): ").strip()
    
    duration_map = {
        '1': 0.25,   # 15 minutes
        '2': 1.0,    # 1 hour
        '3': 4.0,    # 4 hours
        '4': 24.0,   # 24 hours
    }
    
    if choice in duration_map:
        duration = duration_map[choice]
    elif choice == '5':
        duration = float(input("Enter duration in hours: "))
    else:
        print("Invalid choice, using 1 hour default")
        duration = 1.0
    
    print(f"\n🚀 Starting {duration} hour behavioral validation session...")
    print("This will answer: Is the robot's cautious behavior authentic biological learning?")
    
    # Run the analysis
    results = validator.run_extended_learning_session(duration_hours=duration)
    
    # Print key findings
    print(f"\n🎯 KEY FINDINGS:")
    print(f"Authenticity Score: {results['authenticity_score']:.3f}")
    print(f"Biological Assessment: {results['biological_validity']['biological_assessment']}")
    
    print(f"\n📋 RECOMMENDATIONS:")
    for rec in results['recommendations']:
        print(f"   {rec}")


if __name__ == "__main__":
    main()