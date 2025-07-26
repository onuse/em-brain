#!/usr/bin/env python3
"""
Long-term Learning Test
Test the brain's ability to learn and adapt over extended periods
"""

import sys
import os
import numpy as np
import time
from typing import Dict, List

sys.path.insert(0, os.path.dirname(__file__))
from src.brain_factory import BrainFactory


def create_learning_environment(phase: int, cycle: int) -> List[float]:
    """Create different learning environments that change over time"""
    
    if phase == 1:  # Early exploration phase (cycles 0-100)
        # Random, noisy environment to encourage exploration
        base_pattern = [0.5] * 16
        noise = [np.random.uniform(-0.3, 0.3) for _ in range(16)]
        return [max(0.0, min(1.0, base_pattern[i] + noise[i])) for i in range(16)]
    
    elif phase == 2:  # Pattern learning phase (cycles 100-300) 
        # Introduce learnable patterns
        pattern_type = (cycle // 20) % 3
        if pattern_type == 0:
            return [0.8, 0.2, 0.6, 0.4, 0.7, 0.3, 0.9, 0.1] * 2
        elif pattern_type == 1:
            return [0.1, 0.9, 0.3, 0.7, 0.2, 0.8, 0.4, 0.6] * 2
        else:
            return [0.5, 0.5, 0.8, 0.2, 0.5, 0.5, 0.2, 0.8] * 2
    
    elif phase == 3:  # Adaptation phase (cycles 300-500)
        # Gradual environmental change
        shift_amount = (cycle - 300) * 0.001
        base_pattern = [0.6, 0.4, 0.7, 0.3, 0.8, 0.2, 0.5, 0.5] * 2
        return [max(0.0, min(1.0, val + shift_amount * (i % 2 * 2 - 1))) 
                for i, val in enumerate(base_pattern)]
    
    else:  # Mastery phase (cycles 500+)
        # Complex, structured environment
        t = cycle * 0.02
        complex_pattern = [
            0.5 + 0.3 * np.sin(t),
            0.4 + 0.2 * np.cos(t * 1.2),
            0.6 + 0.25 * np.sin(t * 0.8),
            0.3 + 0.35 * np.cos(t * 1.5),
            0.7 + 0.15 * np.sin(t * 2.0),
            0.45 + 0.3 * np.cos(t * 0.6),
            0.55 + 0.2 * np.sin(t * 1.8),
            0.65 + 0.25 * np.cos(t * 0.9)
        ] * 2
        return [max(0.0, min(1.0, val)) for val in complex_pattern]


def analyze_learning_phase(confidences: List[float], actions: List[List[float]], 
                          phase_name: str, start_idx: int, end_idx: int) -> Dict[str, float]:
    """Analyze learning metrics for a specific phase"""
    
    phase_confidences = confidences[start_idx:end_idx]
    phase_actions = actions[start_idx:end_idx]
    
    if len(phase_confidences) < 10:
        return {'insufficient_data': True}
    
    # Confidence progression
    early_conf = np.mean(phase_confidences[:len(phase_confidences)//3])
    late_conf = np.mean(phase_confidences[-len(phase_confidences)//3:])
    confidence_improvement = late_conf - early_conf
    
    # Action stability (decreasing variance indicates learning)
    action_variances = [np.var(action) for action in phase_actions]
    early_variance = np.mean(action_variances[:len(action_variances)//3])
    late_variance = np.mean(action_variances[-len(action_variances)//3:])
    stability_improvement = early_variance - late_variance  # Higher = more stable
    
    # Learning rate (confidence change rate)
    if len(phase_confidences) > 1:
        confidence_changes = [abs(phase_confidences[i+1] - phase_confidences[i]) 
                            for i in range(len(phase_confidences)-1)]
        learning_activity = np.mean(confidence_changes)
    else:
        learning_activity = 0.0
    
    return {
        'phase_name': phase_name,
        'early_confidence': early_conf,
        'late_confidence': late_conf,
        'confidence_improvement': confidence_improvement,
        'early_action_variance': early_variance,
        'late_action_variance': late_variance,
        'stability_improvement': stability_improvement,
        'learning_activity': learning_activity,
        'phase_learning_score': (confidence_improvement + stability_improvement * 0.5) / 1.5
    }


def run_long_term_learning_test(total_cycles: int = 600) -> Dict[str, any]:
    """Run extended learning test across multiple phases"""
    
    print("🕐 Long-term Learning Test")
    print("Testing brain learning over extended periods with changing environments")
    print("=" * 70)
    
    # Clear memory for fresh test
    if os.path.exists('robot_memory'):
        import shutil
        shutil.rmtree('robot_memory')
        print("🗑️ Cleared robot memory for fresh long-term learning")
    
    results = {
        'total_cycles': total_cycles,
        'learning_phases': [],
        'overall_performance': {},
        'confidence_progression': [],
        'action_progression': [],
        'brain_evolution': []
    }
    
    # Create brain
    brain = BrainFactory(quiet_mode=True)
    print(f"🧠 Brain initialized: {brain.brain_type} architecture")
    
    all_confidences = []
    all_actions = []
    
    print(f"\n🔄 Running {total_cycles} long-term learning cycles...")
    print("   Phase 1 (0-100): Exploration")
    print("   Phase 2 (100-300): Pattern Learning") 
    print("   Phase 3 (300-500): Adaptation")
    print("   Phase 4 (500+): Mastery")
    
    for cycle in range(total_cycles):
        # Determine current learning phase
        if cycle < 100:
            phase = 1
        elif cycle < 300:
            phase = 2
        elif cycle < 500:
            phase = 3
        else:
            phase = 4
        
        # Generate environment for current phase
        sensory_data = create_learning_environment(phase, cycle)
        
        # Process through brain
        action, brain_state = brain.process_sensory_input(sensory_data)
        
        # Track metrics
        confidence = brain_state.get('prediction_confidence', 0.0)
        all_confidences.append(confidence)
        all_actions.append(action)
        
        # Store detailed data every 50 cycles
        if cycle % 50 == 0:
            results['confidence_progression'].append(confidence)
            results['action_progression'].append(action.copy())
            
            brain_stats = brain.get_brain_stats()
            results['brain_evolution'].append({
                'cycle': cycle,
                'phase': phase,
                'confidence': confidence,
                'field_energy': brain_stats.get('field_brain_adapter', {}).get('field_brain', {}).get('field_energy', 0.0),
                'brain_cycles': brain_stats.get('field_brain_adapter', {}).get('field_brain', {}).get('brain_cycles', 0)
            })
        
        # Progress reporting
        if cycle % 100 == 0 and cycle > 0:
            recent_conf = np.mean(all_confidences[-20:])
            print(f"   Cycle {cycle:3d} (Phase {phase}): Confidence={recent_conf:.3f}")
    
    print(f"\n📊 Analyzing learning across phases...")
    
    # Analyze each learning phase
    phase_analyses = [
        analyze_learning_phase(all_confidences, all_actions, "Exploration", 0, 100),
        analyze_learning_phase(all_confidences, all_actions, "Pattern Learning", 100, 300),
        analyze_learning_phase(all_confidences, all_actions, "Adaptation", 300, 500),
    ]
    
    if total_cycles > 500:
        phase_analyses.append(
            analyze_learning_phase(all_confidences, all_actions, "Mastery", 500, total_cycles)
        )
    
    results['learning_phases'] = phase_analyses
    
    # Overall analysis
    overall_confidence_improvement = all_confidences[-1] - all_confidences[0] if all_confidences else 0
    
    # Long-term stability (confidence variance in final 100 cycles)
    final_confidences = all_confidences[-100:] if len(all_confidences) >= 100 else all_confidences
    long_term_stability = 1.0 - np.var(final_confidences)  # Higher = more stable
    
    # Learning efficiency (improvement per cycle)
    learning_efficiency = overall_confidence_improvement / total_cycles if total_cycles > 0 else 0
    
    results['overall_performance'] = {
        'overall_confidence_improvement': overall_confidence_improvement,
        'long_term_stability': long_term_stability,
        'learning_efficiency': learning_efficiency,
        'final_confidence': all_confidences[-1] if all_confidences else 0,
        'peak_confidence': max(all_confidences) if all_confidences else 0
    }
    
    print(f"\n📈 Long-term Learning Analysis:")
    for phase_analysis in phase_analyses:
        if 'insufficient_data' not in phase_analysis:
            print(f"   {phase_analysis['phase_name']}: "
                  f"Confidence Δ{phase_analysis['confidence_improvement']:+.3f}, "
                  f"Learning Score: {phase_analysis['phase_learning_score']:.3f}")
    
    print(f"   Overall: Confidence Δ{overall_confidence_improvement:+.3f}, "
          f"Efficiency: {learning_efficiency:.4f}/cycle")
    print(f"   Long-term Stability: {long_term_stability:.3f}")
    
    # Overall long-term learning score
    phase_scores = [p.get('phase_learning_score', 0) for p in phase_analyses if 'phase_learning_score' in p]
    avg_phase_score = np.mean(phase_scores) if phase_scores else 0
    
    long_term_score = (avg_phase_score + overall_confidence_improvement + long_term_stability) / 3.0
    results['long_term_learning_score'] = long_term_score
    
    print(f"   Long-term Learning Score: {long_term_score:.3f}")
    
    if long_term_score > 0.3:
        print("   🎉 EXCELLENT long-term learning!")
    elif long_term_score > 0.2:
        print("   ✅ GOOD long-term learning!")
    elif long_term_score > 0.1:
        print("   ⚠️  MODERATE long-term learning")
    else:
        print("   ❌ LIMITED long-term learning")
    
    brain.finalize_session()
    return results


if __name__ == "__main__":
    results = run_long_term_learning_test(600)
    
    print(f"\n🎯 Long-term Learning Test Complete!")
    score = results.get('long_term_learning_score', 0)
    
    if score > 0.4:
        print(f"🏆 OUTSTANDING long-term learning: {score:.3f}")
    elif score > 0.3:
        print(f"🎉 EXCELLENT long-term learning: {score:.3f}")
    elif score > 0.2:
        print(f"✅ GOOD long-term learning: {score:.3f}")
    else:
        print(f"⚠️  Needs improvement: {score:.3f}")
    
    # Summary of key metrics
    overall = results['overall_performance']
    print(f"\n📊 Key Metrics:")
    print(f"   Final Confidence: {overall['final_confidence']:.3f}")
    print(f"   Peak Confidence: {overall['peak_confidence']:.3f}")
    print(f"   Learning Efficiency: {overall['learning_efficiency']:.4f} per cycle")
    print(f"   Long-term Stability: {overall['long_term_stability']:.3f}")