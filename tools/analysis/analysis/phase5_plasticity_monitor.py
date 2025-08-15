#!/usr/bin/env python3
"""
Phase 5 Plasticity Monitor

Monitors and analyzes the emergence of adaptive plasticity behaviors
in the multi-timescale constraint-based brain system.

Key behaviors tracked:
- Multi-timescale learning (rapid, working memory, consolidation)
- Context-dependent plasticity (activation strength effects)
- Natural forgetting (energy dissipation and interference)
- Homeostatic scaling (energy balance maintenance)
- Sleep-like consolidation (memory transfer processes)
"""

import sys
import os
import time
import json
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Tuple
from collections import defaultdict, deque

# Add path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.vector_stream.sparse_goldilocks_brain import SparseGoldilocksBrain
from src.statistics_control import enable_investigation_mode


class Phase5PlasticityMonitor:
    """
    Monitor for tracking adaptive plasticity emergence in the brain.
    
    Tracks multi-timescale learning patterns, energy dynamics, and
    emergent adaptive behaviors without explicit programming.
    """
    
    def __init__(self, brain: SparseGoldilocksBrain):
        self.brain = brain
        self.monitoring_data = {
            'timestamps': [],
            'total_patterns': [],
            'working_memory_patterns': [],
            'consolidated_patterns': [],
            'total_system_energy': [],
            'homeostatic_pressure': [],
            'emergent_learning_rates': [],
            'emergent_forgetting_rates': [],
            'rapid_learning_rates': [],
            'consolidation_rates': [],
            'context_sensitivity': [],
            'multi_timescale_balance': [],
            'interference_events': [],
            'energy_redistribution_events': [],
            'activation_strengths': [],
            'pattern_energies': []
        }
        
        # Enable detailed statistics collection
        enable_investigation_mode()
        
        print("🧠 Phase 5 Plasticity Monitor initialized")
        print("   📊 Tracking multi-timescale learning emergence")
        print("   ⚡ Monitoring energy dynamics and homeostasis")
        print("   🔄 Analyzing adaptive plasticity behaviors")
    
    def record_plasticity_snapshot(self) -> Dict[str, Any]:
        """Record current plasticity state."""
        timestamp = time.time()
        
        # Get plasticity status
        plasticity_status = self.brain.emergent_adaptive_plasticity.get_plasticity_status()
        
        # Get brain statistics
        brain_stats = self.brain.get_brain_statistics()
        
        # Record all metrics
        self.monitoring_data['timestamps'].append(timestamp)
        self.monitoring_data['total_patterns'].append(plasticity_status['total_patterns'])
        self.monitoring_data['working_memory_patterns'].append(plasticity_status['working_memory_patterns'])
        self.monitoring_data['consolidated_patterns'].append(plasticity_status['consolidated_patterns'])
        self.monitoring_data['total_system_energy'].append(plasticity_status['total_system_energy'])
        self.monitoring_data['homeostatic_pressure'].append(plasticity_status['homeostatic_pressure'])
        self.monitoring_data['rapid_learning_rates'].append(plasticity_status['rapid_learning_rate'])
        self.monitoring_data['consolidation_rates'].append(plasticity_status['consolidation_rate'])
        self.monitoring_data['context_sensitivity'].append(plasticity_status['context_sensitivity'])
        self.monitoring_data['multi_timescale_balance'].append(plasticity_status['multi_timescale_balance'])
        self.monitoring_data['interference_events'].append(plasticity_status['interference_events'])
        self.monitoring_data['energy_redistribution_events'].append(plasticity_status['energy_redistribution_events'])
        
        # Extract pattern energies for detailed analysis
        pattern_energies = {}
        for pattern_id, energy_state in self.brain.emergent_adaptive_plasticity.pattern_energies.items():
            pattern_energies[pattern_id] = {
                'immediate': energy_state.immediate_energy,
                'working': energy_state.working_energy,
                'consolidated': energy_state.consolidated_energy,
                'total': energy_state.total_energy(),
                'activation_count': energy_state.activation_count
            }
        
        self.monitoring_data['pattern_energies'].append(pattern_energies)
        
        return {
            'timestamp': timestamp,
            'plasticity_status': plasticity_status,
            'brain_stats': brain_stats,
            'pattern_energies': pattern_energies
        }
    
    def run_multi_timescale_learning_experiment(self, patterns_per_phase: int = 50) -> Dict[str, Any]:
        """
        Run experiment to demonstrate multi-timescale learning.
        
        Tests rapid learning, working memory formation, and consolidation.
        """
        print(f"🧪 Running multi-timescale learning experiment:")
        print(f"   Phase 1: Rapid learning ({patterns_per_phase} patterns)")
        print(f"   Phase 2: Working memory consolidation ({patterns_per_phase} patterns)")
        print(f"   Phase 3: Long-term consolidation ({patterns_per_phase} patterns)")
        
        results = {
            'phases': [],
            'learning_progression': [],
            'consolidation_progression': []
        }
        
        # Phase 1: Rapid learning with high activation
        print("\n📈 Phase 1: Rapid learning (high activation)")
        for i in range(patterns_per_phase):
            # Generate high-activation pattern
            sensory_input = np.random.randn(self.brain.sensory_config.dim) * 2.0  # High intensity
            action, brain_state = self.brain.process_sensory_input(sensory_input.tolist())
            
            if i % 10 == 0:
                snapshot = self.record_plasticity_snapshot()
                results['phases'].append(('rapid', i, snapshot))
                print(f"   Pattern {i}: Learning rate={brain_state.get('emergent_learning_rate', 0):.3f}, "
                      f"Working memory={snapshot['plasticity_status']['working_memory_patterns']}")
        
        # Phase 2: Working memory consolidation with medium activation
        print("\n🧠 Phase 2: Working memory consolidation (medium activation)")
        for i in range(patterns_per_phase):
            # Generate medium-activation pattern
            sensory_input = np.random.randn(self.brain.sensory_config.dim) * 1.0  # Medium intensity
            action, brain_state = self.brain.process_sensory_input(sensory_input.tolist())
            
            if i % 10 == 0:
                snapshot = self.record_plasticity_snapshot()
                results['phases'].append(('working', i, snapshot))
                print(f"   Pattern {i}: Learning rate={brain_state.get('emergent_learning_rate', 0):.3f}, "
                      f"Consolidated={snapshot['plasticity_status']['consolidated_patterns']}")
        
        # Phase 3: Long-term consolidation with low activation
        print("\n🏛️ Phase 3: Long-term consolidation (low activation)")
        for i in range(patterns_per_phase):
            # Generate low-activation pattern
            sensory_input = np.random.randn(self.brain.sensory_config.dim) * 0.5  # Low intensity
            action, brain_state = self.brain.process_sensory_input(sensory_input.tolist())
            
            if i % 10 == 0:
                snapshot = self.record_plasticity_snapshot()
                results['phases'].append(('consolidation', i, snapshot))
                print(f"   Pattern {i}: Learning rate={brain_state.get('emergent_learning_rate', 0):.3f}, "
                      f"System energy={snapshot['plasticity_status']['total_system_energy']:.3f}")
        
        final_snapshot = self.record_plasticity_snapshot()
        results['final_state'] = final_snapshot
        
        print(f"\n✅ Multi-timescale experiment completed:")
        print(f"   Total patterns: {final_snapshot['plasticity_status']['total_patterns']}")
        print(f"   Working memory: {final_snapshot['plasticity_status']['working_memory_patterns']}")
        print(f"   Consolidated: {final_snapshot['plasticity_status']['consolidated_patterns']}")
        print(f"   System energy: {final_snapshot['plasticity_status']['total_system_energy']:.3f}")
        
        return results
    
    def run_context_dependent_plasticity_experiment(self, pattern_count: int = 100) -> Dict[str, Any]:
        """
        Run experiment to demonstrate context-dependent plasticity.
        
        Tests how activation strength affects learning and forgetting.
        """
        print(f"🧪 Running context-dependent plasticity experiment:")
        print(f"   Testing activation strength effects on learning")
        print(f"   Pattern count: {pattern_count}")
        
        results = {
            'high_activation_results': [],
            'medium_activation_results': [],
            'low_activation_results': []
        }
        
        # Test different activation strengths
        activation_levels = [
            ('high', 2.0, results['high_activation_results']),
            ('medium', 1.0, results['medium_activation_results']),
            ('low', 0.3, results['low_activation_results'])
        ]
        
        for level_name, intensity, result_list in activation_levels:
            print(f"\n🎯 Testing {level_name} activation (intensity={intensity})")
            
            for i in range(pattern_count // 3):
                # Generate pattern with specific activation level
                sensory_input = np.random.randn(self.brain.sensory_config.dim) * intensity
                action, brain_state = self.brain.process_sensory_input(sensory_input.tolist())
                
                if i % 10 == 0:
                    snapshot = self.record_plasticity_snapshot()
                    result_list.append({
                        'pattern_index': i,
                        'activation_level': level_name,
                        'learning_rate': brain_state.get('emergent_learning_rate', 0),
                        'forgetting_rate': brain_state.get('emergent_forgetting_rate', 0),
                        'system_energy': brain_state.get('total_system_energy', 0),
                        'snapshot': snapshot
                    })
        
        # Analyze context effects
        context_analysis = self._analyze_context_effects(results)
        results['context_analysis'] = context_analysis
        
        print(f"\n✅ Context-dependent plasticity experiment completed:")
        print(f"   High activation learning rate: {context_analysis['high_avg_learning']:.3f}")
        print(f"   Medium activation learning rate: {context_analysis['medium_avg_learning']:.3f}")
        print(f"   Low activation learning rate: {context_analysis['low_avg_learning']:.3f}")
        
        return results
    
    def run_sleep_consolidation_experiment(self, wake_duration: float = 30.0, 
                                        sleep_duration: float = 10.0) -> Dict[str, Any]:
        """
        Run experiment to demonstrate sleep-like consolidation.
        
        Tests memory transfer from working to consolidated storage.
        """
        print(f"🧪 Running sleep consolidation experiment:")
        print(f"   Wake phase: {wake_duration}s")
        print(f"   Sleep phase: {sleep_duration}s")
        
        results = {
            'wake_snapshots': [],
            'sleep_snapshots': [],
            'consolidation_transfer': {}
        }
        
        # Wake phase: Active learning
        print("\n☀️ Wake phase: Active learning")
        wake_start = time.time()
        pattern_count = 0
        
        while time.time() - wake_start < wake_duration:
            # Generate learning experiences
            sensory_input = np.random.randn(self.brain.sensory_config.dim) * 1.5
            action, brain_state = self.brain.process_sensory_input(sensory_input.tolist())
            pattern_count += 1
            
            if pattern_count % 20 == 0:
                snapshot = self.record_plasticity_snapshot()
                results['wake_snapshots'].append(snapshot)
                print(f"   Wake pattern {pattern_count}: Working memory={snapshot['plasticity_status']['working_memory_patterns']}, "
                      f"Consolidated={snapshot['plasticity_status']['consolidated_patterns']}")
        
        # Record pre-sleep state
        pre_sleep_snapshot = self.record_plasticity_snapshot()
        results['pre_sleep_state'] = pre_sleep_snapshot
        
        # Sleep phase: Consolidation
        print(f"\n🌙 Sleep phase: Consolidation ({sleep_duration}s)")
        self.brain.emergent_adaptive_plasticity.simulate_sleep_consolidation(sleep_duration)
        
        # Record post-sleep state
        post_sleep_snapshot = self.record_plasticity_snapshot()
        results['post_sleep_state'] = post_sleep_snapshot
        
        # Analyze consolidation transfer
        consolidation_transfer = {
            'working_memory_before': pre_sleep_snapshot['plasticity_status']['working_memory_patterns'],
            'working_memory_after': post_sleep_snapshot['plasticity_status']['working_memory_patterns'],
            'consolidated_before': pre_sleep_snapshot['plasticity_status']['consolidated_patterns'],
            'consolidated_after': post_sleep_snapshot['plasticity_status']['consolidated_patterns'],
            'transfer_efficiency': (post_sleep_snapshot['plasticity_status']['consolidated_patterns'] - 
                                   pre_sleep_snapshot['plasticity_status']['consolidated_patterns']) / 
                                   max(1, pre_sleep_snapshot['plasticity_status']['working_memory_patterns'])
        }
        
        results['consolidation_transfer'] = consolidation_transfer
        
        print(f"\n✅ Sleep consolidation experiment completed:")
        print(f"   Working memory transfer: {consolidation_transfer['working_memory_before']} → {consolidation_transfer['working_memory_after']}")
        print(f"   Consolidated gain: {consolidation_transfer['consolidated_before']} → {consolidation_transfer['consolidated_after']}")
        print(f"   Transfer efficiency: {consolidation_transfer['transfer_efficiency']:.3f}")
        
        return results
    
    def _analyze_context_effects(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze context-dependent plasticity effects."""
        analysis = {}
        
        # Calculate average learning rates by activation level
        for level in ['high', 'medium', 'low']:
            key = f'{level}_activation_results'
            if key in results and results[key]:
                learning_rates = [r['learning_rate'] for r in results[key]]
                forgetting_rates = [r['forgetting_rate'] for r in results[key]]
                
                analysis[f'{level}_avg_learning'] = np.mean(learning_rates)
                analysis[f'{level}_avg_forgetting'] = np.mean(forgetting_rates)
                analysis[f'{level}_learning_std'] = np.std(learning_rates)
                analysis[f'{level}_forgetting_std'] = np.std(forgetting_rates)
        
        # Calculate context sensitivity
        high_lr = analysis.get('high_avg_learning', 0)
        medium_lr = analysis.get('medium_avg_learning', 0)
        low_lr = analysis.get('low_avg_learning', 0)
        
        analysis['context_sensitivity'] = (high_lr - low_lr) / max(medium_lr, 1e-6)
        
        return analysis
    
    def analyze_plasticity_trends(self) -> Dict[str, Any]:
        """Analyze trends in plasticity data."""
        if len(self.monitoring_data['timestamps']) < 2:
            return {'error': 'Insufficient data for trend analysis'}
        
        # Calculate trends
        working_memory = np.array(self.monitoring_data['working_memory_patterns'])
        consolidated = np.array(self.monitoring_data['consolidated_patterns'])
        system_energy = np.array(self.monitoring_data['total_system_energy'])
        homeostatic_pressure = np.array(self.monitoring_data['homeostatic_pressure'])
        
        # Linear regression for trends
        def calculate_trend(y_data):
            if len(y_data) < 2:
                return 0.0
            x_data = np.arange(len(y_data))
            slope, _ = np.polyfit(x_data, y_data, 1)
            return slope
        
        return {
            'data_points': len(self.monitoring_data['timestamps']),
            'working_memory_trend': calculate_trend(working_memory),
            'consolidation_trend': calculate_trend(consolidated),
            'energy_trend': calculate_trend(system_energy),
            'homeostatic_trend': calculate_trend(homeostatic_pressure),
            'final_working_memory': int(working_memory[-1]) if len(working_memory) > 0 else 0,
            'final_consolidated': int(consolidated[-1]) if len(consolidated) > 0 else 0,
            'final_energy': float(system_energy[-1]) if len(system_energy) > 0 else 0.0,
            'final_homeostatic_pressure': float(homeostatic_pressure[-1]) if len(homeostatic_pressure) > 0 else 0.0
        }
    
    def plot_plasticity_evolution(self, save_path: str = None):
        """Plot plasticity evolution over time."""
        if len(self.monitoring_data['timestamps']) < 2:
            print("Insufficient data for plotting")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Phase 5: Adaptive Plasticity Evolution', fontsize=16)
        
        timestamps = np.array(self.monitoring_data['timestamps'])
        time_relative = timestamps - timestamps[0]  # Relative time
        
        # Plot 1: Memory system evolution
        axes[0, 0].plot(time_relative, self.monitoring_data['working_memory_patterns'], 'b-', 
                       linewidth=2, label='Working Memory')
        axes[0, 0].plot(time_relative, self.monitoring_data['consolidated_patterns'], 'r-', 
                       linewidth=2, label='Consolidated')
        axes[0, 0].set_title('Memory System Evolution')
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_ylabel('Pattern Count')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Energy dynamics
        axes[0, 1].plot(time_relative, self.monitoring_data['total_system_energy'], 'g-', linewidth=2)
        axes[0, 1].set_title('System Energy Dynamics')
        axes[0, 1].set_xlabel('Time (s)')
        axes[0, 1].set_ylabel('Total Energy')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Learning rates
        if len(self.monitoring_data['rapid_learning_rates']) > 0:
            axes[1, 0].plot(time_relative, self.monitoring_data['rapid_learning_rates'], 'orange', 
                           linewidth=2, label='Rapid Learning')
            axes[1, 0].plot(time_relative, self.monitoring_data['consolidation_rates'], 'purple', 
                           linewidth=2, label='Consolidation')
            axes[1, 0].set_title('Learning Rate Evolution')
            axes[1, 0].set_xlabel('Time (s)')
            axes[1, 0].set_ylabel('Learning Rate')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Homeostatic pressure
        axes[1, 1].plot(time_relative, self.monitoring_data['homeostatic_pressure'], 'red', linewidth=2)
        axes[1, 1].set_title('Homeostatic Pressure')
        axes[1, 1].set_xlabel('Time (s)')
        axes[1, 1].set_ylabel('Pressure')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Plot saved to {save_path}")
        else:
            plt.show()
    
    def generate_plasticity_report(self) -> str:
        """Generate comprehensive plasticity report."""
        trends = self.analyze_plasticity_trends()
        
        report = f"""
# Phase 5 Adaptive Plasticity Report

## Experiment Overview
- Data points collected: {trends['data_points']}
- Final working memory patterns: {trends['final_working_memory']}
- Final consolidated patterns: {trends['final_consolidated']}
- Final system energy: {trends['final_energy']:.3f}
- Final homeostatic pressure: {trends['final_homeostatic_pressure']:.3f}

## Plasticity Trends

### Working Memory Evolution
- Trend: {trends['working_memory_trend']:.3f} patterns/step
- Interpretation: {"Increasing capacity" if trends['working_memory_trend'] > 0 else "Stable capacity"}

### Consolidation Evolution
- Trend: {trends['consolidation_trend']:.3f} patterns/step
- Interpretation: {"Increasing consolidation" if trends['consolidation_trend'] > 0 else "Stable consolidation"}

### Energy Dynamics
- Trend: {trends['energy_trend']:.6f} energy/step
- Interpretation: {"Increasing energy" if trends['energy_trend'] > 0 else "Stable energy"}

### Homeostatic Regulation
- Trend: {trends['homeostatic_trend']:.6f} pressure/step
- Interpretation: {"Increasing pressure" if trends['homeostatic_trend'] > 0 else "Stable regulation"}

## Assessment

"""
        
        # Add assessment based on trends
        if trends['final_consolidated'] > 5:
            report += "✅ **Strong consolidation** - Long-term memory formation working\n"
        elif trends['final_consolidated'] > 2:
            report += "🔄 **Moderate consolidation** - Some long-term memory formation\n"
        else:
            report += "⚠️ **Weak consolidation** - Limited long-term memory formation\n"
        
        if abs(trends['final_homeostatic_pressure']) < 0.1:
            report += "✅ **Good homeostasis** - Energy regulation working well\n"
        elif abs(trends['final_homeostatic_pressure']) < 0.5:
            report += "🔄 **Moderate homeostasis** - Some energy regulation\n"
        else:
            report += "⚠️ **Poor homeostasis** - Energy regulation struggling\n"
        
        report += f"\n## Multi-Timescale Learning\n"
        report += f"- Working memory patterns: {trends['final_working_memory']}\n"
        report += f"- Consolidated patterns: {trends['final_consolidated']}\n"
        report += f"- System energy balance: {trends['final_energy']:.3f}\n"
        
        return report
    
    def save_monitoring_data(self, filepath: str):
        """Save monitoring data to JSON file."""
        # Convert numpy arrays to lists for JSON serialization
        json_data = {}
        for key, value in self.monitoring_data.items():
            if isinstance(value, list):
                json_data[key] = [v.tolist() if isinstance(v, np.ndarray) else v for v in value]
            else:
                json_data[key] = value.tolist() if isinstance(value, np.ndarray) else value
        
        with open(filepath, 'w') as f:
            json.dump(json_data, f, indent=2)
        print(f"📁 Monitoring data saved to {filepath}")


def main():
    """Main function for running plasticity monitoring."""
    print("🧠 Phase 5 Adaptive Plasticity Monitor")
    print("=" * 60)
    
    # Create brain instance
    brain = SparseGoldilocksBrain(
        sensory_dim=16,
        motor_dim=8,
        temporal_dim=4,
        max_patterns=100_000,
        quiet_mode=True
    )
    
    # Create monitor
    monitor = Phase5PlasticityMonitor(brain)
    
    # Run multi-timescale learning experiment
    print("\n🧪 Running Multi-Timescale Learning Experiment")
    multiscale_results = monitor.run_multi_timescale_learning_experiment(30)
    
    # Run context-dependent plasticity experiment
    print("\n🧪 Running Context-Dependent Plasticity Experiment")
    context_results = monitor.run_context_dependent_plasticity_experiment(60)
    
    # Run sleep consolidation experiment
    print("\n🧪 Running Sleep Consolidation Experiment")
    sleep_results = monitor.run_sleep_consolidation_experiment(20, 5)
    
    # Analyze trends
    trends = monitor.analyze_plasticity_trends()
    print(f"\n📈 Plasticity Trends:")
    print(f"   Working memory trend: {trends['working_memory_trend']:.3f}")
    print(f"   Consolidation trend: {trends['consolidation_trend']:.3f}")
    print(f"   Energy trend: {trends['energy_trend']:.6f}")
    print(f"   Homeostatic trend: {trends['homeostatic_trend']:.6f}")
    
    # Generate report
    report = monitor.generate_plasticity_report()
    print(f"\n📋 Plasticity Report:")
    print(report)
    
    # Save data
    monitor.save_monitoring_data('phase5_plasticity_data.json')
    
    print(f"\n✅ Phase 5 monitoring completed")


if __name__ == "__main__":
    main()