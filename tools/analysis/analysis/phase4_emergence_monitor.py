#!/usr/bin/env python3
"""
Phase 4 Emergence Monitor

Monitors and analyzes the emergence of hierarchical abstraction patterns
in the constraint-based brain system.

Key metrics tracked:
- Pattern collision rates (memory bandwidth pressure)
- Cache hierarchy utilization (natural stratification)
- Compositional reuse patterns (hierarchical emergence)
- Constraint pressure evolution over time
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


class Phase4EmergenceMonitor:
    """
    Monitor for tracking hierarchical abstraction emergence in the brain.
    
    Tracks physical constraint pressures and emergent organizational patterns
    as the system scales from small to large pattern counts.
    """
    
    def __init__(self, brain: SparseGoldilocksBrain):
        self.brain = brain
        self.monitoring_data = {
            'timestamps': [],
            'total_patterns': [],
            'collision_rates': [],
            'cache_miss_ratios': [],
            'reuse_frequencies': [],
            'natural_clusters': [],
            'stratification_depths': [],
            'constraint_pressures': [],
            'optimization_suggestions': [],
            'cache_distributions': [],
            'emergence_indicators': []
        }
        
        # Enable detailed statistics collection
        enable_investigation_mode()
        
        print("🏗️ Phase 4 Emergence Monitor initialized")
        print("   📊 Tracking hierarchical abstraction emergence")
        print("   🔍 Monitoring constraint pressure evolution")
        print("   📈 Analyzing cache hierarchy utilization")
    
    def record_emergence_snapshot(self) -> Dict[str, Any]:
        """Record current emergence state."""
        timestamp = time.time()
        
        # Get hierarchical abstraction status
        abstraction_status = self.brain.emergent_hierarchy_abstraction.get_emergence_status()
        
        # Get brain statistics
        brain_stats = self.brain.get_brain_statistics()
        
        # Record all metrics
        self.monitoring_data['timestamps'].append(timestamp)
        self.monitoring_data['total_patterns'].append(abstraction_status['total_patterns'])
        self.monitoring_data['collision_rates'].append(abstraction_status['collision_rate'])
        self.monitoring_data['cache_miss_ratios'].append(abstraction_status['cache_miss_ratio'])
        self.monitoring_data['reuse_frequencies'].append(abstraction_status['reuse_frequency'])
        self.monitoring_data['natural_clusters'].append(abstraction_status['natural_clusters'])
        self.monitoring_data['stratification_depths'].append(abstraction_status['stratification_depth'])
        self.monitoring_data['constraint_pressures'].append(abstraction_status['emergence_pressure'])
        self.monitoring_data['cache_distributions'].append(abstraction_status['cache_distribution'])
        
        # Calculate emergence indicators
        emergence_score = self._calculate_emergence_score(abstraction_status)
        self.monitoring_data['emergence_indicators'].append(emergence_score)
        
        return {
            'timestamp': timestamp,
            'abstraction_status': abstraction_status,
            'brain_stats': brain_stats,
            'emergence_score': emergence_score
        }
    
    def _calculate_emergence_score(self, status: Dict[str, Any]) -> float:
        """Calculate overall emergence score (0-1)."""
        # Weighted combination of emergence indicators
        collision_score = min(1.0, status['collision_rate'] * 5.0)  # 0.2 collision = 1.0 score
        cache_score = min(1.0, status['cache_miss_ratio'] * 2.0)    # 0.5 miss ratio = 1.0 score
        reuse_score = min(1.0, status['reuse_frequency'] * 1.0)     # 1.0 reuse = 1.0 score
        stratification_score = min(1.0, status['stratification_depth'] / 3.0)  # 3 levels = 1.0 score
        
        # Weighted average
        emergence_score = (
            0.3 * collision_score +
            0.2 * cache_score +
            0.3 * reuse_score +
            0.2 * stratification_score
        )
        
        return emergence_score
    
    def run_emergence_experiment(self, pattern_count: int = 1000, 
                               experiment_duration: int = 60) -> Dict[str, Any]:
        """
        Run controlled emergence experiment.
        
        Feeds the brain increasing numbers of patterns and tracks emergence.
        """
        print(f"🧪 Running emergence experiment:")
        print(f"   Patterns: {pattern_count}")
        print(f"   Duration: {experiment_duration}s")
        print(f"   Monitoring interval: 1s")
        
        start_time = time.time()
        pattern_counter = 0
        
        while time.time() - start_time < experiment_duration:
            # Generate and process pattern
            sensory_input = np.random.randn(self.brain.sensory_config.dim).tolist()
            
            # Process through brain
            action, brain_state = self.brain.process_sensory_input(sensory_input)
            
            pattern_counter += 1
            
            # Record snapshot every second
            if pattern_counter % 10 == 0:  # Approximately every second at 10 Hz
                snapshot = self.record_emergence_snapshot()
                
                # Print progress
                if pattern_counter % 100 == 0:
                    print(f"   📊 {pattern_counter} patterns processed")
                    print(f"      Collision rate: {snapshot['abstraction_status']['collision_rate']:.3f}")
                    print(f"      Cache miss ratio: {snapshot['abstraction_status']['cache_miss_ratio']:.3f}")
                    print(f"      Emergence score: {snapshot['emergence_score']:.3f}")
            
            # Stop if we've reached pattern limit
            if pattern_counter >= pattern_count:
                break
        
        final_snapshot = self.record_emergence_snapshot()
        
        print(f"✅ Experiment completed:")
        print(f"   Total patterns: {pattern_counter}")
        print(f"   Final emergence score: {final_snapshot['emergence_score']:.3f}")
        print(f"   Cache distribution: {final_snapshot['abstraction_status']['cache_distribution']}")
        
        return {
            'total_patterns_processed': pattern_counter,
            'experiment_duration': time.time() - start_time,
            'final_emergence_score': final_snapshot['emergence_score'],
            'final_status': final_snapshot['abstraction_status']
        }
    
    def analyze_emergence_trends(self) -> Dict[str, Any]:
        """Analyze trends in emergence data."""
        if len(self.monitoring_data['timestamps']) < 2:
            return {'error': 'Insufficient data for trend analysis'}
        
        # Calculate trends
        patterns = np.array(self.monitoring_data['total_patterns'])
        collisions = np.array(self.monitoring_data['collision_rates'])
        cache_misses = np.array(self.monitoring_data['cache_miss_ratios'])
        reuse = np.array(self.monitoring_data['reuse_frequencies'])
        emergence = np.array(self.monitoring_data['emergence_indicators'])
        
        # Linear regression for trends
        def calculate_trend(y_data):
            if len(y_data) < 2:
                return 0.0
            x_data = np.arange(len(y_data))
            slope, _ = np.polyfit(x_data, y_data, 1)
            return slope
        
        return {
            'data_points': len(self.monitoring_data['timestamps']),
            'pattern_range': [int(patterns.min()), int(patterns.max())],
            'collision_trend': calculate_trend(collisions),
            'cache_miss_trend': calculate_trend(cache_misses),
            'reuse_trend': calculate_trend(reuse),
            'emergence_trend': calculate_trend(emergence),
            'final_emergence_score': float(emergence[-1]) if len(emergence) > 0 else 0.0,
            'peak_emergence_score': float(emergence.max()) if len(emergence) > 0 else 0.0
        }
    
    def plot_emergence_evolution(self, save_path: str = None):
        """Plot emergence evolution over time."""
        if len(self.monitoring_data['timestamps']) < 2:
            print("Insufficient data for plotting")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Phase 4: Hierarchical Abstraction Emergence Evolution', fontsize=16)
        
        patterns = self.monitoring_data['total_patterns']
        
        # Plot 1: Collision rate vs patterns
        axes[0, 0].plot(patterns, self.monitoring_data['collision_rates'], 'b-', linewidth=2)
        axes[0, 0].set_title('Pattern Collision Rate')
        axes[0, 0].set_xlabel('Total Patterns')
        axes[0, 0].set_ylabel('Collision Rate')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Cache miss ratio vs patterns
        axes[0, 1].plot(patterns, self.monitoring_data['cache_miss_ratios'], 'r-', linewidth=2)
        axes[0, 1].set_title('Cache Miss Ratio')
        axes[0, 1].set_xlabel('Total Patterns')
        axes[0, 1].set_ylabel('Miss Ratio')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Reuse frequency vs patterns
        axes[1, 0].plot(patterns, self.monitoring_data['reuse_frequencies'], 'g-', linewidth=2)
        axes[1, 0].set_title('Pattern Reuse Frequency')
        axes[1, 0].set_xlabel('Total Patterns')
        axes[1, 0].set_ylabel('Reuse Frequency')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Overall emergence score vs patterns
        axes[1, 1].plot(patterns, self.monitoring_data['emergence_indicators'], 'purple', linewidth=2)
        axes[1, 1].set_title('Overall Emergence Score')
        axes[1, 1].set_xlabel('Total Patterns')
        axes[1, 1].set_ylabel('Emergence Score')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Plot saved to {save_path}")
        else:
            plt.show()
    
    def generate_emergence_report(self) -> str:
        """Generate comprehensive emergence report."""
        trends = self.analyze_emergence_trends()
        
        report = f"""
# Phase 4 Hierarchical Abstraction Emergence Report

## Experiment Overview
- Data points collected: {trends['data_points']}
- Pattern range: {trends['pattern_range'][0]} - {trends['pattern_range'][1]}
- Final emergence score: {trends['final_emergence_score']:.3f}
- Peak emergence score: {trends['peak_emergence_score']:.3f}

## Emergence Trends

### Pattern Collision Pressure
- Trend: {trends['collision_trend']:.6f} (per pattern)
- Interpretation: {"Increasing pressure" if trends['collision_trend'] > 0 else "Stable pressure"}

### Cache Miss Ratio
- Trend: {trends['cache_miss_trend']:.6f} (per pattern)
- Interpretation: {"Increasing misses" if trends['cache_miss_trend'] > 0 else "Stable cache performance"}

### Pattern Reuse Frequency
- Trend: {trends['reuse_trend']:.6f} (per pattern)
- Interpretation: {"Increasing reuse" if trends['reuse_trend'] > 0 else "Stable reuse patterns"}

### Overall Emergence
- Trend: {trends['emergence_trend']:.6f} (per pattern)
- Interpretation: {"Hierarchies emerging" if trends['emergence_trend'] > 0 else "No clear emergence"}

## Assessment

"""
        
        # Add assessment based on trends
        if trends['final_emergence_score'] > 0.5:
            report += "✅ **Strong emergence detected** - Hierarchical organization is clearly forming\n"
        elif trends['final_emergence_score'] > 0.3:
            report += "🔄 **Moderate emergence** - Early signs of hierarchical organization\n"
        elif trends['final_emergence_score'] > 0.1:
            report += "⚠️ **Weak emergence** - Some constraint pressure building\n"
        else:
            report += "❌ **No emergence** - Need more scale or different constraints\n"
        
        report += f"\n## Recommendations\n"
        
        if trends['emergence_trend'] > 0:
            report += "- Continue scaling to observe full hierarchy formation\n"
            report += "- Monitor cache stratification patterns\n"
        else:
            report += "- Consider increasing constraint pressure\n"
            report += "- May need larger scale for emergence\n"
        
        return report
    
    def save_monitoring_data(self, filepath: str):
        """Save monitoring data to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.monitoring_data, f, indent=2)
        print(f"📁 Monitoring data saved to {filepath}")


def main():
    """Main function for running emergence monitoring."""
    print("🏗️ Phase 4 Hierarchical Abstraction Emergence Monitor")
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
    monitor = Phase4EmergenceMonitor(brain)
    
    # Run emergence experiment
    experiment_results = monitor.run_emergence_experiment(
        pattern_count=1000,
        experiment_duration=30
    )
    
    print(f"\n📊 Experiment Results:")
    print(f"   Patterns processed: {experiment_results['total_patterns_processed']}")
    print(f"   Final emergence score: {experiment_results['final_emergence_score']:.3f}")
    
    # Analyze trends
    trends = monitor.analyze_emergence_trends()
    print(f"\n📈 Emergence Trends:")
    print(f"   Collision trend: {trends['collision_trend']:.6f}")
    print(f"   Cache miss trend: {trends['cache_miss_trend']:.6f}")
    print(f"   Reuse trend: {trends['reuse_trend']:.6f}")
    print(f"   Emergence trend: {trends['emergence_trend']:.6f}")
    
    # Generate report
    report = monitor.generate_emergence_report()
    print(f"\n📋 Emergence Report:")
    print(report)
    
    # Save data
    monitor.save_monitoring_data('phase4_emergence_data.json')
    
    print(f"\n✅ Phase 4 monitoring completed")


if __name__ == "__main__":
    main()