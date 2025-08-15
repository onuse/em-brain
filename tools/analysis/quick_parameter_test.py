#!/usr/bin/env python3
"""
Quick Parameter Test Tool

Provides rapid feedback on parameter changes using short behavioral tests.
Perfect for iterative development and quick experiments.
"""

import sys
import os
from pathlib import Path
import json
import time
import numpy as np
from typing import Dict, Any, Optional

# Add brain root to path
brain_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(brain_root))
sys.path.insert(0, str(brain_root / 'server'))

from src.communication import MinimalBrainClient
from validation.embodied_learning.environments.sensory_motor_world import SensoryMotorWorld


class QuickParameterTest:
    """Quick test harness for parameter tuning"""
    
    def __init__(self, custom_settings: Optional[Dict] = None):
        self.custom_settings = custom_settings or {}
        self.results = []
        
    def apply_custom_settings(self):
        """Apply custom settings to brain configuration"""
        settings_path = Path("server/settings.json")
        
        # Backup original settings
        if settings_path.exists():
            with open(settings_path, 'r') as f:
                self.original_settings = json.load(f)
        else:
            self.original_settings = {}
        
        # Apply custom settings
        if self.custom_settings:
            new_settings = self.original_settings.copy()
            new_settings.update(self.custom_settings)
            
            with open(settings_path, 'w') as f:
                json.dump(new_settings, f, indent=2)
            
            print(f"📝 Applied custom settings: {self.custom_settings}")
    
    def restore_settings(self):
        """Restore original settings"""
        if hasattr(self, 'original_settings'):
            settings_path = Path("server/settings.json")
            with open(settings_path, 'w') as f:
                json.dump(self.original_settings, f, indent=2)
            print("🔄 Restored original settings")
    
    def run_quick_test(self, duration_seconds: int = 60) -> Dict[str, float]:
        """Run a quick behavioral test"""
        print(f"\n🚀 Running {duration_seconds}s quick test...")
        
        # Initialize environment
        environment = SensoryMotorWorld(
            world_size=10.0,
            num_light_sources=2,
            num_obstacles=3
        )
        
        # Connect to brain
        client = MinimalBrainClient()
        if not client.connect():
            print("❌ Failed to connect to brain server")
            return {}
        
        print("✅ Connected to brain server")
        
        # Metrics tracking
        start_time = time.time()
        actions_taken = []
        light_distances = []
        collision_count = 0
        prediction_errors = []
        cycle_times = []
        
        # Run test cycles
        while (time.time() - start_time) < duration_seconds:
            cycle_start = time.time()
            
            # Get sensory input
            sensory_input = environment.get_sensory_input()
            
            # Get brain prediction
            prediction = client.get_action(sensory_input, timeout=1.0)
            
            if prediction is None:
                print("⚠️  No response from brain")
                continue
            
            # Execute action
            result = environment.execute_action(prediction)
            
            # Track metrics
            action_name = ['FORWARD', 'LEFT', 'RIGHT', 'STOP'][result['action_executed']]
            actions_taken.append(action_name)
            
            metrics = result['metrics']
            light_distances.append(metrics['min_light_distance'])
            
            # Track collisions
            if action_name == 'FORWARD' and not result['success']:
                collision_count += 1
            
            # Calculate prediction error
            pred_error = np.mean(np.abs(np.array(prediction[:4]) - np.array(sensory_input[:4])))
            prediction_errors.append(pred_error)
            
            # Track cycle time
            cycle_time = (time.time() - cycle_start) * 1000  # ms
            cycle_times.append(cycle_time)
        
        client.disconnect()
        
        # Calculate metrics
        total_actions = len(actions_taken)
        forward_actions = actions_taken.count('FORWARD')
        
        metrics = {
            'total_actions': total_actions,
            'actions_per_second': total_actions / duration_seconds,
            'avg_cycle_time_ms': np.mean(cycle_times) if cycle_times else 0,
            'forward_ratio': forward_actions / total_actions if total_actions > 0 else 0,
            'collision_rate': collision_count / forward_actions if forward_actions > 0 else 0,
            'avg_light_distance': np.mean(light_distances) if light_distances else 0,
            'light_distance_improvement': 0,
            'avg_prediction_error': np.mean(prediction_errors) if prediction_errors else 0,
            'prediction_improvement': 0
        }
        
        # Calculate improvements
        if len(light_distances) >= 20:
            early_dist = np.mean(light_distances[:10])
            late_dist = np.mean(light_distances[-10:])
            if early_dist > 0:
                metrics['light_distance_improvement'] = (early_dist - late_dist) / early_dist
        
        if len(prediction_errors) >= 20:
            early_err = np.mean(prediction_errors[:10])
            late_err = np.mean(prediction_errors[-10:])
            if early_err > 0:
                metrics['prediction_improvement'] = (early_err - late_err) / early_err
        
        return metrics
    
    def compare_parameters(self, parameter_sets: Dict[str, Dict], 
                          test_duration: int = 60) -> Dict[str, Dict]:
        """Compare multiple parameter sets"""
        results = {}
        
        for name, params in parameter_sets.items():
            print(f"\n{'='*60}")
            print(f"📊 Testing: {name}")
            print(f"Parameters: {params}")
            
            # Apply settings
            self.custom_settings = params
            self.apply_custom_settings()
            
            # Run test
            try:
                metrics = self.run_quick_test(test_duration)
                results[name] = metrics
                
                # Print summary
                print(f"\n📈 Results for {name}:")
                print(f"   Cycle time: {metrics['avg_cycle_time_ms']:.1f}ms")
                print(f"   Light navigation: {metrics['avg_light_distance']:.2f} "
                      f"(improved {metrics['light_distance_improvement']:.1%})")
                print(f"   Collision rate: {metrics['collision_rate']:.1%}")
                print(f"   Prediction accuracy: {1 - metrics['avg_prediction_error']:.1%} "
                      f"(improved {metrics['prediction_improvement']:.1%})")
                
            except Exception as e:
                print(f"❌ Test failed: {e}")
                results[name] = {'error': str(e)}
            
            # Small delay between tests
            time.sleep(5)
        
        # Restore original settings
        self.restore_settings()
        
        # Find best configuration
        self.print_comparison(results)
        
        return results
    
    def print_comparison(self, results: Dict[str, Dict]):
        """Print comparison of results"""
        print(f"\n{'='*60}")
        print("📊 COMPARISON SUMMARY")
        print(f"{'='*60}")
        
        # Skip error results
        valid_results = {k: v for k, v in results.items() if 'error' not in v}
        
        if not valid_results:
            print("❌ No valid results to compare")
            return
        
        # Create comparison table
        metrics_to_compare = [
            ('Cycle Time (ms)', 'avg_cycle_time_ms', False),  # Lower is better
            ('Light Distance', 'avg_light_distance', False),   # Lower is better  
            ('Distance Improvement', 'light_distance_improvement', True),  # Higher is better
            ('Collision Rate', 'collision_rate', False),       # Lower is better
            ('Prediction Improvement', 'prediction_improvement', True)  # Higher is better
        ]
        
        print(f"\n{'Configuration':<20} ", end='')
        for metric_name, _, _ in metrics_to_compare:
            print(f"{metric_name:<20} ", end='')
        print()
        print("-" * (20 + len(metrics_to_compare) * 20))
        
        # Print each configuration's results
        for config_name, metrics in valid_results.items():
            print(f"{config_name:<20} ", end='')
            for _, metric_key, _ in metrics_to_compare:
                value = metrics.get(metric_key, 0)
                if isinstance(value, float):
                    if metric_key.endswith('_rate') or metric_key.endswith('_improvement'):
                        print(f"{value:>19.1%} ", end='')
                    else:
                        print(f"{value:>19.2f} ", end='')
                else:
                    print(f"{value:>19} ", end='')
            print()
        
        # Find best for each metric
        print(f"\n{'Best performers:':<20}")
        print("-" * 40)
        
        for metric_name, metric_key, higher_better in metrics_to_compare:
            if higher_better:
                best_config = max(valid_results.items(), key=lambda x: x[1].get(metric_key, 0))
            else:
                best_config = min(valid_results.items(), key=lambda x: x[1].get(metric_key, float('inf')))
            
            print(f"{metric_name:<25} {best_config[0]}")
        
        # Overall recommendation
        scores = {}
        for config_name, metrics in valid_results.items():
            # Normalize and combine metrics
            score = 0
            score += max(0, metrics.get('light_distance_improvement', 0)) * 2  # Weight navigation
            score += max(0, metrics.get('prediction_improvement', 0))
            score -= metrics.get('collision_rate', 0) * 2  # Penalize collisions
            score -= max(0, (metrics.get('avg_cycle_time_ms', 0) - 150) / 1000)  # Penalize slow cycles
            scores[config_name] = score
        
        best_overall = max(scores.items(), key=lambda x: x[1])
        
        print(f"\n🏆 RECOMMENDED CONFIGURATION: {best_overall[0]}")
        print(f"   Overall score: {best_overall[1]:.2f}")


def main():
    """Run quick parameter tests"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Quick parameter testing tool")
    parser.add_argument('--duration', type=int, default=60,
                       help='Test duration in seconds (default: 60)')
    parser.add_argument('--compare', action='store_true',
                       help='Compare multiple parameter sets')
    parser.add_argument('--custom', type=str,
                       help='Custom parameters as JSON string')
    
    args = parser.parse_args()
    
    tester = QuickParameterTest()
    
    if args.compare:
        # Compare different parameter sets
        parameter_sets = {
            'baseline': {},  # Use current settings
            'low_resolution': {
                'overrides': {'force_spatial_resolution': 8}
            },
            'high_resolution': {
                'overrides': {'force_spatial_resolution': 32}
            },
            'fast_learning': {
                'overrides': {'force_spatial_resolution': 16},
                'parameters': {'self_modification_rate': 0.1}
            },
            'conservative': {
                'overrides': {'force_spatial_resolution': 16},
                'parameters': {'self_modification_rate': 0.001}
            }
        }
        
        results = tester.compare_parameters(parameter_sets, args.duration)
        
    elif args.custom:
        # Test custom parameters
        try:
            custom_params = json.loads(args.custom)
            tester.custom_settings = custom_params
            tester.apply_custom_settings()
            
            metrics = tester.run_quick_test(args.duration)
            
            print("\n📊 Test Results:")
            for key, value in metrics.items():
                print(f"   {key}: {value}")
            
            tester.restore_settings()
            
        except json.JSONDecodeError:
            print("❌ Invalid JSON for custom parameters")
            
    else:
        # Run single test with current settings
        metrics = tester.run_quick_test(args.duration)
        
        print("\n📊 Test Results:")
        print(f"   Performance: {metrics['actions_per_second']:.1f} actions/sec")
        print(f"   Cycle time: {metrics['avg_cycle_time_ms']:.1f}ms")
        print(f"   Navigation: {metrics['avg_light_distance']:.2f} avg distance")
        print(f"   Learning: {metrics['light_distance_improvement']:.1%} improvement")
        print(f"   Safety: {metrics['collision_rate']:.1%} collision rate")


if __name__ == "__main__":
    main()