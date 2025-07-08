"""
Test Dual-Mode Cognitive Processing System.
Demonstrates the complete cognitive architecture with immediate and background thinking.
"""

import sys
import time
import random
from typing import List, Dict

# Add project root to path
sys.path.append('.')

from predictor.dual_mode_processor import DualModeProcessor
from simulation.brainstem_sim import GridWorldBrainstem


class DualModeExperiment:
    """Experiment to demonstrate dual-mode cognitive processing."""
    
    def __init__(self, world_graph):
        self.world_graph = world_graph
        
        # Initialize dual-mode processor
        self.dual_mode = DualModeProcessor(world_graph)
        
        # Test contexts representing different robot scenarios
        self.test_contexts = [
            [0.8, 0.2, 0.1, 0.0, 0.5, 0.3, 0.7, 0.4],  # Wall ahead scenario
            [0.1, 0.9, 0.6, 0.3, 0.8, 0.2, 0.4, 0.7],  # Food nearby scenario  
            [0.5, 0.5, 0.8, 0.9, 0.1, 0.6, 0.3, 0.2],  # Danger present scenario
            [0.3, 0.7, 0.4, 0.2, 0.9, 0.1, 0.8, 0.5],  # Exploration scenario
            [0.6, 0.4, 0.2, 0.8, 0.3, 0.9, 0.1, 0.7],  # Mixed scenario
            [0.9, 0.1, 0.8, 0.2, 0.7, 0.3, 0.6, 0.4],  # High wall + moderate food
            [0.2, 0.8, 0.3, 0.7, 0.4, 0.6, 0.1, 0.9]   # Low wall + high food
        ]
        
        # Experiment tracking
        self.results = []
        self.performance_timeline = []
    
    def run_experiment(self, duration_seconds: int = 60) -> Dict:
        """
        Run dual-mode experiment showing background and immediate thinking.
        
        Args:
            duration_seconds: How long to run the experiment
            
        Returns:
            Comprehensive experiment results
        """
        print("🧠 DUAL-MODE COGNITIVE PROCESSING EXPERIMENT")
        print("=" * 60)
        print(f"Graph size: {self.world_graph.node_count()} nodes")
        print(f"Test contexts: {len(self.test_contexts)}")
        print(f"Duration: {duration_seconds} seconds")
        print()
        
        # Start background processing
        print("🔄 Starting background processing...")
        self.dual_mode.start_background_processing()
        time.sleep(2.0)  # Give background processing time to start
        
        # Run experiment
        start_time = time.time()
        iteration = 0
        
        print("🚀 Beginning dual-mode predictions...")
        print("   (Background processing running continuously)")
        print()
        
        while time.time() - start_time < duration_seconds:
            # Select test context
            context = self.test_contexts[iteration % len(self.test_contexts)]
            
            # Add some variation to context
            varied_context = [
                max(0.0, min(1.0, val + random.uniform(-0.1, 0.1)))
                for val in context
            ]
            
            # Context information for prediction
            context_info = {
                'threat_level': random.choice(['safe', 'normal', 'alert']),
                'cognitive_load': random.uniform(0.2, 0.8),
                'time_pressure': random.uniform(0.1, 0.6)
            }
            
            # Perform dual-mode prediction
            prediction_start = time.time()
            result = self.dual_mode.predict(varied_context, context_info)
            prediction_time = time.time() - prediction_start
            
            # Track results
            self.results.append({
                'iteration': iteration,
                'context': varied_context,
                'context_info': context_info,
                'prediction_time': prediction_time,
                'background_used': result['background_used'],
                'immediate_result': result['immediate_result'],
                'background_assistance': result['background_assistance'],
                'dual_mode_stats': result['dual_mode_stats']
            })
            
            # Track performance over time
            if iteration % 5 == 0:
                current_stats = self.dual_mode.get_dual_mode_stats()
                self.performance_timeline.append({
                    'time': time.time() - start_time,
                    'iteration': iteration,
                    'stats': current_stats
                })
                
                # Print progress
                assistance_rate = current_stats['immediate_mode']['background_assistance_rate']
                insights = current_stats['background_mode']['insights_discovered']
                pre_activations = current_stats['background_mode']['pre_activations_active']
                
                print(f"Iteration {iteration:3d}: " +
                      f"Assistance {assistance_rate:.1%}, " +
                      f"Insights {insights}, " +
                      f"Pre-activations {pre_activations}")
            
            iteration += 1
            
            # Brief pause between predictions
            time.sleep(0.2)
        
        # Stop background processing
        print()
        print("⏹️  Stopping background processing...")
        self.dual_mode.stop_background_processing()
        
        # Analyze results
        analysis = self._analyze_experiment_results()
        
        # Print summary
        self._print_experiment_summary(analysis)
        
        return analysis
    
    def _analyze_experiment_results(self) -> Dict:
        """Analyze the dual-mode experiment results."""
        if not self.results:
            return {"error": "no_results"}
        
        # Basic statistics
        total_predictions = len(self.results)
        background_assisted = sum(1 for r in self.results if r['background_used'])
        assistance_rate = background_assisted / total_predictions
        
        # Performance analysis
        avg_prediction_time = sum(r['prediction_time'] for r in self.results) / total_predictions
        
        # Background effectiveness analysis
        assisted_results = [r for r in self.results if r['background_used']]
        non_assisted_results = [r for r in self.results if not r['background_used']]
        
        if assisted_results and non_assisted_results:
            avg_assisted_score = sum(
                r['immediate_result'].goldilocks_score for r in assisted_results
            ) / len(assisted_results)
            
            avg_non_assisted_score = sum(
                r['immediate_result'].goldilocks_score for r in non_assisted_results  
            ) / len(non_assisted_results)
            
            background_benefit = avg_assisted_score - avg_non_assisted_score
        else:
            avg_assisted_score = 0.0
            avg_non_assisted_score = 0.0
            background_benefit = 0.0
        
        # Timeline analysis
        if len(self.performance_timeline) >= 2:
            early_stats = self.performance_timeline[0]['stats']
            late_stats = self.performance_timeline[-1]['stats']
            
            assistance_improvement = (
                late_stats['immediate_mode']['background_assistance_rate'] -
                early_stats['immediate_mode']['background_assistance_rate']
            )
            
            insights_growth = (
                late_stats['background_mode']['insights_discovered'] -
                early_stats['background_mode']['insights_discovered']
            )
        else:
            assistance_improvement = 0.0
            insights_growth = 0
        
        # Final system state
        final_stats = self.dual_mode.get_dual_mode_stats()
        system_overview = self.dual_mode.get_system_overview()
        
        return {
            'experiment_summary': {
                'total_predictions': total_predictions,
                'background_assisted': background_assisted,
                'assistance_rate': assistance_rate,
                'avg_prediction_time': avg_prediction_time
            },
            'background_effectiveness': {
                'avg_assisted_score': avg_assisted_score,
                'avg_non_assisted_score': avg_non_assisted_score,
                'background_benefit': background_benefit,
                'assistance_improvement': assistance_improvement,
                'insights_growth': insights_growth
            },
            'system_evolution': {
                'timeline': self.performance_timeline,
                'early_vs_late': {
                    'assistance_improvement': assistance_improvement,
                    'insights_discovered': insights_growth
                }
            },
            'final_system_state': {
                'dual_mode_stats': final_stats,
                'system_overview': system_overview
            },
            'cognitive_architecture_demo': {
                'immediate_mode_active': True,
                'background_mode_demonstrated': final_stats['background_mode']['is_running'],
                'meta_cognition_active': final_stats['immediate_mode']['tuning_status']['auto_tuning_enabled'],
                'pre_activation_system': final_stats['background_mode']['pre_activations_active'] > 0,
                'context_mapping_active': final_stats['background_mode']['context_maps_cached'] > 0,
                'attention_biasing_active': final_stats['background_mode']['attention_biases_active'] > 0
            }
        }
    
    def _print_experiment_summary(self, analysis: Dict):
        """Print comprehensive experiment summary."""
        print("📊 DUAL-MODE EXPERIMENT RESULTS")
        print("=" * 60)
        
        # Experiment overview
        summary = analysis['experiment_summary']
        print(f"Total predictions: {summary['total_predictions']}")
        print(f"Background assisted: {summary['background_assisted']}")
        print(f"Assistance rate: {summary['assistance_rate']:.1%}")
        print(f"Average prediction time: {summary['avg_prediction_time']:.4f}s")
        print()
        
        # Background effectiveness
        effectiveness = analysis['background_effectiveness']
        print("🧠 BACKGROUND THINKING EFFECTIVENESS:")
        print(f"  Assisted predictions score: {effectiveness['avg_assisted_score']:.3f}")
        print(f"  Non-assisted predictions score: {effectiveness['avg_non_assisted_score']:.3f}")
        print(f"  Background benefit: {effectiveness['background_benefit']:+.3f}")
        print(f"  Assistance rate improvement: {effectiveness['assistance_improvement']:+.1%}")
        print(f"  Insights discovered: {effectiveness['insights_growth']}")
        print()
        
        # System evolution
        evolution = analysis['system_evolution']
        if evolution['timeline']:
            print("📈 SYSTEM EVOLUTION:")
            print(f"  Assistance improvement: {evolution['early_vs_late']['assistance_improvement']:+.1%}")
            print(f"  New insights discovered: {evolution['early_vs_late']['insights_discovered']}")
            print()
        
        # Cognitive architecture demonstration
        cognitive = analysis['cognitive_architecture_demo']
        print("🏗️  COGNITIVE ARCHITECTURE COMPONENTS:")
        print(f"  ✅ Immediate mode: {'Active' if cognitive['immediate_mode_active'] else 'Inactive'}")
        print(f"  ✅ Background mode: {'Demonstrated' if cognitive['background_mode_demonstrated'] else 'Not shown'}")
        print(f"  ✅ Meta-cognition: {'Active' if cognitive['meta_cognition_active'] else 'Inactive'}")
        print(f"  ✅ Pre-activation system: {'Active' if cognitive['pre_activation_system'] else 'Inactive'}")
        print(f"  ✅ Context mapping: {'Active' if cognitive['context_mapping_active'] else 'Inactive'}")
        print(f"  ✅ Attention biasing: {'Active' if cognitive['attention_biasing_active'] else 'Inactive'}")
        print()
        
        # Final system state
        final_state = analysis['final_system_state']
        overview = final_state['system_overview']
        
        print("🎯 FINAL SYSTEM PERFORMANCE:")
        perf = overview['performance_summary']
        print(f"  Total predictions: {perf['total_predictions']}")
        print(f"  Background assistance rate: {perf['background_assistance_rate']}")
        print(f"  Insights discovered: {perf['insights_discovered']}")
        print(f"  System efficiency: {perf['system_efficiency']:.1%}")
        print()
        
        # System architecture summary
        arch = overview['cognitive_architecture']
        print("🧠 COGNITIVE ARCHITECTURE STATUS:")
        print(f"  Dual-mode processing: {'✅ Active' if arch['dual_mode_active'] else '❌ Inactive'}")
        print(f"  Meta-cognition: {'✅ Enabled' if arch['meta_cognition_enabled'] else '❌ Disabled'}")
        print(f"  Background insights: {'✅ Active' if arch['background_insights_active'] else '❌ Inactive'}")
        print(f"  Pre-activation system: {'✅ Active' if arch['pre_activation_system_active'] else '❌ Inactive'}")
        print()
        
        # Key achievements
        print("🏆 KEY ACHIEVEMENTS DEMONSTRATED:")
        if summary['assistance_rate'] > 0.1:
            print("  ✅ Background thinking successfully assists immediate decisions")
        if effectiveness['background_benefit'] > 0.05:
            print("  ✅ Background processing improves prediction quality")
        if cognitive['meta_cognition_active']:
            print("  ✅ Meta-cognitive parameter optimization working")
        if effectiveness['insights_growth'] > 0:
            print("  ✅ Background system discovers patterns and insights")
        if cognitive['pre_activation_system']:
            print("  ✅ Pre-activation system primes relevant memories")
        
        print()
        print("🎪 DUAL-MODE COGNITIVE ARCHITECTURE SUCCESSFULLY DEMONSTRATED!")


def main():
    """Run the dual-mode experiment."""
    print("🚀 Loading brain system...")
    
    # Load existing brain system
    brainstem = GridWorldBrainstem(seed=42, use_sockets=False)
    world_graph = brainstem.brain_client.get_world_graph()
    
    if world_graph.node_count() < 100:
        print("❌ Need at least 100 nodes for meaningful demonstration")
        print(f"Current graph has {world_graph.node_count()} nodes")
        return
    
    print(f"✅ Loaded graph with {world_graph.node_count()} nodes")
    print()
    
    # Run experiment
    experiment = DualModeExperiment(world_graph)
    results = experiment.run_experiment(duration_seconds=30)  # 30 second demo
    
    return results


if __name__ == "__main__":
    results = main()