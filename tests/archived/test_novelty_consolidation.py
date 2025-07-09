"""
Test Novelty Detection and Node Consolidation System.
Demonstrates the new intelligent memory management that prevents "memory bloat"
by consolidating similar experiences instead of creating new nodes for everything.
"""

import sys
import time
import random
from typing import List, Dict

# Add project root to path
sys.path.append('.')

from core.novelty_detection import NoveltyDetector, ExperienceSignature
from core.node_consolidation import NodeConsolidationEngine
from core.world_graph import WorldGraph
from core.experience_node import ExperienceNode
from simulation.brainstem_sim import GridWorldBrainstem


class NoveltyConsolidationExperiment:
    """Experiment to demonstrate novelty detection and node consolidation."""
    
    def __init__(self, world_graph: WorldGraph):
        self.world_graph = world_graph
        self.novelty_detector = NoveltyDetector(world_graph)
        self.consolidation_engine = NodeConsolidationEngine(world_graph)
        
        # Test scenarios with varying levels of novelty
        self.test_scenarios = [
            # Scenario 1: Identical experiences (should consolidate)
            {
                'name': 'Identical Wall Encounters',
                'experiences': [
                    ([0.8, 0.2, 0.1, 0.0, 0.5, 0.3, 0.7, 0.4], {'forward_motor': 0.5, 'turn_motor': 0.0}),
                    ([0.8, 0.2, 0.1, 0.0, 0.5, 0.3, 0.7, 0.4], {'forward_motor': 0.5, 'turn_motor': 0.0}),
                    ([0.8, 0.2, 0.1, 0.0, 0.5, 0.3, 0.7, 0.4], {'forward_motor': 0.5, 'turn_motor': 0.0})
                ]
            },
            # Scenario 2: Similar experiences (should strengthen existing)
            {
                'name': 'Similar Food Encounters',
                'experiences': [
                    ([0.1, 0.9, 0.6, 0.3, 0.8, 0.2, 0.4, 0.7], {'forward_motor': 0.3, 'turn_motor': 0.2}),
                    ([0.1, 0.9, 0.6, 0.4, 0.8, 0.2, 0.4, 0.7], {'forward_motor': 0.3, 'turn_motor': 0.2}),
                    ([0.1, 0.9, 0.6, 0.3, 0.8, 0.3, 0.4, 0.7], {'forward_motor': 0.3, 'turn_motor': 0.2})
                ]
            },
            # Scenario 3: Novel experiences (should create new nodes)
            {
                'name': 'Novel Danger Encounters',
                'experiences': [
                    ([0.5, 0.5, 0.8, 0.9, 0.1, 0.6, 0.3, 0.2], {'forward_motor': -0.5, 'turn_motor': 0.8}),
                    ([0.2, 0.3, 0.1, 0.0, 0.9, 0.8, 0.7, 0.6], {'forward_motor': 0.0, 'turn_motor': -0.9}),
                    ([0.9, 0.1, 0.0, 0.0, 0.2, 0.4, 0.9, 0.8], {'forward_motor': 0.8, 'turn_motor': 0.0})
                ]
            }
        ]
        
        # Tracking
        self.experiment_results = []
        self.nodes_before = 0
        self.nodes_after = 0
    
    def run_experiment(self) -> Dict:
        """Run the novelty detection and consolidation experiment."""
        print("🧠 NOVELTY DETECTION & CONSOLIDATION EXPERIMENT")
        print("=" * 60)
        print(f"Initial graph size: {self.world_graph.node_count()} nodes")
        print()
        
        self.nodes_before = self.world_graph.node_count()
        
        # Run each scenario
        for scenario in self.test_scenarios:
            print(f"🔬 Testing: {scenario['name']}")
            scenario_results = self._run_scenario(scenario)
            self.experiment_results.append(scenario_results)
            print(f"   Results: {scenario_results['summary']}")
            print()
        
        self.nodes_after = self.world_graph.node_count()
        
        # Analyze overall results
        analysis = self._analyze_results()
        
        # Print comprehensive summary
        self._print_experiment_summary(analysis)
        
        return analysis
    
    def _run_scenario(self, scenario: Dict) -> Dict:
        """Run a single scenario with multiple experiences."""
        scenario_name = scenario['name']
        experiences = scenario['experiences']
        
        scenario_results = {
            'scenario_name': scenario_name,
            'total_experiences': len(experiences),
            'novelty_scores': [],
            'consolidation_strategies': [],
            'nodes_created': 0,
            'nodes_strengthened': 0,
            'nodes_merged': 0,
            'connections_created': 0,
            'summary': ''
        }
        
        # Track nodes before this scenario
        nodes_before_scenario = self.world_graph.node_count()
        
        # Process each experience
        for i, (mental_context, motor_action) in enumerate(experiences):
            # Create experience signature
            experience_signature = ExperienceSignature(
                mental_context=mental_context,
                motor_action=motor_action,
                sensory_outcome={j: val for j, val in enumerate(mental_context)},
                prediction_accuracy=0.8 + random.uniform(-0.2, 0.2),  # Simulate varying accuracy
                temporal_context=mental_context[:3],  # Use first 3 as temporal context
                drive_states={}
            )
            
            # Update novelty detector's context history
            self.novelty_detector.add_to_context_history(mental_context)
            
            # Evaluate novelty
            novelty_score = self.novelty_detector.evaluate_experience_novelty(experience_signature)
            scenario_results['novelty_scores'].append(novelty_score.overall_novelty)
            
            # Consolidate experience
            consolidation_result = self.consolidation_engine.consolidate_experience(
                experience_signature=experience_signature,
                novelty_score=novelty_score,
                prediction_success=True  # Assume success for this test
            )
            
            # Track consolidation strategy
            strategy = consolidation_result.strategy_used.value
            scenario_results['consolidation_strategies'].append(strategy)
            
            # Update counters
            if strategy == 'create_new':
                scenario_results['nodes_created'] += 1
                # Add to world graph for new nodes
                self.world_graph.add_node(consolidation_result.target_node)
            elif strategy == 'strengthen_existing':
                scenario_results['nodes_strengthened'] += 1
            elif strategy == 'merge_complete':
                scenario_results['nodes_merged'] += 1
            
            scenario_results['connections_created'] += len(consolidation_result.connections_created)
            
            print(f"      Experience {i+1}: Novelty={novelty_score.overall_novelty:.3f}, Strategy={strategy}")
        
        # Calculate nodes actually created in this scenario
        nodes_after_scenario = self.world_graph.node_count()
        actual_nodes_created = nodes_after_scenario - nodes_before_scenario
        
        # Create summary
        avg_novelty = sum(scenario_results['novelty_scores']) / len(scenario_results['novelty_scores'])
        scenario_results['summary'] = (
            f"Avg Novelty: {avg_novelty:.3f}, "
            f"Nodes +{actual_nodes_created}, "
            f"Strengthened: {scenario_results['nodes_strengthened']}, "
            f"Connections: {scenario_results['connections_created']}"
        )
        
        return scenario_results
    
    def _analyze_results(self) -> Dict:
        """Analyze the complete experiment results."""
        # Calculate overall statistics
        total_experiences = sum(r['total_experiences'] for r in self.experiment_results)
        total_nodes_created = sum(r['nodes_created'] for r in self.experiment_results)
        total_nodes_strengthened = sum(r['nodes_strengthened'] for r in self.experiment_results)
        total_nodes_merged = sum(r['nodes_merged'] for r in self.experiment_results)
        total_connections_created = sum(r['connections_created'] for r in self.experiment_results)
        
        # Calculate consolidation efficiency
        consolidation_rate = (total_nodes_strengthened + total_nodes_merged) / max(1, total_experiences)
        memory_efficiency = 1.0 - (total_nodes_created / max(1, total_experiences))
        
        # Get system statistics
        novelty_stats = self.novelty_detector.get_novelty_stats()
        consolidation_stats = self.consolidation_engine.get_consolidation_stats()
        
        # Calculate novelty distribution
        all_novelty_scores = []
        for result in self.experiment_results:
            all_novelty_scores.extend(result['novelty_scores'])
        
        avg_novelty = sum(all_novelty_scores) / len(all_novelty_scores) if all_novelty_scores else 0.0
        
        # Strategy distribution
        all_strategies = []
        for result in self.experiment_results:
            all_strategies.extend(result['consolidation_strategies'])
        
        strategy_distribution = {}
        for strategy in all_strategies:
            strategy_distribution[strategy] = strategy_distribution.get(strategy, 0) + 1
        
        return {
            'experiment_overview': {
                'total_experiences': total_experiences,
                'nodes_before': self.nodes_before,
                'nodes_after': self.nodes_after,
                'net_nodes_created': self.nodes_after - self.nodes_before,
                'memory_efficiency': memory_efficiency,
                'consolidation_rate': consolidation_rate
            },
            'consolidation_breakdown': {
                'nodes_created': total_nodes_created,
                'nodes_strengthened': total_nodes_strengthened,
                'nodes_merged': total_nodes_merged,
                'connections_created': total_connections_created
            },
            'novelty_analysis': {
                'average_novelty': avg_novelty,
                'novelty_distribution': all_novelty_scores,
                'strategy_distribution': strategy_distribution
            },
            'system_performance': {
                'novelty_stats': novelty_stats,
                'consolidation_stats': consolidation_stats
            },
            'scenario_results': self.experiment_results
        }
    
    def _print_experiment_summary(self, analysis: Dict):
        """Print comprehensive experiment summary."""
        print("📊 NOVELTY DETECTION & CONSOLIDATION RESULTS")
        print("=" * 60)
        
        # Overview
        overview = analysis['experiment_overview']
        print(f"Total experiences processed: {overview['total_experiences']}")
        print(f"Nodes before: {overview['nodes_before']}")
        print(f"Nodes after: {overview['nodes_after']}")
        print(f"Net nodes created: {overview['net_nodes_created']}")
        print(f"Memory efficiency: {overview['memory_efficiency']:.1%}")
        print(f"Consolidation rate: {overview['consolidation_rate']:.1%}")
        print()
        
        # Consolidation breakdown
        breakdown = analysis['consolidation_breakdown']
        print("🔧 CONSOLIDATION BREAKDOWN:")
        print(f"  New nodes created: {breakdown['nodes_created']}")
        print(f"  Existing nodes strengthened: {breakdown['nodes_strengthened']}")
        print(f"  Nodes merged: {breakdown['nodes_merged']}")
        print(f"  New connections created: {breakdown['connections_created']}")
        print()
        
        # Novelty analysis
        novelty = analysis['novelty_analysis']
        print("🎯 NOVELTY ANALYSIS:")
        print(f"  Average novelty score: {novelty['average_novelty']:.3f}")
        print("  Strategy distribution:")
        for strategy, count in novelty['strategy_distribution'].items():
            percentage = (count / overview['total_experiences']) * 100
            print(f"    {strategy}: {count} ({percentage:.1f}%)")
        print()
        
        # System performance
        performance = analysis['system_performance']
        print("⚡ SYSTEM PERFORMANCE:")
        novelty_stats = performance['novelty_stats']
        consolidation_stats = performance['consolidation_stats']
        
        print(f"  Novelty evaluations: {novelty_stats['evaluations_performed']}")
        print(f"  Consolidations performed: {consolidation_stats['consolidations_performed']}")
        print(f"  Memory pressure factor: {novelty_stats['memory_pressure_factor']:.2f}")
        print(f"  Average consolidation confidence: {consolidation_stats.get('average_confidence', 0.0):.3f}")
        print()
        
        # Key achievements
        print("🏆 KEY ACHIEVEMENTS:")
        if overview['memory_efficiency'] > 0.5:
            print("  ✅ Significant memory efficiency improvement")
        if overview['consolidation_rate'] > 0.3:
            print("  ✅ High consolidation rate - system is learning patterns")
        if breakdown['nodes_strengthened'] > breakdown['nodes_created']:
            print("  ✅ More strengthening than creation - intelligent consolidation")
        if breakdown['connections_created'] > 0:
            print("  ✅ Creating connections between related experiences")
        
        # Compare to old system
        old_system_nodes = overview['total_experiences']  # Old system would create one node per experience
        nodes_saved = old_system_nodes - overview['net_nodes_created']
        savings_percentage = (nodes_saved / old_system_nodes) * 100 if old_system_nodes > 0 else 0
        
        print()
        print(f"🎪 MEMORY SAVINGS vs OLD SYSTEM:")
        print(f"  Old system would create: {old_system_nodes} nodes")
        print(f"  New system created: {overview['net_nodes_created']} nodes")
        print(f"  Nodes saved: {nodes_saved} ({savings_percentage:.1f}% reduction)")
        print()
        
        if savings_percentage > 50:
            print("🚀 NOVELTY DETECTION SYSTEM SUCCESSFULLY PREVENTS MEMORY BLOAT!")
        else:
            print("⚠️  System needs tuning - savings could be improved")


def main():
    """Run the novelty detection and consolidation experiment."""
    print("🚀 Loading brain system...")
    
    # Load existing brain system
    brainstem = GridWorldBrainstem(seed=42, use_sockets=False)
    world_graph = brainstem.brain_client.get_world_graph()
    
    if world_graph.node_count() < 10:
        print("ℹ️  Starting with minimal graph for cleaner demonstration")
        print(f"Current graph has {world_graph.node_count()} nodes")
    else:
        print(f"✅ Loaded graph with {world_graph.node_count()} nodes")
    
    print()
    
    # Run experiment
    experiment = NoveltyConsolidationExperiment(world_graph)
    results = experiment.run_experiment()
    
    return results


if __name__ == "__main__":
    results = main()