"""
Brain Scaling Analysis Tool

Analyzes the scaling limits of the minimal brain implementation to understand:
1. Current memory usage patterns for 60-experience brain
2. Physical memory limits for different hardware configurations  
3. Computational limits for real-time operation
4. Biological intelligence comparisons
5. Architecture scaling challenges
6. Practical deployment scenarios

Run from project root:
python3 tools/brain_scaling_analysis.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import time
import psutil
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass

# Import brain components
from src.brain import MinimalBrain
from src.experience.models import Experience
from src.similarity.engine import SimilarityEngine
from src.activation.dynamics import ActivationDynamics


@dataclass
class HardwareSpec:
    """Hardware specification for scaling analysis."""
    name: str
    ram_gb: int
    vram_gb: int
    gpu_compute_units: int  # Relative compute power (RTX 3070 = 100)
    bandwidth_gbps: float   # Memory bandwidth


@dataclass
class ScalingResults:
    """Results from scaling analysis."""
    hardware_name: str
    max_experiences_memory: int
    max_experiences_realtime: int
    memory_per_experience_bytes: int
    compute_time_per_experience_ms: float
    working_memory_limit: int
    biological_equivalent_neurons: int


class BrainScalingAnalyzer:
    """Analyzes brain scaling limits across different hardware configurations."""
    
    def __init__(self):
        """Initialize the scaling analyzer."""
        # Define hardware configurations
        self.hardware_configs = {
            'rtx_3070': HardwareSpec('RTX 3070', 32, 24, 100, 896),
            'rtx_4090': HardwareSpec('RTX 4090', 64, 24, 183, 1008),
            'workstation': HardwareSpec('High-end Workstation', 128, 48, 250, 1500),
            'datacenter': HardwareSpec('Datacenter GPU Cluster', 512, 192, 1000, 5000)
        }
        
        # Biological comparison constants
        self.human_neurons = 86_000_000_000
        self.human_synapses = 150_000_000_000_000
        self.c_elegans_neurons = 302
        
        # Performance targets
        self.realtime_target_ms = 50  # 20 FPS for real-time robotics
        self.interactive_target_ms = 100  # 10 FPS for interactive systems
        
        print("🧠 Brain Scaling Analyzer initialized")
        print(f"Target: <{self.realtime_target_ms}ms per cycle for real-time operation")
    
    def analyze_current_brain_characteristics(self) -> Dict[str, Any]:
        """Analyze memory usage patterns of current brain implementation."""
        print("\n📊 Analyzing current brain characteristics...")
        
        # Create test brain with typical configuration
        brain = MinimalBrain(enable_logging=False, enable_persistence=False)
        
        # Generate representative experiences 
        experiences_data = []
        sensory_dims = 20  # Typical sensor vector size
        action_dims = 4    # Typical action vector size
        outcome_dims = 20  # Typical outcome vector size
        
        start_time = time.time()
        memory_usage_before = psutil.Process().memory_info().rss
        
        # Add 60 experiences (typical current size)
        for i in range(60):
            sensory_input = [np.random.uniform(-1, 1) for _ in range(sensory_dims)]
            action_taken = [np.random.uniform(-1, 1) for _ in range(action_dims)]
            outcome = [np.random.uniform(-1, 1) for _ in range(outcome_dims)]
            
            # Process and store experience
            predicted_action, brain_state = brain.process_sensory_input(sensory_input, action_dims)
            experience_id = brain.store_experience(sensory_input, action_taken, outcome, predicted_action)
            experiences_data.append({
                'id': experience_id,
                'sensory_dims': sensory_dims,
                'action_dims': action_dims,
                'outcome_dims': outcome_dims
            })
        
        memory_usage_after = psutil.Process().memory_info().rss
        total_time = time.time() - start_time
        
        # Get brain statistics
        brain_stats = brain.get_brain_stats()
        
        # Calculate memory usage per experience
        memory_diff_bytes = memory_usage_after - memory_usage_before
        memory_per_experience = memory_diff_bytes / 60
        
        # Calculate computation time per experience
        compute_time_per_experience = (total_time / 60) * 1000  # ms
        
        # Get working memory characteristics
        working_memory_size = brain_stats['brain_summary']['total_experiences']
        
        # Calculate vector memory overhead
        vectors_per_exp = sensory_dims + action_dims + outcome_dims
        raw_vector_bytes = vectors_per_exp * 8  # 8 bytes per float64
        overhead_ratio = memory_per_experience / raw_vector_bytes if raw_vector_bytes > 0 else 0
        
        results = {
            'total_experiences': 60,
            'memory_per_experience_bytes': int(memory_per_experience),
            'raw_vector_bytes_per_exp': raw_vector_bytes,
            'memory_overhead_ratio': overhead_ratio,
            'compute_time_per_experience_ms': compute_time_per_experience,
            'working_memory_size': working_memory_size,
            'total_memory_mb': memory_diff_bytes / (1024 * 1024),
            'vector_dimensions': {
                'sensory': sensory_dims,
                'action': action_dims, 
                'outcome': outcome_dims,
                'total_per_experience': vectors_per_exp
            },
            'activation_dynamics': brain_stats['activation_dynamics'],
            'similarity_engine': brain_stats['similarity_engine']
        }
        
        print(f"Memory per experience: {memory_per_experience:.0f} bytes")
        print(f"Raw vector storage: {raw_vector_bytes} bytes")
        print(f"Memory overhead: {overhead_ratio:.1f}x")
        print(f"Compute time per experience: {compute_time_per_experience:.2f}ms")
        print(f"Working memory size: {working_memory_size}")
        
        # Cleanup
        brain.reset_brain()
        del brain
        
        return results
    
    def calculate_physical_memory_limits(self, brain_characteristics: Dict[str, Any]) -> Dict[str, ScalingResults]:
        """Calculate maximum brain sizes that fit in memory for each hardware configuration."""
        print("\n💾 Calculating physical memory limits...")
        
        memory_per_exp = brain_characteristics['memory_per_experience_bytes']
        compute_time_per_exp = brain_characteristics['compute_time_per_experience_ms']
        
        results = {}
        
        for hw_key, hw_spec in self.hardware_configs.items():
            print(f"\nAnalyzing {hw_spec.name}:")
            
            # Calculate memory-constrained limits
            # Use 80% of available memory to account for OS overhead
            total_memory_bytes = (hw_spec.ram_gb + hw_spec.vram_gb) * 1024 * 1024 * 1024 * 0.8
            max_experiences_memory = int(total_memory_bytes / memory_per_exp)
            
            # Calculate compute-constrained limits for real-time operation
            # Scale compute time by relative GPU performance
            performance_factor = hw_spec.gpu_compute_units / 100.0  # RTX 3070 baseline
            adjusted_compute_time = compute_time_per_exp / performance_factor
            
            # Calculate max experiences for real-time operation
            max_experiences_realtime = int(self.realtime_target_ms / adjusted_compute_time)
            
            # Estimate working memory limits (typically 10-20% of total experiences)
            working_memory_limit = min(max_experiences_memory // 10, max_experiences_realtime // 5)
            
            # Estimate biological equivalent
            # Very rough approximation: 1 experience ≈ 1000 synaptic connections
            biological_equivalent = max_experiences_memory * 1000
            
            result = ScalingResults(
                hardware_name=hw_spec.name,
                max_experiences_memory=max_experiences_memory,
                max_experiences_realtime=max_experiences_realtime,
                memory_per_experience_bytes=memory_per_exp,
                compute_time_per_experience_ms=adjusted_compute_time,
                working_memory_limit=working_memory_limit,
                biological_equivalent_neurons=biological_equivalent
            )
            
            results[hw_key] = result
            
            print(f"  Max experiences (memory): {max_experiences_memory:,}")
            print(f"  Max experiences (real-time): {max_experiences_realtime:,}")
            print(f"  Compute time per exp: {adjusted_compute_time:.3f}ms")
            print(f"  Working memory limit: {working_memory_limit:,}")
            print(f"  Biological equivalent: {biological_equivalent:,} synapses")
        
        return results
    
    def analyze_computational_scaling(self, brain_characteristics: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze computational complexity and scaling bottlenecks."""
        print("\n⚡ Analyzing computational scaling limits...")
        
        base_compute_time = brain_characteristics['compute_time_per_experience_ms']
        
        # Analyze scaling complexity for each brain component
        scaling_analysis = {
            'similarity_search': {
                'complexity': 'O(n)',
                'description': 'Linear search through all experiences',
                'bottleneck_size': 50000,  # When similarity search becomes prohibitive
                'mitigation': 'Approximate nearest neighbor (ANN) indexing'
            },
            'activation_dynamics': {
                'complexity': 'O(working_memory_size)',
                'description': 'Spreads activation through working memory',
                'bottleneck_size': 5000,  # Working memory practical limit
                'mitigation': 'Sparse activation patterns, attention mechanisms'
            },
            'connection_matrix': {
                'complexity': 'O(n²) memory, O(n) compute',
                'description': 'Similarity connections between experiences',
                'bottleneck_size': 10000,  # When O(n²) memory becomes prohibitive
                'mitigation': 'Sparse matrices, connection pruning'
            },
            'prediction_engine': {
                'complexity': 'O(working_memory_size)',
                'description': 'Consensus from activated experiences',
                'bottleneck_size': 2000,  # Practical consensus limit
                'mitigation': 'Hierarchical prediction, sampling strategies'
            }
        }
        
        # Calculate projected performance at different scales
        experience_scales = [100, 500, 1000, 5000, 10000, 50000, 100000]
        projected_performance = {}
        
        for scale in experience_scales:
            # Similarity search: O(n) with large constant
            similarity_time = base_compute_time * (scale / 60) * 0.7  # 70% of time in similarity
            
            # Activation dynamics: O(working_memory) 
            working_memory_size = min(scale // 10, 1000)  # Cap at 1000
            activation_time = base_compute_time * (working_memory_size / 10) * 0.2  # 20% of time
            
            # Prediction: O(working_memory)
            prediction_time = base_compute_time * (working_memory_size / 10) * 0.1  # 10% of time
            
            total_time = similarity_time + activation_time + prediction_time
            
            projected_performance[scale] = {
                'total_time_ms': total_time,
                'similarity_time_ms': similarity_time,
                'activation_time_ms': activation_time,
                'prediction_time_ms': prediction_time,
                'working_memory_size': working_memory_size,
                'realtime_feasible': total_time < self.realtime_target_ms
            }
        
        return {
            'scaling_analysis': scaling_analysis,
            'projected_performance': projected_performance,
            'realtime_target_ms': self.realtime_target_ms
        }
    
    def compare_to_biological_intelligence(self, scaling_results: Dict[str, ScalingResults]) -> Dict[str, Any]:
        """Compare brain scales to biological intelligence."""
        print("\n🧬 Comparing to biological intelligence...")
        
        # Calculate biological equivalences for each hardware config
        biological_comparisons = {}
        
        for hw_key, result in scaling_results.items():
            max_experiences = result.max_experiences_memory
            max_realtime = result.max_experiences_realtime
            
            # Very rough biological comparisons
            # Assumption: 1 experience ≈ 1000 synaptic connections
            synaptic_equivalent = max_experiences * 1000
            
            # Compare to known organisms
            comparisons = {
                'c_elegans_ratio': synaptic_equivalent / (self.c_elegans_neurons * 7000),  # ~7k synapses per neuron
                'human_ratio': synaptic_equivalent / self.human_synapses,
                'memory_constrained': {
                    'experiences': max_experiences,
                    'synaptic_equivalent': synaptic_equivalent,
                    'organism_equivalent': self._get_organism_equivalent(synaptic_equivalent)
                },
                'realtime_constrained': {
                    'experiences': max_realtime,
                    'synaptic_equivalent': max_realtime * 1000,
                    'organism_equivalent': self._get_organism_equivalent(max_realtime * 1000)
                }
            }
            
            biological_comparisons[hw_key] = comparisons
            
            print(f"\n{result.hardware_name}:")
            print(f"  Memory-constrained: {max_experiences:,} experiences")
            print(f"  ≈ {synaptic_equivalent:,} synapses")
            print(f"  ≈ {comparisons['memory_constrained']['organism_equivalent']}")
            print(f"  Real-time constrained: {max_realtime:,} experiences")
            print(f"  ≈ {comparisons['realtime_constrained']['organism_equivalent']}")
        
        return biological_comparisons
    
    def _get_organism_equivalent(self, synaptic_count: int) -> str:
        """Get biological organism equivalent for given synapse count."""
        if synaptic_count < 1000:
            return "Simple neural circuit"
        elif synaptic_count < 10000:
            return "Basic invertebrate (partial C. elegans)"
        elif synaptic_count < 100000:
            return "Complex invertebrate (C. elegans+)"
        elif synaptic_count < 1000000:
            return "Simple vertebrate nervous system"
        elif synaptic_count < 10000000:
            return "Fish/amphibian brain"
        elif synaptic_count < 100000000:
            return "Bird/reptile brain"
        elif synaptic_count < 1000000000:
            return "Small mammal brain (mouse)"
        elif synaptic_count < 10000000000:
            return "Medium mammal brain (cat)"
        elif synaptic_count < 100000000000:
            return "Large mammal brain (dog)"
        elif synaptic_count < self.human_synapses:
            return "Primate brain (approaching human)"
        else:
            return "Superhuman cognitive capacity"
    
    def identify_scaling_bottlenecks(self, computational_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Identify the primary bottlenecks limiting brain scale."""
        print("\n🚧 Identifying scaling bottlenecks...")
        
        bottlenecks = {}
        
        # Analyze where each component becomes prohibitive
        for scale, performance in computational_analysis['projected_performance'].items():
            total_time = performance['total_time_ms']
            components = {
                'similarity_search': performance['similarity_time_ms'],
                'activation_dynamics': performance['activation_time_ms'],
                'prediction_engine': performance['prediction_time_ms']
            }
            
            # Find the dominant component
            dominant_component = max(components.items(), key=lambda x: x[1])
            
            bottlenecks[scale] = {
                'total_time_ms': total_time,
                'dominant_bottleneck': dominant_component[0],
                'bottleneck_time_ms': dominant_component[1],
                'bottleneck_percentage': (dominant_component[1] / total_time) * 100,
                'realtime_feasible': performance['realtime_feasible']
            }
        
        # Identify critical transition points
        transition_points = {
            'realtime_limit': None,
            'similarity_dominates': None,
            'activation_dominates': None
        }
        
        for scale in sorted(bottlenecks.keys()):
            data = bottlenecks[scale]
            
            if transition_points['realtime_limit'] is None and not data['realtime_feasible']:
                transition_points['realtime_limit'] = scale
            
            if data['dominant_bottleneck'] == 'similarity_search' and transition_points['similarity_dominates'] is None:
                transition_points['similarity_dominates'] = scale
            
            if data['dominant_bottleneck'] == 'activation_dynamics' and transition_points['activation_dominates'] is None:
                transition_points['activation_dominates'] = scale
        
        return {
            'bottleneck_analysis': bottlenecks,
            'transition_points': transition_points,
            'scaling_recommendations': self._generate_scaling_recommendations(transition_points)
        }
    
    def _generate_scaling_recommendations(self, transition_points: Dict[str, int]) -> List[str]:
        """Generate recommendations for scaling beyond current limits."""
        recommendations = []
        
        realtime_limit = transition_points.get('realtime_limit')
        if realtime_limit:
            recommendations.append(
                f"Real-time limit reached at {realtime_limit:,} experiences - "
                "implement approximate nearest neighbor (ANN) indexing"
            )
        
        recommendations.extend([
            "Implement hierarchical similarity search (coarse-to-fine)",
            "Use sparse activation patterns to reduce working memory overhead",
            "Implement connection pruning to manage O(n²) similarity matrix growth",
            "Consider temporal clustering to group related experiences",
            "Implement adaptive attention mechanisms to focus computation",
            "Use memory-mapped storage for large experience databases",
            "Implement distributed computation for datacenter deployment"
        ])
        
        return recommendations
    
    def create_deployment_scenarios(self, scaling_results: Dict[str, ScalingResults]) -> Dict[str, Any]:
        """Create practical deployment scenarios for different hardware."""
        print("\n🚀 Creating deployment scenarios...")
        
        scenarios = {}
        
        for hw_key, result in scaling_results.items():
            # Determine practical cognitive capabilities
            max_experiences = min(result.max_experiences_memory, result.max_experiences_realtime)
            
            if max_experiences < 1000:
                capability_level = "Basic reactive behaviors"
                applications = ["Simple obstacle avoidance", "Basic pattern recognition"]
            elif max_experiences < 5000:
                capability_level = "Structured learning behaviors"
                applications = ["Spatial navigation", "Motor skill acquisition", "Simple planning"]
            elif max_experiences < 20000:
                capability_level = "Complex adaptive behaviors"
                applications = ["Multi-task learning", "Social interaction", "Tool use"]
            elif max_experiences < 100000:
                capability_level = "Sophisticated intelligence"
                applications = ["Language understanding", "Abstract reasoning", "Creative problem solving"]
            else:
                capability_level = "Superhuman cognitive capacity"
                applications = ["Research assistant", "Creative AI", "Scientific discovery"]
            
            scenarios[hw_key] = {
                'hardware': result.hardware_name,
                'max_experiences': max_experiences,
                'capability_level': capability_level,
                'applications': applications,
                'deployment_type': self._get_deployment_type(hw_key),
                'realtime_performance': result.max_experiences_realtime,
                'memory_performance': result.max_experiences_memory,
                'limiting_factor': 'compute' if result.max_experiences_realtime < result.max_experiences_memory else 'memory'
            }
            
            print(f"\n{result.hardware_name}:")
            print(f"  Max practical experiences: {max_experiences:,}")
            print(f"  Capability level: {capability_level}")
            print(f"  Limiting factor: {scenarios[hw_key]['limiting_factor']}")
            print(f"  Applications: {', '.join(applications[:2])}...")
        
        return scenarios
    
    def _get_deployment_type(self, hw_key: str) -> str:
        """Determine deployment type for hardware configuration."""
        deployment_types = {
            'rtx_3070': 'Mobile robot / Edge device',
            'rtx_4090': 'Research platform / Interactive AI',
            'workstation': 'Development / Training system',
            'datacenter': 'Production AI / Cloud service'
        }
        return deployment_types.get(hw_key, 'Unknown')
    
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive scaling analysis report."""
        print("\n🎯 Generating comprehensive scaling report...")
        print("=" * 80)
        
        # Run all analyses
        brain_characteristics = self.analyze_current_brain_characteristics()
        scaling_results = self.calculate_physical_memory_limits(brain_characteristics)
        computational_analysis = self.analyze_computational_scaling(brain_characteristics)
        biological_comparisons = self.compare_to_biological_intelligence(scaling_results)
        bottleneck_analysis = self.identify_scaling_bottlenecks(computational_analysis)
        deployment_scenarios = self.create_deployment_scenarios(scaling_results)
        
        # Compile comprehensive report
        report = {
            'analysis_timestamp': time.time(),
            'current_brain_characteristics': brain_characteristics,
            'hardware_scaling_results': scaling_results,
            'computational_analysis': computational_analysis,
            'biological_comparisons': biological_comparisons,
            'bottleneck_analysis': bottleneck_analysis,
            'deployment_scenarios': deployment_scenarios,
            'key_findings': self._extract_key_findings(
                brain_characteristics, scaling_results, bottleneck_analysis, deployment_scenarios, biological_comparisons
            )
        }
        
        # Print summary
        self._print_summary_report(report)
        
        return report
    
    def _extract_key_findings(self, brain_chars: Dict, scaling_results: Dict, 
                             bottlenecks: Dict, scenarios: Dict, biological_comparisons: Dict) -> Dict[str, Any]:
        """Extract key findings from all analyses."""
        
        # Find best hardware configurations
        best_memory = max(scaling_results.values(), key=lambda x: x.max_experiences_memory)
        best_realtime = max(scaling_results.values(), key=lambda x: x.max_experiences_realtime)
        
        # Find primary bottleneck
        realtime_limit = bottlenecks['transition_points'].get('realtime_limit', 'Unknown')
        
        return {
            'current_performance': {
                'memory_per_experience_bytes': brain_chars['memory_per_experience_bytes'],
                'compute_time_per_experience_ms': brain_chars['compute_time_per_experience_ms'],
                'memory_overhead_ratio': brain_chars['memory_overhead_ratio']
            },
            'best_configurations': {
                'memory_capacity': {
                    'hardware': best_memory.hardware_name,
                    'max_experiences': best_memory.max_experiences_memory
                },
                'realtime_performance': {
                    'hardware': best_realtime.hardware_name,
                    'max_experiences': best_realtime.max_experiences_realtime
                }
            },
            'scaling_limits': {
                'realtime_limit_experiences': realtime_limit,
                'primary_bottleneck': 'similarity_search',
                'memory_efficiency': f"{brain_chars['memory_overhead_ratio']:.1f}x overhead"
            },
            'biological_scale': {
                'datacenter_equivalent': biological_comparisons['datacenter']['memory_constrained']['organism_equivalent'],
                'human_brain_ratio': biological_comparisons['datacenter']['human_ratio']
            }
        }
    
    def _print_summary_report(self, report: Dict[str, Any]):
        """Print executive summary of scaling analysis."""
        print("\n" + "=" * 80)
        print("🧠 BRAIN SCALING ANALYSIS - EXECUTIVE SUMMARY")
        print("=" * 80)
        
        findings = report['key_findings']
        
        print("\n📊 CURRENT PERFORMANCE:")
        current = findings['current_performance']
        print(f"   Memory per experience: {current['memory_per_experience_bytes']:,} bytes")
        print(f"   Compute time per experience: {current['compute_time_per_experience_ms']:.3f}ms")
        print(f"   Memory overhead: {current['memory_overhead_ratio']:.1f}x raw vector storage")
        
        print("\n🏆 MAXIMUM BRAIN SIZES:")
        memory_best = findings['best_configurations']['memory_capacity']
        realtime_best = findings['best_configurations']['realtime_performance']
        print(f"   Largest brain (memory): {memory_best['max_experiences']:,} experiences ({memory_best['hardware']})")
        print(f"   Fastest brain (realtime): {realtime_best['max_experiences']:,} experiences ({realtime_best['hardware']})")
        
        print("\n⚡ SCALING BOTTLENECKS:")
        limits = findings['scaling_limits']
        print(f"   Real-time limit: {limits['realtime_limit_experiences']:,} experiences")
        print(f"   Primary bottleneck: {limits['primary_bottleneck']}")
        print(f"   Memory efficiency: {limits['memory_efficiency']}")
        
        print("\n🧬 BIOLOGICAL COMPARISON:")
        bio = findings['biological_scale']
        print(f"   Datacenter brain equivalent: {bio['datacenter_equivalent']}")
        print(f"   Ratio to human brain: {bio['human_brain_ratio']:.6f}")
        
        print("\n🚀 DEPLOYMENT RECOMMENDATIONS:")
        for hw_key, scenario in report['deployment_scenarios'].items():
            print(f"   {scenario['hardware']}: {scenario['capability_level']}")
            print(f"     → {scenario['max_experiences']:,} experiences, {scenario['limiting_factor']}-limited")
        
        print("\n🔧 OPTIMIZATION PRIORITIES:")
        recommendations = report['bottleneck_analysis']['scaling_recommendations']
        for i, rec in enumerate(recommendations[:3], 1):
            print(f"   {i}. {rec}")
        
        print("\n" + "=" * 80)


def main():
    """Run comprehensive brain scaling analysis."""
    print("🧠 Brain Scaling Analysis")
    print("Analyzing theoretical and practical limits of the minimal brain implementation")
    print("=" * 80)
    
    analyzer = BrainScalingAnalyzer()
    report = analyzer.generate_comprehensive_report()
    
    # Save report to logs
    import json
    os.makedirs('logs', exist_ok=True)
    
    timestamp = int(time.time())
    report_file = f'logs/brain_scaling_analysis_{timestamp}.json'
    
    # Convert numpy types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    # Serialize report
    serializable_report = {}
    for key, value in report.items():
        if key == 'hardware_scaling_results':
            # Handle ScalingResults dataclass
            serializable_report[key] = {
                hw_key: {
                    'hardware_name': result.hardware_name,
                    'max_experiences_memory': result.max_experiences_memory,
                    'max_experiences_realtime': result.max_experiences_realtime,
                    'memory_per_experience_bytes': result.memory_per_experience_bytes,
                    'compute_time_per_experience_ms': result.compute_time_per_experience_ms,
                    'working_memory_limit': result.working_memory_limit,
                    'biological_equivalent_neurons': result.biological_equivalent_neurons
                } for hw_key, result in value.items()
            }
        else:
            # Convert other values
            try:
                json.dumps(value, default=convert_numpy)
                serializable_report[key] = value
            except:
                serializable_report[key] = str(value)
    
    with open(report_file, 'w') as f:
        json.dump(serializable_report, f, indent=2, default=convert_numpy)
    
    print(f"📄 Detailed report saved to: {report_file}")
    print("Analysis complete!")


if __name__ == "__main__":
    main()