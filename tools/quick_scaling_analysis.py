"""
Quick Brain Scaling Analysis Tool

Performs lightweight analysis of brain scaling limits without running full brain operations.
Analyzes theoretical limits based on data structure sizes and computational complexity.

Run from project root:
python3 tools/quick_scaling_analysis.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import time
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass


@dataclass
class HardwareSpec:
    """Hardware specification for scaling analysis."""
    name: str
    ram_gb: int
    vram_gb: int
    gpu_compute_units: int  # Relative compute power (RTX 3070 = 100)
    bandwidth_gbps: float   # Memory bandwidth


@dataclass
class ExperienceMemoryProfile:
    """Memory profile for a single experience."""
    sensory_dims: int = 20
    action_dims: int = 4
    outcome_dims: int = 20
    similarity_connections: int = 5  # Average connections per experience
    
    def base_vector_bytes(self) -> int:
        """Raw vector storage (float64)."""
        return (self.sensory_dims + self.action_dims + self.outcome_dims) * 8
    
    def similarity_matrix_bytes(self, total_experiences: int) -> int:
        """Similarity matrix storage (sparse, average 5 connections per experience)."""
        return self.similarity_connections * (8 + 50)  # float + UUID overhead
    
    def metadata_bytes(self) -> int:
        """Experience metadata (timestamps, activation, utility, etc.)."""
        return 200  # Fixed overhead
    
    def total_memory_per_experience(self, total_experiences: int) -> int:
        """Total memory per experience including all overhead."""
        base = self.base_vector_bytes()
        similarity = self.similarity_matrix_bytes(total_experiences)
        metadata = self.metadata_bytes()
        
        # GPU tensor overhead (activation levels, similarity matrices)
        gpu_overhead = 50 if total_experiences > 100 else 20
        
        # System overhead (Python objects, indexing, etc.)
        system_overhead = int(base * 3.5)  # 3.5x overhead based on current implementation
        
        return base + similarity + metadata + gpu_overhead + system_overhead


class QuickScalingAnalyzer:
    """Quick scaling analysis without full brain instantiation."""
    
    def __init__(self):
        """Initialize the analyzer."""
        # Hardware configurations
        self.hardware_configs = {
            'rtx_3070': HardwareSpec('RTX 3070', 32, 24, 100, 896),
            'rtx_4090': HardwareSpec('RTX 4090', 64, 24, 183, 1008),
            'workstation': HardwareSpec('High-end Workstation', 128, 48, 250, 1500),
            'datacenter': HardwareSpec('Datacenter GPU Cluster', 512, 192, 1000, 5000)
        }
        
        # Current performance characteristics (from typical runs)
        self.baseline_performance = {
            'experiences_tested': 60,
            'compute_time_ms': 355,  # 355ms for 60 experiences from user spec
            'memory_per_experience_bytes': 12288,  # From actual measurement
            'memory_overhead_ratio': 34.9  # 34.9x overhead from actual measurement
        }
        
        # Biological comparisons
        self.human_neurons = 86_000_000_000
        self.human_synapses = 150_000_000_000_000
        
        # Performance targets
        self.realtime_target_ms = 50  # Real-time robotics
        self.interactive_target_ms = 100  # Interactive systems
        
        print("⚡ Quick Brain Scaling Analyzer initialized")
    
    def analyze_memory_scaling(self) -> Dict[str, Any]:
        """Analyze memory scaling for different brain sizes."""
        print("\n📊 Analyzing memory scaling patterns...")
        
        experience_profile = ExperienceMemoryProfile()
        scaling_points = [100, 500, 1000, 5000, 10000, 50000, 100000, 500000, 1000000]
        
        memory_scaling = {}
        
        for size in scaling_points:
            memory_per_exp = experience_profile.total_memory_per_experience(size)
            total_memory_mb = (memory_per_exp * size) / (1024 * 1024)
            
            # Working memory scales with total size but caps at practical limits
            working_memory_size = min(size // 10, 1000)  # 10% of experiences, max 1000
            working_memory_mb = (working_memory_size * memory_per_exp) / (1024 * 1024)
            
            memory_scaling[size] = {
                'memory_per_experience_bytes': memory_per_exp,
                'total_memory_mb': total_memory_mb,
                'total_memory_gb': total_memory_mb / 1024,
                'working_memory_size': working_memory_size,
                'working_memory_mb': working_memory_mb
            }
            
            print(f"  {size:,} experiences: {total_memory_mb:.1f}MB total, "
                  f"{working_memory_size:,} working memory")
        
        return memory_scaling
    
    def analyze_computational_complexity(self) -> Dict[str, Any]:
        """Analyze computational complexity scaling."""
        print("\n⚡ Analyzing computational complexity...")
        
        # Base performance: 355ms for 60 experiences = 5.9ms per experience
        base_time_per_exp = self.baseline_performance['compute_time_ms'] / self.baseline_performance['experiences_tested']
        
        scaling_points = [100, 500, 1000, 5000, 10000, 50000, 100000]
        computational_scaling = {}
        
        for size in scaling_points:
            # Similarity search: O(n) with large constant (dominates performance)
            # Currently ~70% of computation time
            similarity_time = base_time_per_exp * 0.7 * (size / 60)
            
            # Activation dynamics: O(working_memory_size)
            working_memory_size = min(size // 10, 1000)
            activation_time = base_time_per_exp * 0.2 * (working_memory_size / 10)
            
            # Prediction engine: O(working_memory_size)
            prediction_time = base_time_per_exp * 0.1 * (working_memory_size / 10)
            
            total_time = similarity_time + activation_time + prediction_time
            
            computational_scaling[size] = {
                'total_time_ms': total_time,
                'similarity_time_ms': similarity_time,
                'activation_time_ms': activation_time,
                'prediction_time_ms': prediction_time,
                'working_memory_size': working_memory_size,
                'time_per_experience_ms': total_time,
                'realtime_feasible': total_time < self.realtime_target_ms,
                'interactive_feasible': total_time < self.interactive_target_ms
            }
        
        # Find transition points
        realtime_limit = None
        interactive_limit = None
        
        for size in sorted(scaling_points):
            data = computational_scaling[size]
            if realtime_limit is None and not data['realtime_feasible']:
                realtime_limit = size
            if interactive_limit is None and not data['interactive_feasible']:
                interactive_limit = size
        
        return {
            'scaling_data': computational_scaling,
            'realtime_limit_experiences': realtime_limit or 'Beyond analyzed range',
            'interactive_limit_experiences': interactive_limit or 'Beyond analyzed range',
            'base_time_per_experience_ms': base_time_per_exp
        }
    
    def calculate_hardware_limits(self, memory_scaling: Dict, computational_scaling: Dict) -> Dict[str, Any]:
        """Calculate brain size limits for each hardware configuration."""
        print("\n🖥️  Calculating hardware-specific limits...")
        
        hardware_limits = {}
        
        for hw_key, hw_spec in self.hardware_configs.items():
            print(f"\nAnalyzing {hw_spec.name}:")
            
            # Memory-constrained analysis
            total_memory_gb = (hw_spec.ram_gb + hw_spec.vram_gb) * 0.8  # 80% utilization
            max_experiences_memory = None
            
            for size, data in memory_scaling.items():
                if data['total_memory_gb'] <= total_memory_gb:
                    max_experiences_memory = size
                else:
                    break
            
            # Compute-constrained analysis
            # Scale by relative GPU performance
            performance_factor = hw_spec.gpu_compute_units / 100.0  # RTX 3070 baseline
            
            max_experiences_realtime = None
            max_experiences_interactive = None
            
            base_time = computational_scaling['base_time_per_experience_ms']
            scaled_realtime_target = self.realtime_target_ms * performance_factor
            scaled_interactive_target = self.interactive_target_ms * performance_factor
            
            for size, data in computational_scaling['scaling_data'].items():
                adjusted_time = data['total_time_ms'] / performance_factor
                
                if adjusted_time <= scaled_realtime_target:
                    max_experiences_realtime = size
                if adjusted_time <= scaled_interactive_target:
                    max_experiences_interactive = size
            
            # Practical limits (minimum of memory and compute constraints)
            practical_realtime = min(max_experiences_memory or 0, max_experiences_realtime or 0)
            practical_interactive = min(max_experiences_memory or 0, max_experiences_interactive or 0)
            
            # Biological equivalences (rough approximation)
            # 1 experience ≈ 1000 synaptic connections
            memory_synapses = (max_experiences_memory or 0) * 1000
            realtime_synapses = practical_realtime * 1000
            
            hardware_limits[hw_key] = {
                'hardware_name': hw_spec.name,
                'memory_constraint': {
                    'max_experiences': max_experiences_memory,
                    'total_memory_gb': total_memory_gb,
                    'biological_synapses': memory_synapses,
                    'biological_equivalent': self._get_organism_equivalent(memory_synapses)
                },
                'realtime_constraint': {
                    'max_experiences': max_experiences_realtime,
                    'performance_factor': performance_factor,
                    'biological_synapses': realtime_synapses,
                    'biological_equivalent': self._get_organism_equivalent(realtime_synapses)
                },
                'practical_limits': {
                    'realtime_experiences': practical_realtime,
                    'interactive_experiences': practical_interactive,
                    'limiting_factor': 'memory' if max_experiences_memory and max_experiences_memory < (max_experiences_realtime or float('inf')) else 'compute'
                }
            }
            
            print(f"  Memory limit: {max_experiences_memory:,} experiences")
            print(f"  Real-time limit: {max_experiences_realtime or 'None':,} experiences")
            print(f"  Practical real-time: {practical_realtime:,} experiences")
            print(f"  Limiting factor: {hardware_limits[hw_key]['practical_limits']['limiting_factor']}")
        
        return hardware_limits
    
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
    
    def identify_scaling_bottlenecks(self, computational_scaling: Dict, hardware_limits: Dict) -> Dict[str, Any]:
        """Identify primary scaling bottlenecks."""
        print("\n🚧 Identifying scaling bottlenecks...")
        
        bottlenecks = {
            'similarity_search': {
                'complexity': 'O(n) linear scan',
                'description': 'Similarity search dominates at large scales (70% of compute time)',
                'becomes_critical': 'Around 10,000 experiences',
                'mitigation': 'Approximate Nearest Neighbor (ANN) indexing, hierarchical search'
            },
            'memory_overhead': {
                'complexity': '34.9x vector storage overhead',
                'description': 'Memory overhead from Python objects, GPU tensors, indexing',
                'becomes_critical': 'Around 100,000 experiences (several GB)',
                'mitigation': 'Memory-mapped storage, compressed representations, batch processing'
            },
            'connection_matrix': {
                'complexity': 'O(n²) worst case, O(n) average',
                'description': 'Similarity connections between experiences',
                'becomes_critical': 'Around 50,000 experiences',
                'mitigation': 'Sparse matrices, connection pruning, hierarchical clustering'
            },
            'working_memory': {
                'complexity': 'O(working_memory_size)',
                'description': 'Activation dynamics and prediction consensus',
                'becomes_critical': 'Working memory > 1,000 experiences',
                'mitigation': 'Attention mechanisms, relevance filtering, sampling'
            }
        }
        
        # Find when real-time operation becomes impossible
        realtime_limit = computational_scaling.get('realtime_limit_experiences', 'Unknown')
        
        # Identify primary bottleneck for each hardware config
        primary_bottlenecks = {}
        for hw_key, limits in hardware_limits.items():
            limiting_factor = limits['practical_limits']['limiting_factor']
            realtime_exp = limits['practical_limits']['realtime_experiences']
            
            if realtime_exp < 1000:
                primary = 'similarity_search'
            elif limiting_factor == 'memory':
                primary = 'memory_overhead'
            else:
                primary = 'similarity_search'
            
            primary_bottlenecks[hw_key] = {
                'primary_bottleneck': primary,
                'limiting_factor': limiting_factor,
                'realtime_experiences': realtime_exp
            }
        
        return {
            'bottleneck_descriptions': bottlenecks,
            'hardware_bottlenecks': primary_bottlenecks,
            'overall_realtime_limit': realtime_limit
        }
    
    def create_deployment_scenarios(self, hardware_limits: Dict) -> Dict[str, Any]:
        """Create deployment scenarios for different hardware configurations."""
        print("\n🚀 Creating deployment scenarios...")
        
        scenarios = {}
        
        for hw_key, limits in hardware_limits.items():
            realtime_exp = limits['practical_limits']['realtime_experiences']
            interactive_exp = limits['practical_limits']['interactive_experiences']
            
            # Determine capability level based on experience count
            if realtime_exp < 500:
                capability = "Basic reactive behaviors"
                applications = ["Obstacle avoidance", "Simple pattern recognition", "Basic motor control"]
            elif realtime_exp < 2000:
                capability = "Structured learning behaviors"
                applications = ["Spatial navigation", "Motor skill acquisition", "Simple planning"]
            elif realtime_exp < 10000:
                capability = "Complex adaptive behaviors"
                applications = ["Multi-task learning", "Social interaction", "Tool use", "Exploration"]
            elif realtime_exp < 50000:
                capability = "Sophisticated intelligence"
                applications = ["Language understanding", "Abstract reasoning", "Creative problem solving"]
            else:
                capability = "Advanced cognitive capacity"
                applications = ["Research assistance", "Complex reasoning", "Scientific discovery"]
            
            scenarios[hw_key] = {
                'hardware': limits['hardware_name'],
                'realtime_experiences': realtime_exp,
                'interactive_experiences': interactive_exp,
                'capability_level': capability,
                'applications': applications,
                'deployment_type': self._get_deployment_type(hw_key),
                'limiting_factor': limits['practical_limits']['limiting_factor'],
                'biological_equivalent': limits['realtime_constraint']['biological_equivalent']
            }
            
            print(f"\n{limits['hardware_name']}:")
            print(f"  Real-time: {realtime_exp:,} experiences")
            print(f"  Capability: {capability}")
            print(f"  Bio equivalent: {scenarios[hw_key]['biological_equivalent']}")
        
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
        print("🧠 Quick Brain Scaling Analysis")
        print("=" * 80)
        
        # Run analyses
        memory_scaling = self.analyze_memory_scaling()
        computational_scaling = self.analyze_computational_complexity()
        hardware_limits = self.calculate_hardware_limits(memory_scaling, computational_scaling)
        bottleneck_analysis = self.identify_scaling_bottlenecks(computational_scaling, hardware_limits)
        deployment_scenarios = self.create_deployment_scenarios(hardware_limits)
        
        # Compile report
        report = {
            'analysis_timestamp': time.time(),
            'baseline_performance': self.baseline_performance,
            'memory_scaling': memory_scaling,
            'computational_scaling': computational_scaling,
            'hardware_limits': hardware_limits,
            'bottleneck_analysis': bottleneck_analysis,
            'deployment_scenarios': deployment_scenarios
        }
        
        # Print summary
        self._print_executive_summary(report)
        
        return report
    
    def _print_executive_summary(self, report: Dict[str, Any]):
        """Print executive summary of the analysis."""
        print("\n" + "=" * 80)
        print("📊 EXECUTIVE SUMMARY")
        print("=" * 80)
        
        baseline = report['baseline_performance']
        print(f"\n🎯 CURRENT PERFORMANCE BASELINE:")
        print(f"   Time per experience: {baseline['compute_time_ms'] / baseline['experiences_tested']:.1f}ms")
        print(f"   Memory per experience: {baseline['memory_per_experience_bytes']:,} bytes")
        print(f"   Memory overhead: {baseline['memory_overhead_ratio']:.1f}x raw vectors")
        
        # Find best hardware for different use cases
        hw_limits = report['hardware_limits']
        best_realtime = max(hw_limits.items(), key=lambda x: x[1]['practical_limits']['realtime_experiences'])
        best_memory = max(hw_limits.items(), key=lambda x: x[1]['memory_constraint']['max_experiences'] or 0)
        
        print(f"\n🏆 MAXIMUM BRAIN SCALES:")
        print(f"   Best real-time: {best_realtime[1]['practical_limits']['realtime_experiences']:,} experiences ({best_realtime[1]['hardware_name']})")
        print(f"   Best memory capacity: {best_memory[1]['memory_constraint']['max_experiences']:,} experiences ({best_memory[1]['hardware_name']})")
        
        # Bottlenecks
        bottlenecks = report['bottleneck_analysis']
        realtime_limit = bottlenecks['overall_realtime_limit']
        print(f"\n⚡ SCALING BOTTLENECKS:")
        print(f"   Real-time limit: {realtime_limit} experiences")
        print(f"   Primary bottleneck: Similarity search (O(n) linear scan)")
        print(f"   Memory overhead: {baseline['memory_overhead_ratio']:.1f}x raw data")
        
        # Deployment scenarios
        scenarios = report['deployment_scenarios']
        print(f"\n🚀 DEPLOYMENT SCENARIOS:")
        for hw_key, scenario in scenarios.items():
            print(f"   {scenario['hardware']}: {scenario['realtime_experiences']:,} experiences")
            print(f"     → {scenario['capability_level']}")
            print(f"     → {scenario['biological_equivalent']}")
        
        print(f"\n🔧 OPTIMIZATION PRIORITIES:")
        print(f"   1. Implement Approximate Nearest Neighbor (ANN) for similarity search")
        print(f"   2. Reduce memory overhead (currently 34.9x raw vectors)")
        print(f"   3. Optimize GPU tensor management and mixed precision")
        print(f"   4. Implement hierarchical/sparse activation patterns")
        
        print("\n" + "=" * 80)


def main():
    """Run quick scaling analysis."""
    analyzer = QuickScalingAnalyzer()
    report = analyzer.generate_comprehensive_report()
    
    # Save report
    import json
    os.makedirs('logs', exist_ok=True)
    
    timestamp = int(time.time())
    report_file = f'logs/quick_scaling_analysis_{timestamp}.json'
    
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\n📄 Detailed report saved to: {report_file}")


if __name__ == "__main__":
    main()