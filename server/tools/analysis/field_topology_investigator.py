#!/usr/bin/env python3
"""
Field Topology Investigator

Examines the 36-dimensional field topology during pattern learning to detect
emergent knowledge clustering at the field dynamics level.

This investigates whether different input patterns create distinct field topology
regions, providing natural context separation and knowledge clustering.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import time
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import json

# Add brain server to path
brain_server_path = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, str(brain_server_path))

from src.brain_factory import BrainFactory


@dataclass
class FieldTopologySnapshot:
    """Snapshot of field topology state"""
    timestamp: float
    pattern_type: str
    cycle: int
    
    # Topology dimensions (indices 19-24 in 36D field)
    topo_stable_1: float
    topo_stable_2: float
    topo_boundary: float
    topo_persistence: float
    topo_connectivity: float
    topo_homology: float
    
    # Flow dimensions (indices 11-18 in 36D field)
    flow_grad_x: float
    flow_grad_y: float
    flow_grad_z: float
    flow_momentum_1: float
    flow_momentum_2: float
    flow_divergence: float
    flow_curl: float
    flow_potential: float
    
    # Coupling dimensions (indices 29-33 in 36D field)
    coupling_correlation: float
    coupling_binding: float
    coupling_synchrony: float
    coupling_resonance: float
    coupling_interference: float
    
    # Field summary statistics
    field_energy: float
    field_variance: float
    field_max_activation: float
    field_topology_signature: float


class FieldTopologyInvestigator:
    """Investigates field topology dynamics during pattern learning"""
    
    def __init__(self):
        self.snapshots: List[FieldTopologySnapshot] = []
        
    def extract_field_topology_snapshot(self, brain: BrainFactory, pattern_type: str, cycle: int) -> FieldTopologySnapshot:
        """Extract a detailed snapshot of the field topology state from unified 37D field"""
        
        # Access the field brain through the brain factory
        field_brain = brain.field_brain_adapter.field_brain
        field_impl = field_brain.field_impl
        
        # Get the unified field tensor (37D structure: [20,20,20,10,15,1,1,1...])
        unified_field = field_impl.unified_field
        
        # Debug: Print actual field shape
        print(f"   Field shape: {unified_field.shape}")
        
        # Calculate field summary statistics from the entire field
        field_tensor = unified_field.detach().cpu().numpy()
        field_energy = float(np.sum(field_tensor ** 2))
        field_variance = float(np.var(field_tensor))
        field_max_activation = float(np.max(np.abs(field_tensor)))
        
        # UNIFIED FIELD STRUCTURE ANALYSIS
        # Based on the field shape [20,20,20,10,15,1,1,1...], extract physics family regions
        shape = unified_field.shape
        
        try:
            # TOPOLOGY DIMENSIONS: Extract from spatial regions representing stable configurations
            # Use different spatial locations to represent topology family behaviors
            center_x, center_y, center_z = shape[0]//2, shape[1]//2, shape[2]//2
            
            # Topology stable configurations - sample from different spatial-temporal regions
            topo_stable_1 = float(unified_field[center_x, center_y, center_z, 2, 3].item())
            topo_stable_2 = float(unified_field[center_x, center_y, center_z, 4, 5].item())
            topo_boundary = float(unified_field[0, 0, 0, 1, 2].item())  # Boundary regions
            topo_persistence = float(unified_field[-1, -1, -1, 2, 4].item())  # Edge persistence
            topo_connectivity = float(unified_field[shape[0]//3, shape[1]//3, shape[2]//3, 3, 6].item())
            topo_homology = float(unified_field[2*shape[0]//3, 2*shape[1]//3, 2*shape[2]//3, 5, 7].item())
            
            # FLOW DIMENSIONS: Extract from gradient and momentum analysis
            # Analyze spatial gradients across the 3D spatial dimensions
            spatial_field = unified_field[:,:,:,0,0].detach().cpu().numpy()  # Base spatial layer
            flow_grad_x = float(np.mean(np.gradient(spatial_field, axis=0)))
            flow_grad_y = float(np.mean(np.gradient(spatial_field, axis=1)))
            flow_grad_z = float(np.mean(np.gradient(spatial_field, axis=2)))
            
            # Flow momentum from temporal-spatial correlations
            temporal_field_1 = unified_field[:,:,:,1,0].detach().cpu().numpy()
            temporal_field_2 = unified_field[:,:,:,2,0].detach().cpu().numpy()
            flow_momentum_1 = float(np.corrcoef(spatial_field.flatten(), temporal_field_1.flatten())[0,1])
            flow_momentum_2 = float(np.corrcoef(spatial_field.flatten(), temporal_field_2.flatten())[0,1])
            flow_divergence = flow_grad_x + flow_grad_y + flow_grad_z
            flow_curl = flow_grad_y - flow_grad_x  # Simplified 2D curl
            flow_potential = float(np.mean(spatial_field))
            
            # COUPLING DIMENSIONS: Cross-dimensional correlations and binding
            # Analyze correlations between different field regions
            region_1 = unified_field[center_x, center_y, :, :3, :3].flatten().detach().cpu().numpy()
            region_2 = unified_field[center_x, center_y, :, 3:6, :3].flatten().detach().cpu().numpy()
            region_3 = unified_field[center_x, center_y, :, 6:9, :3].flatten().detach().cpu().numpy()
            
            # Cross-regional correlations (coupling)
            try:
                coupling_correlation = float(np.corrcoef(region_1, region_2)[0,1])
            except:
                coupling_correlation = 0.0
            
            try:
                coupling_binding = float(np.corrcoef(region_2, region_3)[0,1])
            except:
                coupling_binding = 0.0
            
            # Synchronization across field regions
            coupling_synchrony = float(np.std([np.mean(region_1), np.mean(region_2), np.mean(region_3)]))
            
            # Resonance from frequency domain analysis
            fft_region_1 = np.abs(np.fft.fft(region_1))[:len(region_1)//2]
            coupling_resonance = float(np.mean(fft_region_1))
            
            # Interference patterns
            coupling_interference = float(np.var([np.var(region_1), np.var(region_2), np.var(region_3)]))
            
            # Handle NaN values
            for var_name in ['flow_momentum_1', 'flow_momentum_2', 'coupling_correlation', 'coupling_binding']:
                var_val = locals()[var_name]
                if np.isnan(var_val):
                    locals()[var_name] = 0.0
            
        except Exception as e:
            print(f"   Warning: Error extracting unified field measures: {e}")
            # Fallback to simple field statistics
            topo_stable_1 = topo_stable_2 = topo_boundary = field_energy / 1000.0
            topo_persistence = topo_connectivity = topo_homology = field_variance
            flow_grad_x = flow_grad_y = flow_grad_z = field_max_activation / 10.0
            flow_momentum_1 = flow_momentum_2 = flow_divergence = field_energy / 1000.0
            flow_curl = flow_potential = field_variance
            coupling_correlation = coupling_binding = coupling_synchrony = field_variance
            coupling_resonance = coupling_interference = field_max_activation / 10.0
        
        # Calculate topology signature (combination of stable configurations)
        field_topology_signature = float(topo_stable_1 * topo_stable_2 + topo_boundary * topo_persistence)
        
        return FieldTopologySnapshot(
            timestamp=time.time(),
            pattern_type=pattern_type,
            cycle=cycle,
            
            topo_stable_1=topo_stable_1,
            topo_stable_2=topo_stable_2,
            topo_boundary=topo_boundary,
            topo_persistence=topo_persistence,
            topo_connectivity=topo_connectivity,
            topo_homology=topo_homology,
            
            flow_grad_x=flow_grad_x,
            flow_grad_y=flow_grad_y,
            flow_grad_z=flow_grad_z,
            flow_momentum_1=flow_momentum_1,
            flow_momentum_2=flow_momentum_2,
            flow_divergence=flow_divergence,
            flow_curl=flow_curl,
            flow_potential=flow_potential,
            
            coupling_correlation=coupling_correlation,
            coupling_binding=coupling_binding,
            coupling_synchrony=coupling_synchrony,
            coupling_resonance=coupling_resonance,
            coupling_interference=coupling_interference,
            
            field_energy=field_energy,
            field_variance=field_variance,
            field_max_activation=field_max_activation,
            field_topology_signature=field_topology_signature
        )
    
    def investigate_pattern_topology_divergence(self, cycles: int = 50) -> Dict[str, Any]:
        """
        Investigate whether different patterns create distinct field topologies.
        
        Tests similar vs divergent patterns to see if field topology naturally
        clusters different contexts into separate regions.
        """
        print("🔬 Field Topology Investigation")
        print("Examining 36D field topology during pattern learning...")
        print("=" * 60)
        
        # Clear any existing robot memory for fresh brains
        if os.path.exists('robot_memory'):
            import shutil
            shutil.rmtree('robot_memory')
            print("🗑️ Cleared robot memory for fresh field topology")
        
        results = {
            'similar_pattern_snapshots': [],
            'divergent_pattern_snapshots': [],
            'topology_analysis': {},
            'clustering_evidence': {}
        }
        
        # Test 1: Similar pattern topology evolution
        print(f"\n📊 Phase 1: Similar Pattern Field Topology")
        brain1 = BrainFactory(quiet_mode=True)
        similar_pattern = [0.5, 0.3, 0.8, 0.2, 0.6, 0.1, 0.9, 0.4] * 2  # 16D sensory input
        
        for cycle in range(cycles):
            # Present pattern and capture field state
            action, brain_state = brain1.process_sensory_input(similar_pattern)
            
            # Extract topology snapshot every 10 cycles
            if cycle % 10 == 0:
                snapshot = self.extract_field_topology_snapshot(brain1, "similar", cycle)
                results['similar_pattern_snapshots'].append(snapshot)
                self.snapshots.append(snapshot)
                
                if cycle % 20 == 0:
                    print(f"   Cycle {cycle}: Topology signature = {snapshot.field_topology_signature:.6f}, "
                          f"Energy = {snapshot.field_energy:.1f}")
        
        brain1.finalize_session()
        
        # Test 2: Divergent pattern topology evolution  
        print(f"\n📊 Phase 2: Divergent Pattern Field Topology")
        brain2 = BrainFactory(quiet_mode=True)
        divergent_pattern = [0.01, 0.99, 0.02, 0.98, 0.03, 0.97, 0.04, 0.96] * 2  # High contrast
        
        for cycle in range(cycles):
            # Present pattern and capture field state
            action, brain_state = brain2.process_sensory_input(divergent_pattern)
            
            # Extract topology snapshot every 10 cycles
            if cycle % 10 == 0:
                snapshot = self.extract_field_topology_snapshot(brain2, "divergent", cycle)
                results['divergent_pattern_snapshots'].append(snapshot)
                self.snapshots.append(snapshot)
                
                if cycle % 20 == 0:
                    print(f"   Cycle {cycle}: Topology signature = {snapshot.field_topology_signature:.6f}, "
                          f"Energy = {snapshot.field_energy:.1f}")
        
        brain2.finalize_session()
        
        # Analyze topology clustering
        results['topology_analysis'] = self.analyze_topology_clustering(results)
        results['clustering_evidence'] = self.detect_clustering_evidence(results)
        
        return results
    
    def analyze_topology_clustering(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze whether topology dimensions show clustering behavior"""
        
        similar_snapshots = results['similar_pattern_snapshots']
        divergent_snapshots = results['divergent_pattern_snapshots']
        
        if not similar_snapshots or not divergent_snapshots:
            return {'error': 'Insufficient snapshots for analysis'}
        
        analysis = {}
        
        # Topology dimension separation analysis
        topology_dimensions = [
            'topo_stable_1', 'topo_stable_2', 'topo_boundary', 
            'topo_persistence', 'topo_connectivity', 'topo_homology'
        ]
        
        for dim in topology_dimensions:
            similar_values = [getattr(s, dim) for s in similar_snapshots]
            divergent_values = [getattr(s, dim) for s in divergent_snapshots]
            
            similar_mean = np.mean(similar_values)
            divergent_mean = np.mean(divergent_values)
            similar_var = np.var(similar_values)
            divergent_var = np.var(divergent_values)
            
            # Calculate separation metric (distance between means relative to variance)
            combined_std = np.sqrt((similar_var + divergent_var) / 2)
            separation = abs(similar_mean - divergent_mean) / (combined_std + 1e-8)
            
            analysis[dim] = {
                'similar_mean': similar_mean,
                'divergent_mean': divergent_mean,
                'separation_score': separation,
                'clustering_detected': bool(separation > 1.0)  # Strong clustering if >1 std separation
            }
        
        # Flow dimension separation analysis
        flow_dimensions = [
            'flow_grad_x', 'flow_grad_y', 'flow_grad_z', 'flow_momentum_1', 
            'flow_momentum_2', 'flow_divergence', 'flow_curl', 'flow_potential'
        ]
        
        for dim in flow_dimensions:
            similar_values = [getattr(s, dim) for s in similar_snapshots]
            divergent_values = [getattr(s, dim) for s in divergent_snapshots]
            
            similar_mean = np.mean(similar_values)
            divergent_mean = np.mean(divergent_values)
            combined_std = np.sqrt((np.var(similar_values) + np.var(divergent_values)) / 2)
            separation = abs(similar_mean - divergent_mean) / (combined_std + 1e-8)
            
            analysis[dim] = {
                'similar_mean': similar_mean,
                'divergent_mean': divergent_mean,
                'separation_score': separation,
                'clustering_detected': bool(separation > 1.0)
            }
        
        return analysis
    
    def detect_clustering_evidence(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Detect evidence of emergent knowledge clustering"""
        
        analysis = results.get('topology_analysis', {})
        if 'error' in analysis:
            return {'error': analysis['error']}
        
        evidence = {
            'strong_clustering_dimensions': [],
            'moderate_clustering_dimensions': [],
            'no_clustering_dimensions': [],
            'overall_clustering_score': 0.0,
            'emergent_clustering_detected': False
        }
        
        separation_scores = []
        
        for dim_name, dim_analysis in analysis.items():
            if isinstance(dim_analysis, dict) and 'separation_score' in dim_analysis:
                separation = dim_analysis['separation_score']
                separation_scores.append(separation)
                
                if separation > 2.0:
                    evidence['strong_clustering_dimensions'].append({
                        'dimension': dim_name,
                        'separation': separation,
                        'similar_mean': dim_analysis['similar_mean'],
                        'divergent_mean': dim_analysis['divergent_mean']
                    })
                elif separation > 1.0:
                    evidence['moderate_clustering_dimensions'].append({
                        'dimension': dim_name,
                        'separation': separation
                    })
                else:
                    evidence['no_clustering_dimensions'].append(dim_name)
        
        # Overall clustering score
        if separation_scores:
            evidence['overall_clustering_score'] = float(np.mean(separation_scores))
            evidence['emergent_clustering_detected'] = bool(evidence['overall_clustering_score'] > 1.0)
        
        return evidence
    
    def generate_topology_report(self, results: Dict[str, Any]) -> str:
        """Generate a comprehensive topology investigation report"""
        
        clustering_evidence = results.get('clustering_evidence', {})
        topology_analysis = results.get('topology_analysis', {})
        
        if 'error' in clustering_evidence:
            return f"❌ Analysis failed: {clustering_evidence['error']}"
        
        report = []
        report.append("🔬 FIELD TOPOLOGY INVESTIGATION REPORT")
        report.append("=" * 60)
        
        # Overall clustering assessment
        overall_score = clustering_evidence.get('overall_clustering_score', 0.0)
        clustering_detected = clustering_evidence.get('emergent_clustering_detected', False)
        
        if clustering_detected:
            report.append(f"✅ EMERGENT CLUSTERING DETECTED!")
            report.append(f"   Overall clustering score: {overall_score:.3f}")
        else:
            report.append(f"❌ NO CLEAR CLUSTERING DETECTED")
            report.append(f"   Overall clustering score: {overall_score:.3f}")
        
        # Strong clustering dimensions
        strong_dims = clustering_evidence.get('strong_clustering_dimensions', [])
        if strong_dims:
            report.append(f"\n🎯 STRONG CLUSTERING DIMENSIONS ({len(strong_dims)}):")
            for dim_info in strong_dims:
                report.append(f"   {dim_info['dimension']}: {dim_info['separation']:.3f} separation")
                report.append(f"      Similar: {dim_info['similar_mean']:.6f}, Divergent: {dim_info['divergent_mean']:.6f}")
        
        # Moderate clustering dimensions
        moderate_dims = clustering_evidence.get('moderate_clustering_dimensions', [])
        if moderate_dims:
            report.append(f"\n⚠️ MODERATE CLUSTERING DIMENSIONS ({len(moderate_dims)}):")
            for dim_info in moderate_dims:
                report.append(f"   {dim_info['dimension']}: {dim_info['separation']:.3f} separation")
        
        # No clustering dimensions
        no_clustering_dims = clustering_evidence.get('no_clustering_dimensions', [])
        if no_clustering_dims:
            report.append(f"\n❌ NO CLUSTERING DIMENSIONS ({len(no_clustering_dims)}):")
            report.append(f"   {', '.join(no_clustering_dims[:5])}{'...' if len(no_clustering_dims) > 5 else ''}")
        
        return "\n".join(report)
    
    def save_investigation_results(self, results: Dict[str, Any], filename: str = None) -> str:
        """Save investigation results to JSON file"""
        
        if filename is None:
            timestamp = int(time.time())
            filename = f"field_topology_investigation_{timestamp}.json"
        
        # Convert snapshots to serializable format
        serializable_results = {
            'timestamp': time.time(),
            'similar_pattern_snapshots': [
                {
                    'timestamp': s.timestamp,
                    'pattern_type': s.pattern_type,
                    'cycle': s.cycle,
                    'topology_dimensions': {
                        'topo_stable_1': s.topo_stable_1,
                        'topo_stable_2': s.topo_stable_2,
                        'topo_boundary': s.topo_boundary,
                        'topo_persistence': s.topo_persistence,
                        'topo_connectivity': s.topo_connectivity,
                        'topo_homology': s.topo_homology,
                    },
                    'flow_dimensions': {
                        'flow_grad_x': s.flow_grad_x,
                        'flow_grad_y': s.flow_grad_y,
                        'flow_grad_z': s.flow_grad_z,
                        'flow_momentum_1': s.flow_momentum_1,
                        'flow_momentum_2': s.flow_momentum_2,
                        'flow_divergence': s.flow_divergence,
                        'flow_curl': s.flow_curl,
                        'flow_potential': s.flow_potential,
                    },
                    'coupling_dimensions': {
                        'coupling_correlation': s.coupling_correlation,
                        'coupling_binding': s.coupling_binding,
                        'coupling_synchrony': s.coupling_synchrony,
                        'coupling_resonance': s.coupling_resonance,
                        'coupling_interference': s.coupling_interference,
                    },
                    'field_summary': {
                        'field_energy': s.field_energy,
                        'field_variance': s.field_variance,
                        'field_max_activation': s.field_max_activation,
                        'field_topology_signature': s.field_topology_signature,
                    }
                }
                for s in results['similar_pattern_snapshots']
            ],
            'divergent_pattern_snapshots': [
                {
                    'timestamp': s.timestamp,
                    'pattern_type': s.pattern_type,
                    'cycle': s.cycle,
                    'topology_dimensions': {
                        'topo_stable_1': s.topo_stable_1,
                        'topo_stable_2': s.topo_stable_2,
                        'topo_boundary': s.topo_boundary,
                        'topo_persistence': s.topo_persistence,
                        'topo_connectivity': s.topo_connectivity,
                        'topo_homology': s.topo_homology,
                    },
                    'flow_dimensions': {
                        'flow_grad_x': s.flow_grad_x,
                        'flow_grad_y': s.flow_grad_y,
                        'flow_grad_z': s.flow_grad_z,
                        'flow_momentum_1': s.flow_momentum_1,
                        'flow_momentum_2': s.flow_momentum_2,
                        'flow_divergence': s.flow_divergence,
                        'flow_curl': s.flow_curl,
                        'flow_potential': s.flow_potential,
                    },
                    'coupling_dimensions': {
                        'coupling_correlation': s.coupling_correlation,
                        'coupling_binding': s.coupling_binding,
                        'coupling_synchrony': s.coupling_synchrony,
                        'coupling_resonance': s.coupling_resonance,
                        'coupling_interference': s.coupling_interference,
                    },
                    'field_summary': {
                        'field_energy': s.field_energy,
                        'field_variance': s.field_variance,
                        'field_max_activation': s.field_max_activation,
                        'field_topology_signature': s.field_topology_signature,
                    }
                }
                for s in results['divergent_pattern_snapshots']
            ],
            'topology_analysis': results.get('topology_analysis', {}),
            'clustering_evidence': results.get('clustering_evidence', {})
        }
        
        filepath = os.path.join(os.path.dirname(__file__), filename)
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        return filepath


def main():
    """Run the field topology investigation"""
    investigator = FieldTopologyInvestigator()
    
    # Run investigation
    results = investigator.investigate_pattern_topology_divergence(cycles=50)
    
    # Generate and display report
    report = investigator.generate_topology_report(results)
    print(f"\n{report}")
    
    # Save detailed results
    filepath = investigator.save_investigation_results(results)
    print(f"\n💾 Detailed results saved to: {filepath}")
    
    return results


if __name__ == "__main__":
    main()