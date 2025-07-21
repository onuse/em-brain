#!/usr/bin/env python3
"""
Field Dynamics Diagnostics

Phase 1 diagnostic tool to understand why field energy stays at 0.0 
and what's happening with the Enhanced Field Dynamics.

This will trace the field evolution process step-by-step.
"""

import sys
import os
import numpy as np
import time
from typing import Dict, List, Tuple, Any, Optional

# Add brain server to path
brain_server_path = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, str(brain_server_path))

from src.brain_factory import BrainFactory


class FieldDynamicsDiagnostics:
    """Diagnose field dynamics behavior and energy evolution"""
    
    def __init__(self):
        pass
    
    def diagnose_field_dynamics(self, cycles: int = 20) -> Dict[str, Any]:
        """
        Diagnose field dynamics evolution step by step.
        
        Traces:
        1. Field brain initialization
        2. Enhanced dynamics activation  
        3. Field evolution during processing
        4. Energy evolution over cycles
        5. Field state changes
        """
        print("🔬 FIELD DYNAMICS DIAGNOSTICS")
        print("Tracing field evolution and energy dynamics...")
        print("=" * 60)
        
        # Clear memory for clean test
        if os.path.exists('robot_memory'):
            import shutil
            shutil.rmtree('robot_memory')
            print("🗑️ Cleared robot memory for clean diagnostics")
        
        results = {
            'initialization': {},
            'field_evolution': [],
            'energy_evolution': [],
            'dynamics_activity': [],
            'issues_detected': []
        }
        
        # Create brain with field configuration
        print(f"\n📊 Creating Field Brain...")
        brain = BrainFactory(quiet_mode=False)  # Verbose for diagnostics
        
        # Get references to field components
        field_brain = brain.field_brain_adapter.field_brain
        field_impl = field_brain.field_impl
        enhanced_dynamics = getattr(field_brain, 'enhanced_dynamics', None)
        
        # Log initialization details
        results['initialization'] = {
            'field_brain_type': type(field_brain).__name__,
            'field_impl_type': type(field_impl).__name__,
            'enhanced_dynamics_enabled': enhanced_dynamics is not None,
            'enhanced_dynamics_type': type(enhanced_dynamics).__name__ if enhanced_dynamics else None,
            'field_device': str(field_impl.field_device) if hasattr(field_impl, 'field_device') else 'unknown',
            'spatial_resolution': getattr(field_brain, 'spatial_resolution', 'unknown')
        }
        
        print(f"   Field brain: {results['initialization']['field_brain_type']}")
        print(f"   Field implementation: {results['initialization']['field_impl_type']}")  
        print(f"   Enhanced dynamics: {'✅ Enabled' if enhanced_dynamics else '❌ Disabled'}")
        if enhanced_dynamics:
            print(f"   Enhanced type: {results['initialization']['enhanced_dynamics_type']}")
        
        # Diagnostic pattern
        test_pattern = [0.5, 0.3, 0.8, 0.2, 0.6, 0.1, 0.9, 0.4] * 2  # 16D sensory input
        
        print(f"\n📊 Processing {cycles} cycles with diagnostic pattern...")
        
        for cycle in range(cycles):
            print(f"\n--- Cycle {cycle} ---")
            
            # Capture field state before processing
            field_state_before = self._capture_field_state(field_impl, enhanced_dynamics)
            
            # Process input
            action, brain_state = brain.process_sensory_input(test_pattern)
            
            # Capture field state after processing  
            field_state_after = self._capture_field_state(field_impl, enhanced_dynamics)
            
            # Calculate changes
            field_changes = self._calculate_field_changes(field_state_before, field_state_after)
            
            # Log cycle results
            cycle_result = {
                'cycle': cycle,
                'field_before': field_state_before,
                'field_after': field_state_after,
                'field_changes': field_changes,
                'brain_state': {
                    'field_energy': brain_state.get('field_energy', 0.0),
                    'field_evolution_cycles': brain_state.get('field_evolution_cycles', 0),
                    'prediction_confidence': brain_state.get('prediction_confidence', 0.0)
                },
                'action': action
            }
            
            results['field_evolution'].append(cycle_result)
            results['energy_evolution'].append(field_state_after.get('field_energy', 0.0))
            
            # Print cycle summary
            print(f"   Field energy: {field_state_before.get('field_energy', 0.0):.6f} → {field_state_after.get('field_energy', 0.0):.6f}")
            print(f"   Field change: {field_changes.get('energy_change', 0.0):.6f}")
            print(f"   Enhanced dynamics active: {field_changes.get('dynamics_active', False)}")
            
            # Detect issues
            if field_changes.get('energy_change', 0.0) == 0.0:
                if cycle == 0:
                    results['issues_detected'].append(f"Cycle {cycle}: No field energy change detected")
            
            if cycle > 5 and all(e == 0.0 for e in results['energy_evolution'][-5:]):
                results['issues_detected'].append(f"Cycle {cycle}: Field energy remained static for 5 cycles")
                break
        
        brain.finalize_session()
        
        # Generate diagnostics report
        report = self._generate_diagnostics_report(results)
        print(f"\n{report}")
        
        return results
    
    def _capture_field_state(self, field_impl, enhanced_dynamics) -> Dict[str, Any]:
        """Capture current field state for comparison"""
        state = {}
        
        try:
            # Get unified field if available
            if hasattr(field_impl, 'unified_field'):
                unified_field = field_impl.unified_field
                state['field_shape'] = list(unified_field.shape)
                state['field_energy'] = float(unified_field.sum().item() ** 2)
                state['field_max'] = float(unified_field.max().item())
                state['field_min'] = float(unified_field.min().item())
                state['field_mean'] = float(unified_field.mean().item())
                state['field_std'] = float(unified_field.std().item())
                
            # Get enhanced dynamics state if available
            if enhanced_dynamics:
                state['has_enhanced_dynamics'] = True
                if hasattr(enhanced_dynamics, 'phase_state'):
                    state['phase_state'] = enhanced_dynamics.phase_state
                if hasattr(enhanced_dynamics, 'attractor_states'):
                    state['num_attractors'] = len(enhanced_dynamics.attractor_states)
                if hasattr(enhanced_dynamics, 'energy_redistribution_count'):
                    state['energy_redistributions'] = enhanced_dynamics.energy_redistribution_count
            else:
                state['has_enhanced_dynamics'] = False
                
        except Exception as e:
            state['error'] = str(e)
        
        return state
    
    def _calculate_field_changes(self, before: Dict[str, Any], after: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate what changed in the field between states"""
        changes = {}
        
        try:
            # Energy change
            energy_before = before.get('field_energy', 0.0)
            energy_after = after.get('field_energy', 0.0)
            changes['energy_change'] = energy_after - energy_before
            
            # Statistics changes
            for stat in ['field_max', 'field_min', 'field_mean', 'field_std']:
                if stat in before and stat in after:
                    changes[f'{stat}_change'] = after[stat] - before[stat]
            
            # Enhanced dynamics activity
            changes['dynamics_active'] = after.get('has_enhanced_dynamics', False)
            
            # Attractor changes
            attractors_before = before.get('num_attractors', 0)
            attractors_after = after.get('num_attractors', 0)
            changes['attractor_change'] = attractors_after - attractors_before
            
        except Exception as e:
            changes['error'] = str(e)
        
        return changes
    
    def _generate_diagnostics_report(self, results: Dict[str, Any]) -> str:
        """Generate comprehensive diagnostics report"""
        report = []
        report.append("🔬 FIELD DYNAMICS DIAGNOSTICS REPORT")
        report.append("=" * 60)
        
        # Initialization summary
        init = results['initialization']
        report.append(f"📋 INITIALIZATION:")
        report.append(f"   Field Brain: {init['field_brain_type']}")
        report.append(f"   Field Implementation: {init['field_impl_type']}")
        report.append(f"   Enhanced Dynamics: {'✅ Enabled' if init['enhanced_dynamics_enabled'] else '❌ Disabled'}")
        
        # Energy evolution analysis
        energy_evolution = results['energy_evolution']
        if energy_evolution:
            initial_energy = energy_evolution[0]
            final_energy = energy_evolution[-1]
            max_energy = max(energy_evolution)
            energy_changes = [abs(energy_evolution[i] - energy_evolution[i-1]) for i in range(1, len(energy_evolution))]
            total_energy_change = abs(final_energy - initial_energy)
            
            report.append(f"\n⚡ ENERGY EVOLUTION:")
            report.append(f"   Initial energy: {initial_energy:.6f}")
            report.append(f"   Final energy: {final_energy:.6f}")
            report.append(f"   Max energy: {max_energy:.6f}")
            report.append(f"   Total change: {total_energy_change:.6f}")
            report.append(f"   Average change per cycle: {np.mean(energy_changes):.6f}")
        
        # Issues detected
        issues = results['issues_detected']
        if issues:
            report.append(f"\n⚠️ ISSUES DETECTED ({len(issues)}):")
            for issue in issues:
                report.append(f"   - {issue}")
        else:
            report.append(f"\n✅ NO ISSUES DETECTED")
        
        # Recommendations
        report.append(f"\n🎯 RECOMMENDATIONS:")
        if all(e == 0.0 for e in energy_evolution):
            report.append(f"   - Field energy is completely static - field evolution not working")
            report.append(f"   - Check field imprinting and evolution methods")
            report.append(f"   - Verify enhanced dynamics are actually being called")
        elif max(energy_evolution) < 1e-6:
            report.append(f"   - Field energy changes are extremely small")
            report.append(f"   - Consider increasing field evolution rate")
            report.append(f"   - Check field intensity calculations")
        else:
            report.append(f"   - Field dynamics appear to be working")
            report.append(f"   - Energy evolution detected within expected range")
        
        return "\n".join(report)


def main():
    """Run field dynamics diagnostics"""
    diagnostics = FieldDynamicsDiagnostics()
    results = diagnostics.diagnose_field_dynamics(cycles=15)
    return results


if __name__ == "__main__":
    main()