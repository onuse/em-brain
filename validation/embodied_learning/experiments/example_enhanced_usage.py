#!/usr/bin/env python3
"""
Example of enhanced telemetry usage in biological embodied learning.

Shows how to integrate rich evolved brain telemetry into validation experiments.
"""

# Example modifications to biological_embodied_learning.py:

def enhanced_run_learning_session(self, session_id: int, is_baseline: bool = False) -> SessionResults:
    """Enhanced version that captures evolution dynamics."""
    
    # ... existing session setup ...
    
    # Create enhanced telemetry adapter
    from enhanced_telemetry_adapter import EnhancedTelemetryAdapter
    telemetry_adapter = EnhancedTelemetryAdapter(self.monitoring_client)
    
    # Track evolution state at session start
    start_telemetry = telemetry_adapter.get_evolved_telemetry()
    if start_telemetry:
        print(f"   🧬 Brain state at session start:")
        print(f"      Self-modification: {start_telemetry.get('evolution_state', {}).get('self_modification_strength', 0):.1%}")
        print(f"      Evolution cycles: {start_telemetry.get('evolution_state', {}).get('evolution_cycles', 0)}")
    
    # ... main learning loop ...
    
    # During the loop, periodically capture telemetry
    if actions_executed % 100 == 0:  # Every 100 actions
        telemetry = telemetry_adapter.get_evolved_telemetry()
        if telemetry:
            # Track behavior transitions
            energy_state = telemetry.get('energy_state', {})
            exploration = energy_state.get('exploration_drive', 0.5)
            
            # Log interesting state changes
            if exploration > 0.7:
                print(f"   🔍 High exploration mode detected (drive: {exploration:.2f})")
            elif exploration < 0.3:
                print(f"   🎯 Exploitation mode detected (drive: {exploration:.2f})")
    
    # ... end of session ...
    
    # Get comprehensive telemetry report
    telemetry_report = telemetry_adapter.get_comprehensive_report()
    
    # Enhanced session results with evolution data
    session_results = SessionResults(
        # ... existing fields ...
        
        # Add new telemetry-based fields
        confidence_progression=telemetry_report['learning_metrics'].get('confidence_progression', []),
        confidence_patterns=['exploring' if e > 0.6 else 'exploiting' for e in 
                           telemetry_report['behavioral_analysis'].get('exploration_drives', [])],
        final_confidence_state={
            'prediction_confidence': telemetry_report['learning_metrics'].get('prediction_confidence', 0.5),
            'self_modification': telemetry_report['evolution_analysis'].get('final_self_modification', 0.01),
            'working_memory_patterns': telemetry_report['memory_utilization'].get('working_memory_patterns', 0),
            'behavior_state': telemetry_report['behavioral_analysis'].get('dominant_behavior', 'unknown')
        }
    )
    
    # Enhanced summary output
    print(f"   🧬 Evolution Progress:")
    print(f"      Self-modification growth: {telemetry_report['evolution_analysis'].get('self_modification_growth', 0):.3%}")
    print(f"      Behavior stability: {telemetry_report['behavioral_analysis'].get('behavior_stability', 0):.2f}")
    print(f"      Memory utilization: {telemetry_report['memory_utilization'].get('memory_saturation', 0):.1%}")
    
    return session_results


def enhanced_consolidation_analysis(self, session_id: int) -> ConsolidationAnalysis:
    """Enhanced consolidation that tracks topology changes."""
    
    # Get telemetry before consolidation
    pre_telemetry = self.monitoring_client.request_data("telemetry") if self.monitoring_client else None
    pre_topology = {}
    
    if pre_telemetry and pre_telemetry.get('status') == 'success':
        data = pre_telemetry.get('data', {})
        pre_topology = data.get('topology_regions', {})
        print(f"   🧠 Pre-consolidation topology:")
        print(f"      Regions: {pre_topology.get('total', 0)}")
        print(f"      Causal links: {pre_topology.get('causal_links', 0)}")
    
    # ... consolidation sleep ...
    
    # Get telemetry after consolidation
    post_telemetry = self.monitoring_client.request_data("telemetry") if self.monitoring_client else None
    post_topology = {}
    
    if post_telemetry and post_telemetry.get('status') == 'success':
        data = post_telemetry.get('data', {})
        post_topology = data.get('topology_regions', {})
        print(f"   🧠 Post-consolidation topology:")
        print(f"      Regions: {post_topology.get('total', 0)} (Δ{post_topology.get('total', 0) - pre_topology.get('total', 0)})")
        print(f"      Causal links: {post_topology.get('causal_links', 0)} (Δ{post_topology.get('causal_links', 0) - pre_topology.get('causal_links', 0)})")
    
    # Calculate topology-based consolidation benefit
    topology_growth = (post_topology.get('total', 0) - pre_topology.get('total', 0)) / max(1, pre_topology.get('total', 1))
    
    return ConsolidationAnalysis(
        # ... existing fields ...
        topology_growth=topology_growth,
        causal_link_formation=post_topology.get('causal_links', 0) - pre_topology.get('causal_links', 0)
    )


# Example of analyzing evolution trajectory
def analyze_brain_evolution(telemetry_history: List[Dict]) -> Dict:
    """Analyze how the brain evolved during the experiment."""
    
    if len(telemetry_history) < 10:
        return {'insufficient_data': True}
    
    # Extract self-modification progression
    self_mod_values = []
    evolution_cycles = []
    working_memory_sizes = []
    
    for telemetry in telemetry_history:
        if 'evolution_state' in telemetry:
            evo = telemetry['evolution_state']
            self_mod_values.append(evo.get('self_modification_strength', 0.01))
            evolution_cycles.append(evo.get('evolution_cycles', 0))
            wm = evo.get('working_memory', {})
            working_memory_sizes.append(wm.get('n_patterns', 0))
    
    # Analyze trends
    import numpy as np
    
    # Self-modification should gradually increase
    self_mod_trend = np.polyfit(range(len(self_mod_values)), self_mod_values, 1)[0]
    
    # Working memory should stabilize
    wm_variance_early = np.var(working_memory_sizes[:len(working_memory_sizes)//2])
    wm_variance_late = np.var(working_memory_sizes[len(working_memory_sizes)//2:])
    wm_stabilization = 1.0 - (wm_variance_late / (wm_variance_early + 1e-6))
    
    return {
        'self_modification_trend': self_mod_trend,
        'final_self_modification': self_mod_values[-1] if self_mod_values else 0.01,
        'evolution_acceleration': evolution_cycles[-1] / len(telemetry_history) if evolution_cycles else 0,
        'working_memory_stabilization': wm_stabilization,
        'brain_maturity_score': min(1.0, self_mod_values[-1] * 10) if self_mod_values else 0  # 10% self-mod = mature
    }


# Example visualization of evolution dynamics
def plot_evolution_dynamics(results: Dict):
    """Plot brain evolution dynamics over experiment."""
    import matplotlib.pyplot as plt
    
    sessions = results['session_results']
    
    # Extract evolution data from session confidence states
    self_mod_values = []
    wm_patterns = []
    confidence_values = []
    
    for session in sessions:
        final_state = session.get('final_confidence_state', {})
        self_mod_values.append(final_state.get('self_modification', 0.01))
        wm_patterns.append(final_state.get('working_memory_patterns', 0))
        confidence_values.append(final_state.get('prediction_confidence', 0.5))
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
    
    # Self-modification strength
    ax1.plot(range(len(self_mod_values)), self_mod_values, 'b-o', linewidth=2)
    ax1.set_ylabel('Self-Modification Strength')
    ax1.set_title('Brain Evolution: Self-Modification Growth')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 0.12)
    
    # Working memory patterns
    ax2.plot(range(len(wm_patterns)), wm_patterns, 'g-s', linewidth=2)
    ax2.set_ylabel('Working Memory Patterns')
    ax2.set_title('Brain Evolution: Working Memory Development')
    ax2.grid(True, alpha=0.3)
    
    # Prediction confidence
    ax3.plot(range(len(confidence_values)), confidence_values, 'r-^', linewidth=2)
    ax3.set_xlabel('Session')
    ax3.set_ylabel('Prediction Confidence')
    ax3.set_title('Brain Evolution: Confidence Growth')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig('brain_evolution_dynamics.png', dpi=300, bbox_inches='tight')
    plt.close()