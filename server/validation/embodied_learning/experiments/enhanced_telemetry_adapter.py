"""
Enhanced Telemetry Adapter for Biological Embodied Learning

Extends the monitoring client to access rich evolved brain telemetry
during validation experiments.
"""

import json
import time
from typing import Dict, Any, Optional, List
from src.communication.monitoring_client import BrainMonitoringClient


class EnhancedTelemetryAdapter:
    """
    Adapter that provides rich telemetry access for validation experiments.
    
    Tracks evolution dynamics, regional specialization, and emergent behaviors
    during biological embodied learning experiments.
    """
    
    def __init__(self, monitoring_client: Optional[BrainMonitoringClient]):
        """Initialize adapter with monitoring client."""
        self.client = monitoring_client
        self.telemetry_history = []
        self.evolution_snapshots = []
        self.behavior_transitions = []
        
    def get_evolved_telemetry(self, session_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Get comprehensive evolved brain telemetry."""
        if not self.client:
            return None
            
        try:
            # Request telemetry via monitoring socket
            if session_id:
                response = self.client.request_data(f"telemetry {session_id}")
            else:
                response = self.client.request_data("telemetry")
                
            if response and response.get('status') == 'success':
                data = response.get('data', {})
                
                # Debug: check what we received
                if not data:
                    print(f"   ⚠️  Empty telemetry data received")
                elif isinstance(data, dict) and len(data) == 0:
                    print(f"   ⚠️  Telemetry returned empty dict")
                
                # Handle different response formats
                if isinstance(data, dict):
                    # Check if data has session keys
                    if any(key.startswith('session_') for key in data.keys()):
                        # This is a dict of sessions
                        if session_id and session_id in data:
                            # Specific session requested
                            telemetry = data[session_id]
                        else:
                            # Get first session
                            session_keys = [k for k in data.keys() if k.startswith('session_')]
                            if session_keys:
                                first_session = session_keys[0]
                                telemetry = data[first_session]
                                if not session_id:  # Only print if no session was specified
                                    # print(f"   📊 Using telemetry from {first_session}")  # Commented out to reduce log spam
                                    pass
                            else:
                                telemetry = data
                    else:
                        # Direct telemetry data
                        telemetry = data
                else:
                    telemetry = data
                
                self.telemetry_history.append(telemetry)
                
                # Extract evolution data if available
                if 'evolution_state' in telemetry:
                    self.evolution_snapshots.append({
                        'timestamp': telemetry.get('timestamp', 0),
                        'cycles': telemetry.get('cycle', 0),
                        'self_modification': telemetry['evolution_state'].get('self_modification_strength', 0),
                        'evolution_cycles': telemetry['evolution_state'].get('evolution_cycles', 0)
                    })
                
                return telemetry
        except Exception as e:
            print(f"Failed to get evolved telemetry: {e}")
            
        return None
    
    def get_learning_metrics(self) -> Dict[str, Any]:
        """Extract learning-relevant metrics from telemetry."""
        if not self.telemetry_history:
            return {}
            
        latest = self.telemetry_history[-1]
        
        # Extract key learning indicators
        metrics = {
            'prediction_confidence': latest.get('prediction_confidence', 0.5),
            'field_energy': latest.get('field_energy', 0.0),
            'learning_detected': False,
            'efficiency': 0.0
        }
        
        # Check evolution state
        if 'evolution_state' in latest:
            evo = latest['evolution_state']
            metrics['self_modification_strength'] = evo.get('self_modification_strength', 0.01)
            metrics['evolution_cycles'] = evo.get('evolution_cycles', 0)
            metrics['working_memory_patterns'] = evo.get('working_memory', {}).get('n_patterns', 0)
            
        # Check energy state for behavior
        if 'energy_state' in latest:
            energy = latest['energy_state']
            metrics['exploration_drive'] = energy.get('exploration_drive', 0.5)
            metrics['energy_level'] = energy.get('energy', 0.5)
            metrics['novelty'] = energy.get('novelty', 0.0)
            
        # Calculate efficiency from multiple factors
        confidence = metrics['prediction_confidence']
        energy = metrics['field_energy']
        
        # If we have evolution data, use it for efficiency calculation
        if 'self_modification_strength' in metrics and 'evolution_cycles' in metrics:
            # Efficiency based on: confidence, self-modification activity, and working memory usage
            sm_factor = min(1.0, metrics['self_modification_strength'] / 0.1)  # Normalize to 0-1
            wm_factor = min(1.0, metrics['working_memory_patterns'] / 10.0)  # Normalize to 0-1
            
            # Weighted combination: confidence matters most, but activity indicators also contribute
            metrics['efficiency'] = (confidence * 0.5 + sm_factor * 0.3 + wm_factor * 0.2)
        elif energy > 0:
            # Fallback to energy-based calculation if available
            if confidence > 0.6 and 0.2 < energy < 0.8:
                metrics['efficiency'] = confidence * 0.8
            else:
                metrics['efficiency'] = confidence * 0.5
        else:
            # Final fallback: use confidence with learning detection boost
            base_efficiency = confidence * 0.5
            if metrics.get('learning_detected', False):
                metrics['efficiency'] = min(1.0, base_efficiency * 1.5)
            else:
                metrics['efficiency'] = base_efficiency
            
        # Detect learning from confidence improvement
        if len(self.telemetry_history) > 10:
            early_confidence = sum(t.get('prediction_confidence', 0.5) for t in self.telemetry_history[:5]) / 5
            recent_confidence = sum(t.get('prediction_confidence', 0.5) for t in self.telemetry_history[-5:]) / 5
            metrics['learning_detected'] = recent_confidence > early_confidence * 1.1
            
        return metrics
    
    def get_evolution_analysis(self) -> Dict[str, Any]:
        """Analyze brain evolution over experiment."""
        if len(self.evolution_snapshots) < 2:
            return {'status': 'insufficient_data'}
            
        # Track self-modification progression
        self_mod_values = [s['self_modification'] for s in self.evolution_snapshots]
        evolution_cycles = [s['evolution_cycles'] for s in self.evolution_snapshots]
        
        return {
            'initial_self_modification': self_mod_values[0],
            'final_self_modification': self_mod_values[-1],
            'self_modification_growth': self_mod_values[-1] - self_mod_values[0],
            'total_evolution_cycles': evolution_cycles[-1] if evolution_cycles else 0,
            'self_modification_trend': self_mod_values
        }
    
    def get_behavioral_analysis(self) -> Dict[str, Any]:
        """Analyze behavioral patterns from telemetry."""
        if not self.telemetry_history:
            return {}
            
        # Track behavior states
        behavior_states = []
        exploration_drives = []
        
        for telemetry in self.telemetry_history[-20:]:  # Last 20 samples
            if 'energy_state' in telemetry:
                energy_state = telemetry['energy_state']
                exploration = energy_state.get('exploration_drive', 0.5)
                exploration_drives.append(exploration)
                
                # Classify behavior
                if exploration > 0.6:
                    behavior_states.append('exploring')
                elif exploration < 0.4:
                    behavior_states.append('exploiting')
                else:
                    behavior_states.append('balanced')
        
        # Count transitions
        transitions = 0
        for i in range(1, len(behavior_states)):
            if behavior_states[i] != behavior_states[i-1]:
                transitions += 1
                
        return {
            'dominant_behavior': max(set(behavior_states), key=behavior_states.count) if behavior_states else 'unknown',
            'behavior_transitions': transitions,
            'avg_exploration_drive': sum(exploration_drives) / len(exploration_drives) if exploration_drives else 0.5,
            'behavior_stability': 1.0 - (transitions / max(1, len(behavior_states) - 1))
        }
    
    def get_memory_utilization(self) -> Dict[str, Any]:
        """Analyze memory system utilization."""
        if not self.telemetry_history:
            return {}
            
        latest = self.telemetry_history[-1]
        
        memory_stats = {
            'memory_saturation': latest.get('memory_saturation', 0.0),
            'working_memory_patterns': 0,
            'topology_regions': {},
            'sensory_organization': {}
        }
        
        # Extract working memory from evolution state
        if 'evolution_state' in latest:
            wm = latest['evolution_state'].get('working_memory', {})
            memory_stats['working_memory_patterns'] = wm.get('n_patterns', 0)
            
        # Extract topology regions
        if 'topology_regions' in latest:
            topology = latest['topology_regions']
            memory_stats['topology_regions'] = {
                'total': topology.get('total', 0),
                'active': topology.get('active', 0),
                'abstract': topology.get('abstract', 0),
                'causal_links': topology.get('causal_links', 0)
            }
            
        # Extract sensory organization
        if 'sensory_organization' in latest:
            sensory = latest['sensory_organization']
            memory_stats['sensory_organization'] = {
                'unique_patterns': sensory.get('unique_patterns', 0),
                'mapping_events': sensory.get('mapping_events', 0),
                'clustering_coefficient': sensory.get('clustering_coefficient', 0.0)
            }
            
        return memory_stats
    
    def get_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive telemetry report for experiment."""
        return {
            'learning_metrics': self.get_learning_metrics(),
            'evolution_analysis': self.get_evolution_analysis(),
            'behavioral_analysis': self.get_behavioral_analysis(),
            'memory_utilization': self.get_memory_utilization(),
            'telemetry_samples': len(self.telemetry_history),
            'evolution_snapshots': len(self.evolution_snapshots)
        }