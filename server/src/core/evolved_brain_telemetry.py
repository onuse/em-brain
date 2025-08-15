"""
Enhanced Telemetry for Evolved Brain Architecture

Provides comprehensive observability into the emergent brain's evolution,
self-modification dynamics, and regional specialization.
"""

from typing import Dict, Any, List, Optional, Deque
from dataclasses import dataclass, field
from collections import deque
import time
import numpy as np
import torch


@dataclass
class EvolvedBrainTelemetry:
    """Complete telemetry snapshot from evolved brain"""
    
    # Core metrics
    brain_cycles: int
    field_information: float
    max_activation: float
    prediction_confidence: float
    cycle_time_ms: float
    
    # Evolution dynamics
    evolution_cycles: int
    self_modification_strength: float
    smoothed_information: float
    smoothed_confidence: float
    
    # Field dynamics
    exploration_drive: float
    internal_ratio: float  # Internal vs external processing
    information_state: str  # "low", "medium", "high"
    confidence_state: str  # "uncertain", "moderate", "confident"
    behavior_state: str  # "exploring", "exploiting", "balanced"
    
    # Working memory
    working_memory_patterns: int
    temporal_decay_rate: float
    spatial_decay_rate: float
    
    # Topology regions
    total_regions: int
    active_regions: int
    abstract_regions: int
    causal_links: int
    memory_saturation: float
    
    # Sensory organization
    unique_sensory_patterns: int
    sensory_mapping_events: int
    sensory_clustering_coefficient: float
    sensory_occupancy_ratio: float
    
    # Reward topology
    reward_impressions: int
    attractor_strength: float
    repulsor_strength: float
    
    # Motor patterns
    motor_acceptance_rate: float
    motor_variability: float
    
    # Device info
    device: str
    tensor_shape: List[int]
    memory_usage_mb: float
    
    # Prediction system phases (NEW)
    sensory_predictions: Optional[Dict[str, Any]] = None
    prediction_errors: Optional[Dict[str, Any]] = None
    hierarchical_predictions: Optional[Dict[str, Any]] = None
    action_selection: Optional[Dict[str, Any]] = None
    active_sensing: Optional[Dict[str, Any]] = None
    
    # Timestamp
    timestamp: float = field(default_factory=time.time)


class EvolvedBrainTelemetryAdapter:
    """
    Telemetry adapter specifically for SimplifiedUnifiedBrain with evolved dynamics.
    
    Tracks brain evolution, regional specialization, and emergent behaviors.
    """
    
    def __init__(self, brain):
        """
        Initialize telemetry adapter.
        
        Args:
            brain: SimplifiedUnifiedBrain instance or wrapper
        """
        # Handle wrapped brains
        if hasattr(brain, 'brain'):
            self.brain = brain.brain
        else:
            self.brain = brain
            
        # Telemetry history
        self.telemetry_history: Deque[EvolvedBrainTelemetry] = deque(maxlen=1000)
        
        # Evolution tracking
        self.evolution_snapshots = deque(maxlen=100)
        self.last_evolution_check = 0
        self.evolution_check_interval = 100  # cycles
        
        # Regional specialization tracking
        self.regional_patterns = {}
        self.regional_dynamics = {}
        
        # Behavior pattern tracking
        self.behavior_transitions = deque(maxlen=50)
        self.last_behavior_state = None
    
    def get_telemetry(self) -> EvolvedBrainTelemetry:
        """Extract comprehensive telemetry from evolved brain."""
        
        # Get brain state
        brain_state = self.brain._create_brain_state()
        
        # Get evolution state
        evo_state = self.brain.get_evolution_state()
        
        # Get field dynamics state
        field_dynamics = self.brain.field_dynamics
        emergent_props = field_dynamics.get_emergent_properties()
        
        # Get modulation state
        modulation = self.brain.modulation
        
        # Determine states
        information_level = emergent_props.get('smoothed_information', emergent_props.get('smoothed_energy', 0.5))
        if information_level < 0.3:
            information_state = "low"
        elif information_level < 0.7:
            information_state = "medium"
        else:
            information_state = "high"
            
        confidence = emergent_props['smoothed_confidence']
        if confidence < 0.4:
            confidence_state = "uncertain"
        elif confidence < 0.7:
            confidence_state = "moderate"
        else:
            confidence_state = "confident"
            
        exploration = modulation.get('exploration_drive', 0.5)
        if exploration > 0.6:
            behavior_state = "exploring"
        elif exploration < 0.4:
            behavior_state = "exploiting"
        else:
            behavior_state = "balanced"
        
        # Track behavior transitions
        if behavior_state != self.last_behavior_state:
            self.behavior_transitions.append({
                'from': self.last_behavior_state,
                'to': behavior_state,
                'cycle': self.brain.brain_cycles,
                'timestamp': time.time()
            })
            self.last_behavior_state = behavior_state
        
        # Get topology stats
        topology_stats = brain_state['topology_regions']
        
        # Get sensory organization stats
        sensory_stats = brain_state['sensory_organization']
        
        # Get reward topology state
        reward_state = brain_state['topology_shaping']
        
        # Get motor stats
        motor_stats = self.brain.motor_cortex.get_statistics()
        
        # Calculate memory usage
        memory_mb = self.brain._calculate_memory_usage()
        
        # Extract 5-phase prediction metrics
        sensory_predictions = self._extract_sensory_predictions()
        prediction_errors = self._extract_prediction_errors()
        hierarchical_predictions = self._extract_hierarchical_predictions()
        action_selection = self._extract_action_selection()
        active_sensing = self._extract_active_sensing()
        
        # Create telemetry snapshot
        telemetry = EvolvedBrainTelemetry(
            # Core metrics
            brain_cycles=brain_state['cycle'],
            field_information=brain_state.get('field_information', brain_state.get('field_energy', 0.0)),
            max_activation=brain_state['max_activation'],
            prediction_confidence=brain_state['prediction_confidence'],
            cycle_time_ms=brain_state['cycle_time_ms'],
            
            # Evolution dynamics
            evolution_cycles=evo_state['evolution_cycles'],
            self_modification_strength=evo_state['self_modification_strength'],
            smoothed_information=evo_state.get('smoothed_information', evo_state.get('smoothed_energy', 0.5)),
            smoothed_confidence=evo_state['smoothed_confidence'],
            
            # Field dynamics
            exploration_drive=modulation.get('exploration_drive', 0.5),
            internal_ratio=modulation.get('internal_ratio', 0.5),
            information_state=information_state,
            confidence_state=confidence_state,
            behavior_state=behavior_state,
            
            # Working memory
            working_memory_patterns=evo_state['working_memory']['n_patterns'],
            temporal_decay_rate=0.995,  # Fixed values from evolved field dynamics
            spatial_decay_rate=0.95,
            
            # Topology regions
            total_regions=topology_stats['total'],
            active_regions=topology_stats['active'],
            abstract_regions=topology_stats['abstract'],
            causal_links=topology_stats['causal_links'],
            memory_saturation=brain_state['memory_saturation'],
            
            # Sensory organization
            unique_sensory_patterns=sensory_stats['unique_patterns'],
            sensory_mapping_events=sensory_stats['mapping_events'],
            sensory_clustering_coefficient=sensory_stats['clustering_coefficient'],
            sensory_occupancy_ratio=sensory_stats['occupancy_ratio'],
            
            # Reward topology
            reward_impressions=reward_state.get('active_attractors', 0),
            attractor_strength=reward_state.get('strongest_attractor', 0.0),
            repulsor_strength=0.0,  # Not tracked in current implementation
            
            # Motor patterns
            motor_acceptance_rate=motor_stats.get('acceptance_rate', 0.0),
            motor_variability=0.0,  # Not currently tracked
            
            # Prediction system phases
            sensory_predictions=sensory_predictions,
            prediction_errors=prediction_errors,
            hierarchical_predictions=hierarchical_predictions,
            action_selection=action_selection,
            active_sensing=active_sensing,
            
            # Device info
            device=brain_state['device'],
            tensor_shape=brain_state['tensor_shape'],
            memory_usage_mb=memory_mb
        )
        
        # Store in history
        self.telemetry_history.append(telemetry)
        
        # Check for evolution milestone
        if self.brain.brain_cycles - self.last_evolution_check >= self.evolution_check_interval:
            self._capture_evolution_snapshot()
            self.last_evolution_check = self.brain.brain_cycles
        
        return telemetry
    
    def _extract_sensory_predictions(self) -> Dict[str, Any]:
        """Extract Phase 1 sensory prediction metrics."""
        try:
            if hasattr(self.brain, 'predictive_field_system'):
                pfs = self.brain.predictive_field_system
                stats = pfs.get_statistics() if hasattr(pfs, 'get_statistics') else {}
                
                # Get per-sensor confidence
                confidence_per_sensor = []
                if hasattr(pfs, 'sensor_confidence'):
                    confidence_per_sensor = pfs.sensor_confidence.tolist()
                elif 'sensor_confidence' in stats:
                    confidence_per_sensor = stats['sensor_confidence']
                
                # Count specialized regions
                specialized_regions = 0
                if hasattr(self.brain.topology_region_system, 'regions'):
                    for region in self.brain.topology_region_system.regions:
                        if hasattr(region, 'is_sensory_predictive') and region.is_sensory_predictive:
                            specialized_regions += 1
                
                return {
                    'accuracy': stats.get('prediction_accuracy', 0.0),
                    'confidence_per_sensor': confidence_per_sensor,
                    'specialized_regions': specialized_regions,
                    'momentum_predictions': stats.get('momentum_predictions', False)
                }
            
            # Fallback if predictive system not available
            return {
                'accuracy': self.brain._create_brain_state().get('prediction_confidence', 0.0),
                'confidence_per_sensor': [],
                'specialized_regions': 0,
                'momentum_predictions': False
            }
        except Exception:
            return {}
    
    def _extract_prediction_errors(self) -> Dict[str, Any]:
        """Extract Phase 2 prediction error learning metrics."""
        try:
            evo_state = self.brain.get_evolution_state()
            
            # Get current prediction error
            if hasattr(self.brain, '_last_prediction_error'):
                error_magnitude = float(torch.mean(torch.abs(self.brain._last_prediction_error)))
            else:
                error_magnitude = 0.0
            
            # Get self-modification strength
            mod_strength = evo_state.get('self_modification_strength', 0.01)
            
            # High error regions
            high_error_count = 0
            if hasattr(self.brain, '_error_field') and self.brain._error_field is not None:
                threshold = torch.mean(self.brain._error_field) + torch.std(self.brain._error_field)
                high_error_count = int(torch.sum(self.brain._error_field > threshold))
            
            return {
                'magnitude': error_magnitude,
                'modification_strength': mod_strength,
                'high_error_count': high_error_count
            }
        except Exception:
            return {}
    
    def _extract_hierarchical_predictions(self) -> Dict[str, Any]:
        """Extract Phase 3 hierarchical timescale metrics."""
        try:
            if hasattr(self.brain, 'hierarchical_prediction_system'):
                hps = self.brain.hierarchical_prediction_system
                stats = hps.get_statistics() if hasattr(hps, 'get_statistics') else {}
                
                return {
                    'immediate_accuracy': stats.get('immediate_accuracy', 0.0),
                    'short_term_accuracy': stats.get('short_term_accuracy', 0.0),
                    'long_term_accuracy': stats.get('long_term_accuracy', 0.0),
                    'abstract_pattern_count': stats.get('abstract_patterns', 0)
                }
            
            # Estimate from field structure if system not available
            field = self.brain.unified_field
            return {
                'immediate_accuracy': 0.0,
                'short_term_accuracy': 0.0,
                'long_term_accuracy': 0.0,
                'abstract_pattern_count': len(self.brain.topology_region_system.regions) if hasattr(self.brain.topology_region_system, 'regions') else 0
            }
        except Exception:
            return {}
    
    def _extract_action_selection(self) -> Dict[str, Any]:
        """Extract Phase 4 action prediction metrics."""
        try:
            if hasattr(self.brain, 'action_prediction_system'):
                aps = self.brain.action_prediction_system
                stats = aps.get_statistics() if hasattr(aps, 'get_statistics') else {}
                
                # Get strategy counts
                history = getattr(aps, 'strategy_history', [])
                total = len(history)
                exploit = sum(1 for s in history if s == 'exploit')
                explore = sum(1 for s in history if s == 'explore')
                test = sum(1 for s in history if s == 'test')
                
                # Get current strategy
                current_strategy = 'unknown'
                if hasattr(aps, '_last_strategy'):
                    current_strategy = aps._last_strategy
                elif total > 0:
                    current_strategy = history[-1]
                
                return {
                    'current_strategy': current_strategy,
                    'total_actions': total,
                    'exploit_count': exploit,
                    'explore_count': explore,
                    'test_count': test,
                    'preview_accuracy': stats.get('preview_accuracy', 0.0)
                }
            
            # Estimate from exploration drive
            modulation = self.brain.modulation
            exploration = modulation.get('exploration_drive', 0.5)
            
            if exploration > 0.6:
                strategy = 'explore'
            elif exploration < 0.4:
                strategy = 'exploit'
            else:
                strategy = 'test'
            
            return {
                'current_strategy': strategy,
                'total_actions': self.brain.brain_cycles,
                'exploit_count': 0,
                'explore_count': 0,
                'test_count': 0,
                'preview_accuracy': 0.0
            }
        except Exception:
            return {}
    
    def _extract_active_sensing(self) -> Dict[str, Any]:
        """Extract Phase 5 active sensing metrics."""
        try:
            if hasattr(self.brain, 'active_vision') and self.brain.active_vision is not None:
                avs = self.brain.active_vision
                stats = avs.get_attention_statistics() if hasattr(avs, 'get_attention_statistics') else {}
                
                # Get uncertainty
                if hasattr(avs, '_last_uncertainty_map'):
                    uncertainty = avs._last_uncertainty_map.total_uncertainty
                else:
                    uncertainty = stats.get('uncertainty_level', 0.0)
                
                # Get pattern
                pattern = stats.get('current_pattern', 'unknown')
                
                # Get information gain
                info_gain = 0.0
                if hasattr(avs, 'information_gain_history') and len(avs.information_gain_history) > 0:
                    info_gain = np.mean(avs.information_gain_history[-10:])
                
                return {
                    'total_uncertainty': uncertainty,
                    'current_pattern': pattern,
                    'saccade_count': stats.get('saccade_count', 0),
                    'smooth_pursuit_time': stats.get('smooth_pursuit_ratio', 0.0),
                    'information_gain': info_gain
                }
            
            return {
                'total_uncertainty': 0.0,
                'current_pattern': 'inactive',
                'saccade_count': 0,
                'smooth_pursuit_time': 0.0,
                'information_gain': 0.0
            }
        except Exception:
            return {}
    
    def _capture_evolution_snapshot(self):
        """Capture detailed evolution snapshot for long-term tracking."""
        
        # Extract dynamics features from field
        field = self.brain.unified_field
        dynamics_features = field[:, :, :, -16:].detach().cpu()
        
        # Compute regional statistics
        regional_stats = []
        
        # Sample regions (e.g., 8 spatial regions)
        spatial_res = field.shape[0]
        region_size = spatial_res // 2
        
        for x in range(0, spatial_res, region_size):
            for y in range(0, spatial_res, region_size):
                for z in range(0, spatial_res, region_size):
                    region = dynamics_features[
                        x:x+region_size,
                        y:y+region_size,
                        z:z+region_size,
                        :
                    ]
                    
                    # Compute statistics
                    stats = {
                        'location': (x, y, z),
                        'mean_decay': torch.mean(region[:, :, :, 0]).item(),
                        'mean_diffusion': torch.mean(region[:, :, :, 4]).item(),
                        'mean_coupling': torch.mean(region[:, :, :, 8]).item(),
                        'mean_plasticity': torch.mean(region[:, :, :, 12]).item(),
                        'activity': torch.mean(torch.abs(field[
                            x:x+region_size,
                            y:y+region_size,
                            z:z+region_size,
                            :-16
                        ])).item()
                    }
                    regional_stats.append(stats)
        
        snapshot = {
            'cycle': self.brain.brain_cycles,
            'timestamp': time.time(),
            'self_modification_strength': self.brain.field_dynamics.self_modification_strength,
            'regional_stats': regional_stats,
            'global_energy': torch.mean(torch.abs(field[:, :, :, :-16])).item(),
            'dynamics_variance': torch.var(dynamics_features).item()
        }
        
        self.evolution_snapshots.append(snapshot)
    
    def get_telemetry_history(self, num_samples: int = 10) -> List[EvolvedBrainTelemetry]:
        """Get recent telemetry history."""
        return list(self.telemetry_history)[-num_samples:]
    
    def get_evolution_trajectory(self) -> Dict[str, Any]:
        """Analyze brain's evolution trajectory."""
        
        if len(self.evolution_snapshots) < 2:
            return {'status': 'insufficient_data'}
        
        # Extract time series
        cycles = [s['cycle'] for s in self.evolution_snapshots]
        self_mod_strengths = [s['self_modification_strength'] for s in self.evolution_snapshots]
        global_energies = [s['global_energy'] for s in self.evolution_snapshots]
        dynamics_variances = [s['dynamics_variance'] for s in self.evolution_snapshots]
        
        # Compute trends
        def compute_trend(values):
            if len(values) < 2:
                return 0.0
            x = np.arange(len(values))
            return np.polyfit(x, values, 1)[0]
        
        return {
            'snapshots': len(self.evolution_snapshots),
            'cycles_tracked': cycles[-1] - cycles[0] if cycles else 0,
            'self_modification_trend': compute_trend(self_mod_strengths),
            'energy_trend': compute_trend(global_energies),
            'dynamics_variance_trend': compute_trend(dynamics_variances),
            'current_self_modification': self_mod_strengths[-1] if self_mod_strengths else 0,
            'regional_specialization': self._analyze_regional_specialization()
        }
    
    def _analyze_regional_specialization(self) -> Dict[str, Any]:
        """Analyze how different regions have specialized."""
        
        if len(self.evolution_snapshots) < 2:
            return {'status': 'insufficient_data'}
        
        # Compare first and last snapshots
        first = self.evolution_snapshots[0]
        last = self.evolution_snapshots[-1]
        
        # Group regions by their dynamics
        region_types = {
            'fast': [],     # Low decay, high diffusion
            'slow': [],     # High decay, low diffusion
            'coupled': [],  # High coupling
            'plastic': []   # High plasticity
        }
        
        for region in last['regional_stats']:
            if region['mean_decay'] < -0.01:  # Negative decay = persistence
                region_types['slow'].append(region)
            if region['mean_diffusion'] > 0.01:
                region_types['fast'].append(region)
            if region['mean_coupling'] > 0.01:
                region_types['coupled'].append(region)
            if region['mean_plasticity'] > 0.01:
                region_types['plastic'].append(region)
        
        return {
            'fast_regions': len(region_types['fast']),
            'slow_regions': len(region_types['slow']),
            'coupled_regions': len(region_types['coupled']),
            'plastic_regions': len(region_types['plastic']),
            'specialization_index': self._compute_specialization_index(last['regional_stats'])
        }
    
    def _compute_specialization_index(self, regional_stats: List[Dict]) -> float:
        """Compute how specialized regions have become (0-1)."""
        
        if not regional_stats:
            return 0.0
        
        # Compute variance across regions for each parameter
        params = ['mean_decay', 'mean_diffusion', 'mean_coupling', 'mean_plasticity']
        variances = []
        
        for param in params:
            values = [r[param] for r in regional_stats]
            if values:
                variances.append(np.var(values))
        
        # Higher variance = more specialization
        return float(np.mean(variances)) * 100  # Scale for readability
    
    def get_behavior_analysis(self) -> Dict[str, Any]:
        """Analyze behavior patterns and transitions."""
        
        recent_telemetry = list(self.telemetry_history)[-100:]
        
        if not recent_telemetry:
            return {'status': 'no_data'}
        
        # Count time in each state
        state_times = {
            'exploring': 0,
            'exploiting': 0,
            'balanced': 0
        }
        
        for t in recent_telemetry:
            state_times[t.behavior_state] += 1
        
        # Analyze transitions
        transition_matrix = {
            'exploring': {'exploring': 0, 'exploiting': 0, 'balanced': 0},
            'exploiting': {'exploring': 0, 'exploiting': 0, 'balanced': 0},
            'balanced': {'exploring': 0, 'exploiting': 0, 'balanced': 0}
        }
        
        for transition in self.behavior_transitions:
            if transition['from'] and transition['to']:
                transition_matrix[transition['from']][transition['to']] += 1
        
        return {
            'state_distribution': state_times,
            'dominant_state': max(state_times, key=state_times.get),
            'transition_count': len(self.behavior_transitions),
            'transition_matrix': transition_matrix,
            'stability_index': self._compute_behavior_stability(recent_telemetry)
        }
    
    def _compute_behavior_stability(self, telemetry_samples: List[EvolvedBrainTelemetry]) -> float:
        """Compute behavior stability (0-1, higher = more stable)."""
        
        if len(telemetry_samples) < 2:
            return 1.0
        
        # Count state changes
        changes = 0
        for i in range(1, len(telemetry_samples)):
            if telemetry_samples[i].behavior_state != telemetry_samples[i-1].behavior_state:
                changes += 1
        
        # Normalize by sample count
        return 1.0 - (changes / (len(telemetry_samples) - 1))
    
    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive telemetry summary."""
        
        current = self.get_telemetry()
        evolution = self.get_evolution_trajectory()
        behavior = self.get_behavior_analysis()
        
        return {
            'current_state': {
                'cycles': current.brain_cycles,
                'information': f"{current.information_state} ({current.field_information:.4f})",
                'confidence': f"{current.confidence_state} ({current.prediction_confidence:.2%})",
                'behavior': current.behavior_state,
                'self_modification': f"{current.self_modification_strength:.1%}",
                'evolution_cycles': current.evolution_cycles
            },
            'memory': {
                'working_patterns': current.working_memory_patterns,
                'topology_regions': current.total_regions,
                'saturation': f"{current.memory_saturation:.1%}",
                'sensory_patterns': current.unique_sensory_patterns
            },
            'evolution': {
                'trajectory': evolution.get('self_modification_trend', 0),
                'specialization': evolution.get('regional_specialization', {})
            },
            'behavior': {
                'dominant': behavior.get('dominant_state', 'unknown'),
                'stability': behavior.get('stability_index', 0),
                'transitions': behavior.get('transition_count', 0)
            },
            'performance': {
                'cycle_time_ms': current.cycle_time_ms,
                'device': current.device,
                'memory_mb': current.memory_usage_mb
            }
        }