#!/usr/bin/env python3
"""
Enhanced Telemetry Adapter for 5-Phase Prediction System

Provides detailed monitoring of the predictive brain architecture:
1. Sensory prediction accuracy per sensor
2. Prediction error magnitudes and learning response
3. Hierarchical timescale predictions
4. Action selection strategies (exploit/explore/test)
5. Active sensing and uncertainty-driven attention
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import deque
import time


class EnhancedTelemetryAdapter:
    """Adapter to extract detailed metrics from the predictive brain via monitoring client."""
    
    def __init__(self, monitoring_client):
        """Initialize telemetry adapter with monitoring client."""
        self.monitoring_client = monitoring_client
        
        # History tracking for trend analysis
        self.prediction_history = deque(maxlen=1000)
        self.error_history = deque(maxlen=1000)
        self.action_strategy_history = deque(maxlen=100)
        self.uncertainty_history = deque(maxlen=100)
        self.self_modification_history = deque(maxlen=100)
        
        # Phase-specific metrics
        self.phase_metrics = {
            'phase1_sensory_prediction': {},
            'phase2_error_learning': {},
            'phase3_hierarchical': {},
            'phase4_action_strategies': {},
            'phase5_active_sensing': {}
        }
        
        # Initialize tracking
        self.last_telemetry = None
        self.telemetry_count = 0
        
    def get_evolved_telemetry(self) -> Optional[Dict]:
        """Get comprehensive telemetry data from the brain."""
        if not self.monitoring_client:
            return None
            
        try:
            # Get raw telemetry
            telemetry = self.monitoring_client.get_brain_state()
            if not telemetry:
                return None
                
            self.last_telemetry = telemetry
            self.telemetry_count += 1
            
            # Extract and enhance telemetry with prediction-specific metrics
            enhanced_telemetry = self._enhance_telemetry(telemetry)
            
            # Track history
            self._update_history(enhanced_telemetry)
            
            return enhanced_telemetry
            
        except Exception as e:
            print(f"Telemetry error: {e}")
            return None
    
    def _enhance_telemetry(self, telemetry: Dict) -> Dict:
        """Enhance raw telemetry with prediction-specific analysis."""
        enhanced = telemetry.copy()
        
        # Phase 1: Sensory Prediction Analysis
        if 'sensory_predictions' in telemetry:
            pred_data = telemetry['sensory_predictions']
            enhanced['phase1_sensory_prediction'] = {
                'per_sensor_confidence': pred_data.get('confidence_per_sensor', []),
                'specialized_regions': pred_data.get('specialized_regions', 0),
                'avg_prediction_accuracy': pred_data.get('accuracy', 0.0),
                'predictable_sensors': self._identify_predictable_sensors(pred_data)
            }
        
        # Phase 2: Error-Driven Learning Analysis
        if 'prediction_errors' in telemetry:
            error_data = telemetry['prediction_errors']
            enhanced['phase2_error_learning'] = {
                'error_magnitude': error_data.get('magnitude', 0.0),
                'self_modification_response': error_data.get('modification_strength', 0.0),
                'error_to_modification_ratio': self._calculate_error_response_ratio(error_data),
                'high_error_regions': error_data.get('high_error_count', 0)
            }
        
        # Phase 3: Hierarchical Timescales Analysis
        if 'hierarchical_predictions' in telemetry:
            hier_data = telemetry['hierarchical_predictions']
            enhanced['phase3_hierarchical'] = {
                'immediate_accuracy': hier_data.get('immediate_accuracy', 0.0),
                'short_term_accuracy': hier_data.get('short_term_accuracy', 0.0),
                'long_term_accuracy': hier_data.get('long_term_accuracy', 0.0),
                'abstract_patterns': hier_data.get('abstract_pattern_count', 0),
                'timescale_coherence': self._calculate_timescale_coherence(hier_data)
            }
        
        # Phase 4: Action Prediction Analysis
        if 'action_selection' in telemetry:
            action_data = telemetry['action_selection']
            enhanced['phase4_action_strategies'] = {
                'strategy': action_data.get('current_strategy', 'unknown'),
                'exploit_ratio': action_data.get('exploit_count', 0) / max(1, action_data.get('total_actions', 1)),
                'explore_ratio': action_data.get('explore_count', 0) / max(1, action_data.get('total_actions', 1)),
                'test_ratio': action_data.get('test_count', 0) / max(1, action_data.get('total_actions', 1)),
                'outcome_preview_accuracy': action_data.get('preview_accuracy', 0.0)
            }
        
        # Phase 5: Active Sensing Analysis
        if 'active_sensing' in telemetry:
            sensing_data = telemetry['active_sensing']
            enhanced['phase5_active_sensing'] = {
                'uncertainty_level': sensing_data.get('total_uncertainty', 0.0),
                'attention_pattern': sensing_data.get('current_pattern', 'unknown'),
                'saccade_count': sensing_data.get('saccade_count', 0),
                'smooth_pursuit_ratio': sensing_data.get('smooth_pursuit_time', 0.0),
                'information_gain': sensing_data.get('information_gain', 0.0)
            }
        
        return enhanced
    
    def _identify_predictable_sensors(self, pred_data: Dict) -> List[int]:
        """Identify which sensors have high prediction confidence."""
        confidences = pred_data.get('confidence_per_sensor', [])
        if not confidences:
            return []
        
        # Sensors with >60% confidence are considered predictable
        threshold = 0.6
        return [i for i, conf in enumerate(confidences) if conf > threshold]
    
    def _calculate_error_response_ratio(self, error_data: Dict) -> float:
        """Calculate how strongly self-modification responds to errors."""
        error_mag = error_data.get('magnitude', 0.0)
        mod_strength = error_data.get('modification_strength', 0.0)
        
        if error_mag > 0:
            return mod_strength / error_mag
        return 0.0
    
    def _calculate_timescale_coherence(self, hier_data: Dict) -> float:
        """Calculate coherence between prediction timescales."""
        immediate = hier_data.get('immediate_accuracy', 0.0)
        short_term = hier_data.get('short_term_accuracy', 0.0)
        long_term = hier_data.get('long_term_accuracy', 0.0)
        
        # Coherence is high when predictions flow smoothly across timescales
        if immediate > 0:
            coherence = 1.0 - abs(short_term - immediate) - abs(long_term - short_term)
            return max(0.0, coherence)
        return 0.0
    
    def _update_history(self, telemetry: Dict):
        """Update historical tracking for trend analysis."""
        # Track prediction accuracy
        if 'phase1_sensory_prediction' in telemetry:
            self.prediction_history.append({
                'time': time.time(),
                'accuracy': telemetry['phase1_sensory_prediction']['avg_prediction_accuracy'],
                'specialized_regions': telemetry['phase1_sensory_prediction']['specialized_regions']
            })
        
        # Track error magnitude
        if 'phase2_error_learning' in telemetry:
            self.error_history.append({
                'time': time.time(),
                'magnitude': telemetry['phase2_error_learning']['error_magnitude'],
                'modification': telemetry['phase2_error_learning']['self_modification_response']
            })
        
        # Track action strategies
        if 'phase4_action_strategies' in telemetry:
            self.action_strategy_history.append({
                'time': time.time(),
                'strategy': telemetry['phase4_action_strategies']['strategy']
            })
        
        # Track uncertainty
        if 'phase5_active_sensing' in telemetry:
            self.uncertainty_history.append({
                'time': time.time(),
                'uncertainty': telemetry['phase5_active_sensing']['uncertainty_level'],
                'pattern': telemetry['phase5_active_sensing']['attention_pattern']
            })
    
    def get_learning_metrics(self) -> Dict:
        """Calculate learning metrics from telemetry history."""
        metrics = {
            'learning_detected': False,
            'efficiency': 0.0,
            'self_modification_strength': 0.01,
            'working_memory_patterns': 0
        }
        
        if not self.last_telemetry:
            return metrics
        
        # Learning detection based on prediction improvement
        if len(self.prediction_history) >= 10:
            recent_accuracy = np.mean([h['accuracy'] for h in list(self.prediction_history)[-5:]])
            early_accuracy = np.mean([h['accuracy'] for h in list(self.prediction_history)[:5]])
            metrics['learning_detected'] = recent_accuracy > early_accuracy * 1.1
        
        # Efficiency based on error reduction
        if len(self.error_history) >= 10:
            recent_errors = np.mean([h['magnitude'] for h in list(self.error_history)[-5:]])
            early_errors = np.mean([h['magnitude'] for h in list(self.error_history)[:5]])
            if early_errors > 0:
                metrics['efficiency'] = max(0, min(1, (early_errors - recent_errors) / early_errors))
        else:
            # Default efficiency when not enough error history
            # Use a more realistic default than 0.0
            metrics['efficiency'] = 0.5
        
        # Extract current values
        if 'evolution_state' in self.last_telemetry:
            evo_state = self.last_telemetry['evolution_state']
            metrics['self_modification_strength'] = evo_state.get('self_modification_strength', 0.01)
            metrics['working_memory_patterns'] = evo_state.get('working_memory', {}).get('n_patterns', 0)
        
        return metrics
    
    def get_evolution_analysis(self) -> Dict:
        """Analyze brain evolution over time."""
        if len(self.self_modification_history) < 2:
            return {'status': 'insufficient_data'}
        
        history = list(self.self_modification_history)
        initial = history[0]
        final = history[-1]
        
        return {
            'initial_self_modification': initial.get('strength', 0.01),
            'final_self_modification': final.get('strength', 0.01),
            'self_modification_growth': final.get('strength', 0.01) - initial.get('strength', 0.01),
            'total_evolution_cycles': final.get('cycles', 0),
            'phase_specialization': self._analyze_phase_specialization()
        }
    
    def _analyze_phase_specialization(self) -> Dict:
        """Analyze how well each prediction phase has specialized."""
        specialization = {}
        
        # Phase 1: Sensory prediction specialization
        if len(self.prediction_history) > 0:
            recent = list(self.prediction_history)[-1]
            specialization['sensory_prediction'] = {
                'specialized_regions': recent.get('specialized_regions', 0),
                'accuracy': recent.get('accuracy', 0.0)
            }
        
        # Phase 4: Action strategy balance
        if len(self.action_strategy_history) >= 10:
            strategies = [h['strategy'] for h in self.action_strategy_history]
            unique_strategies = len(set(strategies))
            specialization['action_strategies'] = {
                'diversity': unique_strategies / 3.0,  # 3 possible strategies
                'dominant': max(set(strategies), key=strategies.count)
            }
        
        return specialization
    
    def get_comprehensive_report(self) -> Dict:
        """Generate comprehensive report of all prediction phases."""
        report = {
            'summary': {
                'telemetry_count': self.telemetry_count,
                'learning_detected': False,
                'phase_status': {}
            },
            'phase_analysis': {},
            'behavioral_analysis': {},
            'recommendations': []
        }
        
        # Get latest metrics
        learning_metrics = self.get_learning_metrics()
        evolution_analysis = self.get_evolution_analysis()
        
        report['summary']['learning_detected'] = learning_metrics['learning_detected']
        
        # Phase-by-phase analysis
        if self.last_telemetry:
            # Phase 1: Sensory Prediction
            if 'phase1_sensory_prediction' in self.last_telemetry:
                phase1 = self.last_telemetry['phase1_sensory_prediction']
                report['phase_analysis']['sensory_prediction'] = {
                    'status': 'active' if phase1['avg_prediction_accuracy'] > 0.1 else 'initializing',
                    'accuracy': phase1['avg_prediction_accuracy'],
                    'specialized_regions': phase1['specialized_regions'],
                    'predictable_sensors': len(phase1['predictable_sensors'])
                }
            
            # Phase 2: Error Learning
            if 'phase2_error_learning' in self.last_telemetry:
                phase2 = self.last_telemetry['phase2_error_learning']
                report['phase_analysis']['error_learning'] = {
                    'status': 'responsive' if phase2['error_to_modification_ratio'] > 0.5 else 'weak',
                    'error_magnitude': phase2['error_magnitude'],
                    'response_ratio': phase2['error_to_modification_ratio']
                }
            
            # Phase 3: Hierarchical
            if 'phase3_hierarchical' in self.last_telemetry:
                phase3 = self.last_telemetry['phase3_hierarchical']
                report['phase_analysis']['hierarchical'] = {
                    'immediate_accuracy': phase3['immediate_accuracy'],
                    'short_term_accuracy': phase3['short_term_accuracy'],
                    'long_term_accuracy': phase3['long_term_accuracy'],
                    'coherence': phase3['timescale_coherence']
                }
            
            # Phase 4: Action Strategies
            if 'phase4_action_strategies' in self.last_telemetry:
                phase4 = self.last_telemetry['phase4_action_strategies']
                report['phase_analysis']['action_strategies'] = {
                    'current': phase4['strategy'],
                    'balance': {
                        'exploit': phase4['exploit_ratio'],
                        'explore': phase4['explore_ratio'],
                        'test': phase4['test_ratio']
                    }
                }
            
            # Phase 5: Active Sensing
            if 'phase5_active_sensing' in self.last_telemetry:
                phase5 = self.last_telemetry['phase5_active_sensing']
                report['phase_analysis']['active_sensing'] = {
                    'uncertainty': phase5['uncertainty_level'],
                    'pattern': phase5['attention_pattern'],
                    'information_gain': phase5['information_gain']
                }
        
        # Behavioral analysis
        report['behavioral_analysis'] = self._analyze_behavior_patterns()
        
        # Generate recommendations
        report['recommendations'] = self._generate_recommendations(report)
        
        return report
    
    def _analyze_behavior_patterns(self) -> Dict:
        """Analyze emergent behavioral patterns."""
        patterns = {
            'dominant_behavior': 'unknown',
            'behavior_stability': 0.0,
            'exploration_tendency': 0.5
        }
        
        # Analyze action strategy patterns
        if len(self.action_strategy_history) >= 20:
            recent_strategies = [h['strategy'] for h in list(self.action_strategy_history)[-20:]]
            
            # Dominant behavior
            strategy_counts = {s: recent_strategies.count(s) for s in set(recent_strategies)}
            patterns['dominant_behavior'] = max(strategy_counts, key=strategy_counts.get)
            
            # Stability (how consistent is the behavior)
            patterns['behavior_stability'] = max(strategy_counts.values()) / len(recent_strategies)
            
            # Exploration tendency
            explore_count = recent_strategies.count('explore') + recent_strategies.count('test') * 0.5
            patterns['exploration_tendency'] = explore_count / len(recent_strategies)
        
        return patterns
    
    def _generate_recommendations(self, report: Dict) -> List[str]:
        """Generate recommendations based on analysis."""
        recommendations = []
        
        # Check sensory prediction
        if 'sensory_prediction' in report['phase_analysis']:
            phase1 = report['phase_analysis']['sensory_prediction']
            if phase1['accuracy'] < 0.2:
                recommendations.append("Low sensory prediction accuracy - consider more predictable input patterns")
            if phase1['specialized_regions'] < 2:
                recommendations.append("Few specialized regions - extend training time for specialization")
        
        # Check error learning
        if 'error_learning' in report['phase_analysis']:
            phase2 = report['phase_analysis']['error_learning']
            if phase2['response_ratio'] < 0.5:
                recommendations.append("Weak error response - check self-modification parameters")
        
        # Check action balance
        if 'action_strategies' in report['phase_analysis']:
            phase4 = report['phase_analysis']['action_strategies']
            balance = phase4['balance']
            if balance['explore'] < 0.1:
                recommendations.append("Low exploration - system may be stuck in local optima")
            if balance['exploit'] < 0.2:
                recommendations.append("Low exploitation - system not leveraging learned patterns")
        
        return recommendations