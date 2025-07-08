"""
Cognitive Development System.

Implements developmental stages based on emergent cognitive constraints rather
than arbitrary thresholds. Transitions happen when the brain hits actual
limitations in search depth, memory capacity, or learning rate.
"""

import time
import statistics
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum

from core.world_graph import WorldGraph
from .development_types import ActionProficiency, DevelopmentalStage


@dataclass
class CognitiveMetrics:
    """Tracks cognitive performance and constraints over time."""
    
    # Search and traversal metrics
    average_search_depth: float = 0.0
    max_achievable_depth: float = 0.0
    search_efficiency: float = 0.0  # Quality per unit time
    depth_decline_rate: float = 0.0  # How quickly depth is declining
    
    # Memory and capacity metrics
    memory_utilization: float = 0.0  # How much of brain capacity is used
    search_time_pressure: float = 0.0  # Time constraints on thinking
    connection_density: float = 0.0  # How interconnected the graph is
    
    # Learning and improvement metrics
    learning_rate: float = 0.0  # Rate of improvement in predictions
    skill_diversity: float = 0.0  # How many different skills being developed
    improvement_plateau: bool = False  # Whether learning has stagnated
    
    # Efficiency and optimization metrics
    behavioral_consistency: float = 0.0  # How stable behavior patterns are
    energy_efficiency: float = 0.0  # How efficiently using cognitive resources
    specialization_level: float = 0.0  # How specialized vs. generalist


@dataclass
class CognitiveConstraint:
    """Represents a specific cognitive limitation that triggers development."""
    constraint_type: str
    current_value: float
    threshold_value: float
    trend_direction: float  # Positive = improving, negative = declining
    severity: float  # How much this constraint is limiting performance
    description: str


class CognitiveDevelopmentSystem:
    """
    Manages developmental transitions based on emergent cognitive constraints
    rather than arbitrary thresholds.
    """
    
    def __init__(self, world_graph: WorldGraph):
        self.world_graph = world_graph
        
        # Cognitive metrics tracking
        self.cognitive_metrics = CognitiveMetrics()
        self.metrics_history = deque(maxlen=100)  # Rolling window of metrics
        
        # Development state
        self.current_stage = DevelopmentalStage.INFANCY
        self.time_in_current_stage = 0.0
        self.stage_transition_history = []
        
        # Constraint monitoring
        self.active_constraints = []
        self.constraint_thresholds = self._initialize_constraint_thresholds()
        
        # Performance tracking for constraint detection
        self.search_depth_history = deque(maxlen=50)
        self.learning_rate_history = deque(maxlen=30)
        self.efficiency_history = deque(maxlen=40)
        self.prediction_accuracy_history = deque(maxlen=60)
        
        # Timing
        self.last_update_time = time.time()
        self.development_start_time = time.time()
    
    def _initialize_constraint_thresholds(self) -> Dict[str, Dict]:
        """Initialize thresholds for different cognitive constraints."""
        return {
            # Search depth constraints
            'search_depth_decline': {
                'threshold': -0.1,  # 10% decline in search depth
                'severity_multiplier': 2.0,
                'triggers_stage': DevelopmentalStage.ADOLESCENCE,
                'description': 'Brain can no longer maintain deep search due to size/complexity'
            },
            
            # Memory capacity constraints  
            'memory_pressure': {
                'threshold': 0.30,  # 30% memory utilization (more sensitive)
                'severity_multiplier': 1.5,
                'triggers_stage': DevelopmentalStage.ADOLESCENCE,
                'description': 'Memory usage forcing more selective attention'
            },
            
            # Learning plateau constraints
            'learning_plateau': {
                'threshold': 0.01,  # Less than 1% improvement rate (more sensitive)
                'severity_multiplier': 1.8,
                'triggers_stage': DevelopmentalStage.EARLY_LEARNING,  # Trigger earlier stage
                'description': 'Learning rate has plateaued, pattern recognition begins'
            },
            
            # Efficiency optimization constraints
            'efficiency_pressure': {
                'threshold': 0.6,  # 60% time pressure
                'severity_multiplier': 1.3,
                'triggers_stage': DevelopmentalStage.MATURITY,
                'description': 'Time constraints require more efficient cognitive strategies'
            },
            
            # Specialization constraints
            'specialization_pressure': {
                'threshold': 0.8,  # 80% behavioral consistency
                'severity_multiplier': 1.2,
                'triggers_stage': DevelopmentalStage.EXPERTISE,
                'description': 'High specialization achieved, refinement phase begins'
            }
        }
    
    def update_cognitive_metrics(self, traversal_result=None, prediction_accuracy: float = None,
                               thinking_time: float = None, action_taken: Dict = None):
        """Update cognitive metrics based on recent brain activity."""
        current_time = time.time()
        self.time_in_current_stage = current_time - self.last_update_time
        
        # Update search and traversal metrics
        if traversal_result:
            self._update_search_metrics(traversal_result, thinking_time)
        
        # Update learning metrics
        if prediction_accuracy is not None:
            self._update_learning_metrics(prediction_accuracy)
        
        # Update memory and capacity metrics
        self._update_capacity_metrics()
        
        # Update behavioral metrics
        if action_taken:
            self._update_behavioral_metrics(action_taken)
        
        # Calculate overall cognitive state
        self._calculate_cognitive_state()
        
        # Store metrics history
        self.metrics_history.append({
            'timestamp': current_time,
            'stage': self.current_stage,
            'metrics': CognitiveMetrics(**self.cognitive_metrics.__dict__)
        })
        
        # Check for developmental transitions
        self._check_for_stage_transitions()
        
        self.last_update_time = current_time
    
    def _update_search_metrics(self, traversal_result, thinking_time: float):
        """Update metrics related to search depth and efficiency."""
        if hasattr(traversal_result, 'steps_taken'):
            search_depth = traversal_result.steps_taken
            self.search_depth_history.append(search_depth)
            
            # Calculate average search depth
            if self.search_depth_history:
                self.cognitive_metrics.average_search_depth = statistics.mean(self.search_depth_history)
                
                # Detect search depth decline
                if len(self.search_depth_history) >= 20:
                    early_depths = list(self.search_depth_history)[:10]
                    recent_depths = list(self.search_depth_history)[-10:]
                    
                    early_avg = statistics.mean(early_depths)
                    recent_avg = statistics.mean(recent_depths)
                    
                    if early_avg > 0:
                        self.cognitive_metrics.depth_decline_rate = (recent_avg - early_avg) / early_avg
        
        # Calculate search efficiency (quality per unit time)
        if thinking_time and hasattr(traversal_result, 'prediction'):
            quality = self._estimate_result_quality(traversal_result)
            efficiency = quality / max(0.001, thinking_time)
            self.efficiency_history.append(efficiency)
            
            if self.efficiency_history:
                self.cognitive_metrics.search_efficiency = statistics.mean(self.efficiency_history)
    
    def _update_learning_metrics(self, prediction_accuracy: float):
        """Update metrics related to learning and improvement."""
        self.prediction_accuracy_history.append(prediction_accuracy)
        
        # Calculate learning rate (trend in prediction accuracy)
        if len(self.prediction_accuracy_history) >= 10:
            recent_accuracies = list(self.prediction_accuracy_history)[-10:]
            
            # Simple linear trend calculation
            n = len(recent_accuracies)
            x_mean = (n - 1) / 2
            y_mean = statistics.mean(recent_accuracies)
            
            numerator = sum((i - x_mean) * (acc - y_mean) for i, acc in enumerate(recent_accuracies))
            denominator = sum((i - x_mean) ** 2 for i in range(n))
            
            if denominator > 0:
                learning_rate = numerator / denominator
                self.learning_rate_history.append(learning_rate)
                
                if self.learning_rate_history:
                    self.cognitive_metrics.learning_rate = statistics.mean(self.learning_rate_history)
                    
                    # Detect learning plateau
                    if len(self.learning_rate_history) >= 5:
                        recent_learning_rates = list(self.learning_rate_history)[-5:]
                        avg_recent_rate = statistics.mean(recent_learning_rates)
                        self.cognitive_metrics.improvement_plateau = avg_recent_rate < 0.005  # Very slow learning
    
    def _update_capacity_metrics(self):
        """Update metrics related to memory and cognitive capacity."""
        if self.world_graph:
            # Memory utilization (nodes vs. some reasonable maximum)
            node_count = self.world_graph.node_count()
            estimated_max_capacity = 10000  # Could be dynamic based on system
            self.cognitive_metrics.memory_utilization = min(1.0, node_count / estimated_max_capacity)
            
            # Connection density (how interconnected the graph is)
            if node_count > 0:
                # This is a simplified metric - could be more sophisticated
                avg_connections_per_node = self._calculate_average_connections()
                max_possible_connections = min(node_count - 1, 50)  # Reasonable maximum
                self.cognitive_metrics.connection_density = min(1.0, avg_connections_per_node / max_possible_connections)
            
            # Time pressure (as graph grows, search takes longer)
            if hasattr(self, 'recent_thinking_times') and self.recent_thinking_times:
                avg_thinking_time = statistics.mean(self.recent_thinking_times)
                max_acceptable_time = 0.2  # 200ms threshold
                self.cognitive_metrics.search_time_pressure = min(1.0, avg_thinking_time / max_acceptable_time)
    
    def _update_behavioral_metrics(self, action_taken: Dict):
        """Update metrics related to behavioral patterns and consistency."""
        # This would track behavioral consistency over time
        # For now, placeholder - could be enhanced with actual behavior analysis
        pass
    
    def _calculate_cognitive_state(self):
        """Calculate overall cognitive state from individual metrics."""
        # This synthesizes all metrics into overall cognitive assessment
        # Could be enhanced with more sophisticated analysis
        pass
    
    def _calculate_average_connections(self) -> float:
        """Calculate average number of connections per node."""
        if not self.world_graph.has_nodes():
            return 0.0
        
        all_nodes = self.world_graph.all_nodes()
        total_connections = 0
        
        for node in all_nodes:
            # Count significant connections
            significant_connections = sum(1 for weight in node.connection_weights.values() if weight > 0.3)
            total_connections += significant_connections
        
        return total_connections / len(all_nodes) if all_nodes else 0.0
    
    def _estimate_result_quality(self, traversal_result) -> float:
        """Estimate the quality of a traversal result."""
        # Simplified quality estimation - could be enhanced
        if hasattr(traversal_result, 'prediction') and traversal_result.prediction:
            return getattr(traversal_result.prediction, 'confidence_level', 0.5)
        return 0.5
    
    def _check_for_stage_transitions(self):
        """Check if cognitive constraints warrant a developmental stage transition."""
        # Detect active constraints
        self.active_constraints = self._detect_active_constraints()
        
        # Check if any constraints trigger stage transitions
        for constraint in self.active_constraints:
            triggered_stage = self._get_constraint_triggered_stage(constraint)
            
            if triggered_stage and self._should_transition_to_stage(triggered_stage):
                self._transition_to_stage(triggered_stage, constraint)
                break  # Only one transition per update
    
    def _detect_active_constraints(self) -> List[CognitiveConstraint]:
        """Detect cognitive constraints that are currently limiting performance."""
        constraints = []
        
        # Search depth decline constraint
        if self.cognitive_metrics.depth_decline_rate < self.constraint_thresholds['search_depth_decline']['threshold']:
            severity = abs(self.cognitive_metrics.depth_decline_rate) * 2.0
            constraints.append(CognitiveConstraint(
                constraint_type='search_depth_decline',
                current_value=self.cognitive_metrics.depth_decline_rate,
                threshold_value=self.constraint_thresholds['search_depth_decline']['threshold'],
                trend_direction=self.cognitive_metrics.depth_decline_rate,
                severity=severity,
                description=self.constraint_thresholds['search_depth_decline']['description']
            ))
        
        # Memory pressure constraint
        if self.cognitive_metrics.memory_utilization > self.constraint_thresholds['memory_pressure']['threshold']:
            severity = (self.cognitive_metrics.memory_utilization - self.constraint_thresholds['memory_pressure']['threshold']) * 2.0
            constraints.append(CognitiveConstraint(
                constraint_type='memory_pressure',
                current_value=self.cognitive_metrics.memory_utilization,
                threshold_value=self.constraint_thresholds['memory_pressure']['threshold'],
                trend_direction=1.0,  # Always increasing with more memories
                severity=severity,
                description=self.constraint_thresholds['memory_pressure']['description']
            ))
        
        # Learning plateau constraint
        if self.cognitive_metrics.improvement_plateau:
            severity = 1.0 - max(0.0, self.cognitive_metrics.learning_rate)
            constraints.append(CognitiveConstraint(
                constraint_type='learning_plateau',
                current_value=self.cognitive_metrics.learning_rate,
                threshold_value=self.constraint_thresholds['learning_plateau']['threshold'],
                trend_direction=self.cognitive_metrics.learning_rate,
                severity=severity,
                description=self.constraint_thresholds['learning_plateau']['description']
            ))
        
        # Time pressure constraint
        if self.cognitive_metrics.search_time_pressure > self.constraint_thresholds['efficiency_pressure']['threshold']:
            severity = (self.cognitive_metrics.search_time_pressure - self.constraint_thresholds['efficiency_pressure']['threshold']) * 1.5
            constraints.append(CognitiveConstraint(
                constraint_type='efficiency_pressure',
                current_value=self.cognitive_metrics.search_time_pressure,
                threshold_value=self.constraint_thresholds['efficiency_pressure']['threshold'],
                trend_direction=1.0,  # Time pressure generally increases
                severity=severity,
                description=self.constraint_thresholds['efficiency_pressure']['description']
            ))
        
        # Behavioral consistency constraint (for expertise transition)
        if self.cognitive_metrics.behavioral_consistency > self.constraint_thresholds['specialization_pressure']['threshold']:
            severity = self.cognitive_metrics.behavioral_consistency
            constraints.append(CognitiveConstraint(
                constraint_type='specialization_pressure',
                current_value=self.cognitive_metrics.behavioral_consistency,
                threshold_value=self.constraint_thresholds['specialization_pressure']['threshold'],
                trend_direction=1.0,  # Consistency generally increases
                severity=severity,
                description=self.constraint_thresholds['specialization_pressure']['description']
            ))
        
        return constraints
    
    def _get_constraint_triggered_stage(self, constraint: CognitiveConstraint) -> Optional[DevelopmentalStage]:
        """Get the developmental stage that a constraint should trigger."""
        if constraint.constraint_type in self.constraint_thresholds:
            return self.constraint_thresholds[constraint.constraint_type].get('triggers_stage')
        return None
    
    def _should_transition_to_stage(self, target_stage: DevelopmentalStage) -> bool:
        """Determine if we should transition to the target stage."""
        # Only allow forward transitions
        stage_order = [
            DevelopmentalStage.INFANCY,
            DevelopmentalStage.EARLY_LEARNING,
            DevelopmentalStage.ADOLESCENCE,
            DevelopmentalStage.MATURITY,
            DevelopmentalStage.EXPERTISE
        ]
        
        try:
            current_index = stage_order.index(self.current_stage)
            target_index = stage_order.index(target_stage)
            
            # Allow advancement to next stage or beyond if constraints are severe enough
            return target_index > current_index
        except ValueError:
            return False
    
    def _transition_to_stage(self, new_stage: DevelopmentalStage, triggering_constraint: CognitiveConstraint):
        """Perform transition to new developmental stage."""
        old_stage = self.current_stage
        self.current_stage = new_stage
        
        transition_record = {
            'from_stage': old_stage.value,
            'to_stage': new_stage.value,
            'trigger_constraint': triggering_constraint.constraint_type,
            'constraint_description': triggering_constraint.description,
            'constraint_severity': triggering_constraint.severity,
            'transition_time': time.time(),
            'time_in_previous_stage': self.time_in_current_stage,
            'cognitive_state_at_transition': CognitiveMetrics(**self.cognitive_metrics.__dict__)
        }
        
        self.stage_transition_history.append(transition_record)
        
        print(f"🧠 Cognitive development: {old_stage.value} → {new_stage.value}")
        print(f"   Triggered by: {triggering_constraint.description}")
        print(f"   Constraint severity: {triggering_constraint.severity:.2f}")
        print(f"   Time in previous stage: {self.time_in_current_stage:.1f}s")
    
    def get_development_status(self) -> Dict[str, Any]:
        """Get comprehensive cognitive development status."""
        return {
            'current_stage': self.current_stage.value,
            'time_in_current_stage': self.time_in_current_stage,
            'total_development_time': time.time() - self.development_start_time,
            'cognitive_metrics': {
                'search_depth': self.cognitive_metrics.average_search_depth,
                'depth_decline_rate': self.cognitive_metrics.depth_decline_rate,
                'memory_utilization': self.cognitive_metrics.memory_utilization,
                'connection_density': self.cognitive_metrics.connection_density,
                'learning_rate': self.cognitive_metrics.learning_rate,
                'improvement_plateau': self.cognitive_metrics.improvement_plateau,
                'search_efficiency': self.cognitive_metrics.search_efficiency,
                'time_pressure': self.cognitive_metrics.search_time_pressure
            },
            'active_constraints': [
                {
                    'type': c.constraint_type,
                    'severity': c.severity,
                    'description': c.description,
                    'current_value': c.current_value,
                    'threshold': c.threshold_value
                }
                for c in self.active_constraints
            ],
            'stage_transitions': self.stage_transition_history,
            'constraint_based_development': True
        }
    
    def get_stage_progression_parameters(self) -> Dict[str, float]:
        """Get current parameters for action generation based on cognitive development."""
        # Base parameters for each stage
        stage_configs = {
            DevelopmentalStage.INFANCY: {
                'exploration_rate': 0.90,
                'proficiency_bias_strength': 0.05,
                'skill_building_rate': 0.05,
                'description': 'High exploration, minimal proficiency bias'
            },
            DevelopmentalStage.EARLY_LEARNING: {
                'exploration_rate': 0.75,
                'proficiency_bias_strength': 0.15,
                'skill_building_rate': 0.10,
                'description': 'Beginning to notice patterns, moderate exploration'
            },
            DevelopmentalStage.ADOLESCENCE: {
                'exploration_rate': 0.50,
                'proficiency_bias_strength': 0.35,
                'skill_building_rate': 0.15,
                'description': 'Balancing exploration with emerging competencies'
            },
            DevelopmentalStage.MATURITY: {
                'exploration_rate': 0.25,
                'proficiency_bias_strength': 0.60,
                'skill_building_rate': 0.15,
                'description': 'Competent behavior with selective exploration'
            },
            DevelopmentalStage.EXPERTISE: {
                'exploration_rate': 0.15,
                'proficiency_bias_strength': 0.75,
                'skill_building_rate': 0.10,
                'description': 'Expert specialization with minimal exploration'
            }
        }
        
        base_config = stage_configs.get(self.current_stage, stage_configs[DevelopmentalStage.INFANCY])
        
        # Modify parameters based on active constraints
        adjusted_config = base_config.copy()
        
        for constraint in self.active_constraints:
            if constraint.constraint_type == 'memory_pressure':
                # Memory pressure increases proficiency bias (less exploration)
                adjusted_config['exploration_rate'] *= (1.0 - constraint.severity * 0.2)
                adjusted_config['proficiency_bias_strength'] *= (1.0 + constraint.severity * 0.3)
                
            elif constraint.constraint_type == 'efficiency_pressure':
                # Time pressure increases proficiency bias
                adjusted_config['proficiency_bias_strength'] *= (1.0 + constraint.severity * 0.2)
                adjusted_config['exploration_rate'] *= (1.0 - constraint.severity * 0.15)
                
            elif constraint.constraint_type == 'learning_plateau':
                # Learning plateau might increase exploration slightly
                adjusted_config['exploration_rate'] *= (1.0 + constraint.severity * 0.1)
        
        # Ensure parameters stay within valid ranges
        adjusted_config['exploration_rate'] = max(0.05, min(0.95, adjusted_config['exploration_rate']))
        adjusted_config['proficiency_bias_strength'] = max(0.05, min(0.90, adjusted_config['proficiency_bias_strength']))
        adjusted_config['skill_building_rate'] = max(0.05, min(0.30, adjusted_config['skill_building_rate']))
        
        return adjusted_config