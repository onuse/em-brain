"""
Brain Interface - Adaptive bridge between world-agnostic brain and environment-specific brainstem.
Handles dynamic sensory vector length adaptation and experience creation.
"""

from typing import List, Dict, Optional, Any
from core.world_graph import WorldGraph
from core.experience_node import ExperienceNode
from core.communication import PredictionPacket, SensoryPacket
from core.adaptive_tuning import AdaptiveParameterTuner
from core.persistent_memory import PersistentMemoryManager
from core.actuator_discovery import UniversalActuatorDiscovery
from core.novelty_detection import NoveltyDetector, ExperienceSignature
from core.node_consolidation import NodeConsolidationEngine
from predictor.triple_predictor import TriplePredictor
from predictor.consensus_resolver import ConsensusResult
from brain_prediction_profiler import get_brain_profiler, profile_section


class BrainInterface:
    """
    Adaptive interface between the brain and brainstem.
    
    The brain is completely world-agnostic and works with arbitrary vector lengths.
    This interface learns the sensory dimensions from the brainstem and adapts accordingly.
    """
    
    def __init__(self, predictor: TriplePredictor, memory_path: str = "./robot_memory", 
                 enable_persistence: bool = True):
        """
        Initialize brain interface.
        
        Args:
            predictor: The world-agnostic prediction engine
            memory_path: Path for persistent memory storage
            enable_persistence: Whether to enable persistent memory
        """
        self.predictor = predictor
        self.world_graph = WorldGraph()
        
        # Adaptive parameter tuning system
        self.adaptive_tuner = AdaptiveParameterTuner()
        
        # Universal actuator discovery system
        self.actuator_discovery = UniversalActuatorDiscovery()
        
        # Novelty detection and consolidation systems
        self.novelty_detector = NoveltyDetector(self.world_graph)
        self.consolidation_engine = NodeConsolidationEngine(self.world_graph)
        
        # Persistent memory system
        self.enable_persistence = enable_persistence
        if enable_persistence:
            self.memory_manager = PersistentMemoryManager(memory_path)
            self._load_persistent_state()
        else:
            self.memory_manager = None
        
        # Adaptive parameters learned from brainstem
        self.sensory_vector_length: Optional[int] = None
        self.motor_action_keys: Optional[List[str]] = None
        
        # Experience tracking
        self.last_prediction: Optional[PredictionPacket] = None
        self.last_sensory: Optional[SensoryPacket] = None
        
    def process_sensory_input(self, sensory_packet: SensoryPacket, 
                            mental_context: List[float], 
                            threat_level: str = "normal") -> PredictionPacket:
        """
        Process sensory input and generate next prediction.
        Automatically adapts to sensory vector length.
        
        Args:
            sensory_packet: Current sensor readings from brainstem
            mental_context: Current mental state (position, energy, etc.)
            threat_level: Threat assessment for time budgeting
            
        Returns:
            Prediction packet with motor actions and expected sensory outcome
        """
        with get_brain_profiler().start_prediction_timing():
            # Learn sensory dimensions from first encounter
            with profile_section("sensory_length_learning"):
                if self.sensory_vector_length is None:
                    self.sensory_vector_length = len(sensory_packet.sensor_values)
                    print(f"Brain learned sensory vector length: {self.sensory_vector_length}")
                
                # Validate consistent sensory length (brainstem should be consistent)
                if len(sensory_packet.sensor_values) != self.sensory_vector_length:
                    raise ValueError(f"Inconsistent sensory length: expected {self.sensory_vector_length}, "
                                   f"got {len(sensory_packet.sensor_values)}")
            
            # Create experience from previous prediction (if any)
            with profile_section("experience_creation"):
                if self.last_prediction and self.last_sensory:
                    self._create_experience_from_prediction()
            
            # Extract ACTUAL survival state from sensory data (indices 17-21 are internal state)
            with profile_section("survival_state_extraction"):
                actual_health = sensory_packet.sensor_values[17] if len(sensory_packet.sensor_values) > 17 else 1.0
                actual_energy = sensory_packet.sensor_values[18] if len(sensory_packet.sensor_values) > 18 else 1.0
            
            # Generate new prediction with ACTUAL survival state
            with profile_section("prediction_generation"):
                consensus_result = self.predictor.generate_prediction(
                    mental_context, self.world_graph, 
                    sensory_packet.sequence_id, threat_level,
                    robot_health=actual_health,  # Pass real health
                    robot_energy=actual_energy   # Pass real energy
                )
            
            # Ensure prediction has correct sensory vector length
            with profile_section("prediction_finalization"):
                if consensus_result.prediction:
                    consensus_result.prediction.set_sensory_vector_length(
                        self.sensory_vector_length, default_value=0.0
                    )
                
                # Store for next experience creation
                self.last_prediction = consensus_result.prediction
                self.last_sensory = sensory_packet
            
            return consensus_result.prediction
    
    def _create_experience_from_prediction(self):
        """Create experience node from previous prediction and current sensory reality."""
        with profile_section("experience_creation_validation"):
            if not self.last_prediction or not self.last_sensory:
                return
        
        # Calculate prediction error
        with profile_section("prediction_error_calculation"):
            prediction_error = self._calculate_prediction_error(
                self.last_prediction.expected_sensory,
                self.last_sensory.sensor_values
            )
        
        # Use stored mental context from the last prediction
        # Use the first 8 sensor values as mental context (robot's perception of its environment)
        with profile_section("mental_context_extraction"):
            mental_context = self.last_sensory.sensor_values[:8] if len(self.last_sensory.sensor_values) >= 8 else self.last_sensory.sensor_values
            
            # Extract ACTUAL survival state from sensory data (indices 17-21 are internal state)
            actual_health = self.last_sensory.sensor_values[17] if len(self.last_sensory.sensor_values) > 17 else 1.0
            actual_energy = self.last_sensory.sensor_values[18] if len(self.last_sensory.sensor_values) > 18 else 1.0
        
        # Create experience signature for novelty detection
        with profile_section("experience_signature_creation"):
            experience_signature = ExperienceSignature(
                mental_context=mental_context,
                motor_action=self.last_prediction.motor_action or {},
                sensory_outcome=dict(enumerate(self.last_sensory.sensor_values)),
                prediction_accuracy=1.0 - prediction_error,  # Convert error to accuracy
                temporal_context=mental_context[-5:] if len(mental_context) > 5 else mental_context,
                drive_states={}  # Could be populated from drive system
            )
            
            # Update novelty detector's context history
            self.novelty_detector.add_to_context_history(mental_context)
        
        # Evaluate novelty of this experience
        with profile_section("novelty_evaluation"):
            novelty_score = self.novelty_detector.evaluate_experience_novelty(experience_signature)
        
        # Consolidate experience based on novelty (instead of always creating new node)
        with profile_section("experience_consolidation"):
            consolidation_result = self.consolidation_engine.consolidate_experience(
                experience_signature=experience_signature,
                novelty_score=novelty_score,
                prediction_success=(prediction_error < 0.5)  # Consider low error as success
            )
            
            # Get the final experience node (either new or consolidated)
            experience = consolidation_result.target_node
        
        # Let drives evaluate experience significance AND pain/pleasure (proper separation of concerns)
        with profile_section("drive_evaluation"):
            if hasattr(self.predictor, 'motivation_system'):
                with profile_section("drive_context_creation"):
                    context = self._create_drive_context(actual_health, actual_energy)
                    total_pain = 0.0
                
                with profile_section("drive_experience_evaluation"):
                    for drive in self.predictor.motivation_system.drives.values():
                        # Evaluate significance for memory strengthening
                        significance = drive.evaluate_experience_significance(experience, context)
                        if significance > 1.0:
                            # Drive considers this experience significant - adjust strength
                            experience.strength *= significance
                        
                        # Evaluate pain/pleasure for learning bias
                        valence = drive.evaluate_experience_valence(experience, context)
                        total_pain += valence
                
                with profile_section("pain_pleasure_processing"):
                    # Apply pain bias to experience - painful experiences should be more memorable
                    # and create negative association with actions that caused them
                    if total_pain < -0.1:  # Significant pain
                        # Make painful experiences much more memorable
                        pain_multiplier = min(5.0, abs(total_pain) * 3.0)
                        experience.strength *= pain_multiplier
                        
                        # Store pain signal for future action evaluation (this creates avoidance)
                        if not hasattr(experience, 'pain_signal'):
                            experience.pain_signal = total_pain
                            
                    elif total_pain > 0.1:  # Significant pleasure
                        # Make pleasurable experiences more memorable too
                        pleasure_multiplier = min(3.0, total_pain * 2.0)
                        experience.strength *= pleasure_multiplier
                        
                        # Store pleasure signal for future action evaluation (this creates approach)
                        if not hasattr(experience, 'pleasure_signal'):
                            experience.pleasure_signal = total_pain
        
        # Add to world graph (only if it's a new node)
        with profile_section("world_graph_add_node"):
            if consolidation_result.strategy_used.value == "create_new":
                self.world_graph.add_node(experience)
            # For other strategies, the node is already in the graph and has been modified
        
        # Adaptive parameter tuning based on prediction error and sensory characteristics
        with profile_section("adaptive_parameter_tuning"):
            adapted_params = self.adaptive_tuner.adapt_parameters_from_prediction_error(
                prediction_error=prediction_error,
                sensory_vector=self.last_sensory.sensor_values,
                context={
                    'sequence_id': self.last_sensory.sequence_id,
                    'mental_context': mental_context,
                    'action_taken': self.last_prediction.motor_action
                }
            )
            
            # Apply adapted parameters to world graph and predictor
            self._apply_adapted_parameters(adapted_params)
        
        # Universal actuator discovery - learn actuator effects from experience
        with profile_section("actuator_discovery"):
            actuator_discovery_result = self.actuator_discovery.observe_actuator_effects(
                actuator_commands=self.last_prediction.motor_action,
                sensory_reading=self.last_sensory.sensor_values
            )
        
        # Update memory pressure for novelty detection
        with profile_section("memory_pressure_update"):
            self.novelty_detector.update_memory_pressure(
                self.world_graph.node_count(), 
                max_nodes=10000  # Could be configurable
            )
            
            # Optimize consolidation parameters periodically
            if self.consolidation_engine.consolidations_performed % 50 == 0:
                self.consolidation_engine.optimize_consolidation_parameters()
        
        # Update exploration rate based on prediction error (legacy)
        with profile_section("exploration_rate_update"):
            self._update_exploration_rate(prediction_error)
        
    def _calculate_prediction_error(self, predicted: List[float], actual: List[float]) -> float:
        """Calculate prediction error between predicted and actual sensory values."""
        if len(predicted) != len(actual):
            return float('inf')  # Major error if lengths don't match
            
        if len(predicted) == 0:
            return 0.0  # No prediction means no error
            
        # Simple Euclidean distance
        squared_diffs = [(p - a) ** 2 for p, a in zip(predicted, actual)]
        return (sum(squared_diffs) / len(squared_diffs)) ** 0.5
    
    def _create_drive_context(self, actual_health: float, actual_energy: float):
        """Create drive context for experience evaluation."""
        from drives.base_drive import DriveContext
        
        return DriveContext(
            current_sensory=self.last_sensory.sensor_values,
            robot_health=actual_health,
            robot_energy=actual_energy,
            robot_position=(0, 0),  # Simplified for now
            robot_orientation=0,
            recent_experiences=[],
            prediction_errors=[],
            time_since_last_food=5,  # Will be calculated properly by drives
            time_since_last_damage=10,
            threat_level="normal",
            step_count=0
        )
    
    
    def _update_exploration_rate(self, prediction_error: float):
        """Adaptively update exploration rate based on prediction performance."""
        # If prediction error is high, increase exploration
        # If prediction error is low, decrease exploration (exploit good predictions)
        if hasattr(self.predictor, 'exploration_rate'):
            current_rate = self.predictor.exploration_rate
            
            # Adaptive scaling: high error -> more exploration
            if prediction_error > 0.5:  # High error threshold
                new_rate = min(0.8, current_rate + 0.05)  # Increase exploration
            elif prediction_error < 0.1:  # Low error threshold  
                new_rate = max(0.1, current_rate - 0.02)  # Decrease exploration
            else:
                new_rate = current_rate  # Keep current rate
            
            # Update predictor's exploration rate
            if hasattr(self.predictor, 'single_traversal') and hasattr(self.predictor.single_traversal, 'curiosity_predictor'):
                self.predictor.single_traversal.curiosity_predictor.update_exploration_rate(new_rate)
                self.predictor.exploration_rate = new_rate
    
    def _apply_adapted_parameters(self, adapted_params: Dict[str, float]):
        """Apply adapted parameters to world graph and predictor systems."""
        # Apply memory/graph parameters to world graph
        if hasattr(self.world_graph, 'similarity_threshold'):
            self.world_graph.similarity_threshold = adapted_params.get('similarity_threshold', 
                                                                       self.world_graph.similarity_threshold)
        
        if hasattr(self.world_graph, 'activation_spread_iterations'):
            self.world_graph.activation_spread_iterations = int(adapted_params.get('activation_spread_iterations', 
                                                                                  self.world_graph.activation_spread_iterations))
        
        if hasattr(self.world_graph, 'consolidation_frequency'):
            self.world_graph.consolidation_frequency = int(adapted_params.get('consolidation_frequency', 
                                                                             self.world_graph.consolidation_frequency))
        
        if hasattr(self.world_graph, 'connection_learning_rate'):
            self.world_graph.connection_learning_rate = adapted_params.get('connection_learning_rate', 
                                                                          self.world_graph.connection_learning_rate)
        
        # Apply prediction parameters to predictor
        if hasattr(self.predictor, 'exploration_rate'):
            self.predictor.exploration_rate = adapted_params.get('exploration_rate', 
                                                               self.predictor.exploration_rate)
        
        # Apply time budget adaptation
        if hasattr(self.predictor, 'base_time_budget'):
            self.predictor.base_time_budget = adapted_params.get('time_budget_base', 
                                                               self.predictor.base_time_budget)
        
        # Apply to specific predictor components if they exist
        if hasattr(self.predictor, 'single_traversal'):
            if hasattr(self.predictor.single_traversal, 'curiosity_predictor'):
                curiosity_predictor = self.predictor.single_traversal.curiosity_predictor
                if hasattr(curiosity_predictor, 'update_exploration_rate'):
                    curiosity_predictor.update_exploration_rate(
                        adapted_params.get('exploration_rate', 0.3)
                    )
    
    def get_brain_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about brain state and learning."""
        stats = {
            "graph_stats": self.world_graph.get_graph_statistics(),
            "interface_stats": {
                "sensory_vector_length": self.sensory_vector_length,
                "total_experiences": self.world_graph.get_graph_statistics()["total_nodes"],
                "is_learning": self.sensory_vector_length is not None,
                "persistence_enabled": self.enable_persistence
            },
            "adaptive_tuning_stats": self.adaptive_tuner.get_adaptation_statistics(),
            "persistent_memory_stats": self.get_memory_statistics() if self.enable_persistence else {},
            "actuator_discovery_stats": self.actuator_discovery.get_discovery_statistics(),
            "novelty_detection_stats": self.novelty_detector.get_novelty_stats(),
            "consolidation_stats": self.consolidation_engine.get_consolidation_stats()
        }
        
        # Get predictor statistics (handle both old and new predictor types)
        if hasattr(self.predictor, 'get_predictor_statistics'):
            stats["predictor_stats"] = self.predictor.get_predictor_statistics()
        elif hasattr(self.predictor, 'get_prediction_statistics'):
            stats["predictor_stats"] = self.predictor.get_prediction_statistics()
        else:
            stats["predictor_stats"] = {}
        
        return stats
    
    def reset_brain(self):
        """Reset the brain to initial state (keeping learned dimensions)."""
        self.world_graph = WorldGraph()
        self.predictor.reset_statistics()
        self.last_prediction = None
        self.last_sensory = None
        
        # Reset novelty detection and consolidation systems
        self.novelty_detector.reset_session_stats()
        self.consolidation_engine.reset_session_stats()
        # Keep sensory_vector_length - brain remembers its interface
    
    def get_world_graph(self) -> WorldGraph:
        """Get the current world graph (for visualization)."""
        return self.world_graph
    
    def get_predictor(self) -> TriplePredictor:
        """Get the prediction engine (for configuration)."""
        return self.predictor
    
    # Persistent Memory Methods
    
    def start_memory_session(self, session_summary: str = "Robot brain session") -> Optional[str]:
        """Start a new persistent memory session."""
        if not self.enable_persistence:
            return None
        
        session_id = self.memory_manager.start_new_session(session_summary)
        print(f"Started brain memory session: {session_id}")
        return session_id
    
    def save_current_state(self) -> Optional[Dict[str, str]]:
        """Save current brain state to persistent memory."""
        if not self.enable_persistence:
            return None
        
        try:
            graph_path = self.memory_manager.save_world_graph(self.world_graph)
            params_path = self.memory_manager.save_adaptive_parameters(self.adaptive_tuner)
            
            return {
                'graph_path': graph_path,
                'params_path': params_path,
                'experiences_count': self.world_graph.node_count()
            }
        except Exception as e:
            print(f"Error saving brain state: {e}")
            return None
    
    def end_memory_session(self) -> Optional[Dict[str, Any]]:
        """End current memory session and save final state."""
        if not self.enable_persistence:
            return None
        
        try:
            return self.memory_manager.end_session(self.world_graph, self.adaptive_tuner)
        except Exception as e:
            print(f"Error ending memory session: {e}")
            return None
    
    def _load_persistent_state(self):
        """Load previous brain state from persistent memory."""
        try:
            # Load latest world graph
            loaded_graph = self.memory_manager.load_latest_world_graph()
            if loaded_graph:
                self.world_graph = loaded_graph
                print(f"Loaded {self.world_graph.node_count()} experiences from persistent memory")
            
            # Load latest adaptive parameters
            adaptive_data = self.memory_manager.load_latest_adaptive_parameters()
            if adaptive_data:
                # Restore current parameters
                if 'current_parameters' in adaptive_data:
                    self.adaptive_tuner.current_parameters = adaptive_data['current_parameters']
                
                # Restore adaptation statistics
                metadata = adaptive_data.get('metadata', {})
                self.adaptive_tuner.total_adaptations = metadata.get('total_adaptations', 0)
                self.adaptive_tuner.successful_adaptations = metadata.get('successful_adaptations', 0)
                
                print(f"Loaded adaptive parameters ({self.adaptive_tuner.total_adaptations} adaptations)")
        
        except Exception as e:
            print(f"Error loading persistent state: {e}")
            print("Starting with fresh brain state")
    
    def get_memory_statistics(self) -> Dict[str, Any]:
        """Get persistent memory statistics."""
        if not self.enable_persistence:
            return {"persistence_enabled": False}
        
        return self.memory_manager.get_memory_statistics()
    
    def search_similar_experiences(self, context: List[float], 
                                 similarity_threshold: float = 0.7,
                                 limit: int = 10) -> List[str]:
        """Search for similar experiences in persistent memory."""
        if not self.enable_persistence:
            return []
        
        try:
            results = self.memory_manager.search_experiences_by_context(
                context, similarity_threshold, limit
            )
            return [node_id for node_id, similarity in results]
        except Exception as e:
            print(f"Error searching experiences: {e}")
            return []
    
    def get_archived_experiences(self, experience_type: str, limit: int = 100) -> List[str]:
        """Get archived experiences by type (high_importance, spatial, skills, mastery, recent)."""
        if not self.enable_persistence:
            return []
        
        try:
            return self.memory_manager.get_archived_experiences_by_type(experience_type, limit)
        except Exception as e:
            print(f"Error getting archived experiences: {e}")
            return []
    
    # Universal Actuator Discovery Methods
    
    def get_discovered_actuator_categories(self) -> Dict[str, Dict[str, Any]]:
        """Get emergent actuator categories discovered by the brain."""
        return self.actuator_discovery.get_actuator_categories()
    
    def get_actuator_analysis(self, actuator_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed analysis of a specific actuator's discovered effects."""
        return self.actuator_discovery.get_actuator_analysis(actuator_id)
    
    def get_actuators_by_emergent_type(self, emergent_type: str) -> List[str]:
        """Get actuators that appear to have a specific emergent type (spatial, manipulative, environmental)."""
        categories = self.actuator_discovery.get_actuator_categories()
        matching_actuators = []
        
        for category_data in categories.values():
            properties = category_data.get('emergent_properties', {})
            
            if emergent_type == 'spatial' and properties.get('appears_spatial'):
                matching_actuators.extend(category_data['member_actuators'])
            elif emergent_type == 'manipulative' and properties.get('appears_manipulative'):
                matching_actuators.extend(category_data['member_actuators'])
            elif emergent_type == 'environmental' and properties.get('appears_environmental'):
                matching_actuators.extend(category_data['member_actuators'])
        
        return matching_actuators