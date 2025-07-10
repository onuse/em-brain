"""
Refactored Curiosity Drive - Pure experience node creation through novelty-seeking.

This drive combines the novelty-seeking aspects of the old curiosity and exploration drives
into a unified system focused on experiencing new things. It seeks:
- Novel spatial locations
- Novel action combinations  
- Novel sensory experiences
- Novel environmental contexts

The drive uses emergent boredom based on experience accumulation rather than 
hardcoded oscillation detection.
"""

import math
import time
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict
from .base_motivator import BaseMotivator, MotivatorContext, ActionEvaluation
from core.world_graph import WorldGraph

# Import GPU sensory predictor for predictive novelty assessment
try:
    from prediction.sensory.gpu_sensory_predictor import GPUSensoryPredictor
    from prediction.sensory.sensory_predictor import SensoryPrediction
    GPU_PREDICTION_AVAILABLE = True
except ImportError:
    GPU_PREDICTION_AVAILABLE = False


class CuriosityMotivator(BaseMotivator):
    """
    Refactored Curiosity Drive focused on pure experience node creation.
    
    This drive seeks novelty in all forms:
    - Spatial novelty (new locations)
    - Action novelty (new action combinations)
    - Sensory novelty (new experiences)
    - Environmental novelty (new contexts)
    
    Uses emergent boredom from experience accumulation instead of hardcoded penalties.
    """
    
    def __init__(self, base_weight: float = 0.35):
        super().__init__("Curiosity", base_weight)
        
        # Experience-based familiarity tracking
        self.spatial_familiarity = defaultdict(float)  # How familiar each location is
        self.action_familiarity = defaultdict(float)   # How familiar each action is
        self.sensory_familiarity = defaultdict(float)  # How familiar each sensory context is
        
        # Natural boredom tracking
        self.current_boredom_level = 0.0
        self.boredom_threshold = 0.7  # When to seek new experiences
        
        # Novelty seeking parameters
        self.novelty_decay_rate = 0.998  # How quickly novelty fades (much slower decay)
        self.experience_diversity_target = 0.8  # Target diversity level
        
        # Performance tracking
        self.novel_experiences_count = 0
        self.total_experience_evaluations = 0
        
        # GPU sensory predictor for predictive novelty assessment
        self.gpu_predictor = None
        self.predictive_novelty_enabled = False
        
        # World graph for memory-based novelty (simplified approach)
        self.world_graph = None
        self.memory_based_novelty_enabled = False
        
    def initialize_gpu_predictor(self, world_graph: WorldGraph):
        """Initialize GPU predictor for predictive novelty assessment."""
        if GPU_PREDICTION_AVAILABLE and world_graph is not None:
            try:
                self.gpu_predictor = GPUSensoryPredictor(world_graph)
                self.predictive_novelty_enabled = True
                print(f"🧠 CuriosityDrive: GPU predictive novelty assessment enabled")
            except Exception as e:
                print(f"⚠️  CuriosityDrive: Could not enable GPU predictor: {e}")
                self.predictive_novelty_enabled = False
        else:
            self.predictive_novelty_enabled = False
    
            
    def evaluate_action(self, action: Dict[str, float], context: MotivatorContext) -> ActionEvaluation:
        """
        Evaluate action based on its potential to create novel experiences.
        """
        self.total_evaluations += 1
        self.total_experience_evaluations += 1
        
        # Calculate different types of novelty
        spatial_novelty = self._calculate_spatial_novelty(action, context)
        action_novelty = self._calculate_action_novelty(action, context)
        sensory_novelty = self._calculate_sensory_novelty(action, context)
        environmental_novelty = self._calculate_environmental_novelty(action, context)
        
        # Combine novelty scores
        total_novelty = (
            spatial_novelty * 0.3 +
            action_novelty * 0.25 +
            sensory_novelty * 0.25 +
            environmental_novelty * 0.2
        )
        
        # Apply emergent boredom logic - when bored with current situation,
        # familiar actions become LESS attractive (to encourage leaving)
        boredom_penalty = self._calculate_boredom_penalty(context, total_novelty)
        novelty_score = max(0.0, min(1.0, total_novelty - boredom_penalty))
        
        # Calculate confidence based on novelty clarity
        confidence = self._calculate_novelty_confidence(
            spatial_novelty, action_novelty, sensory_novelty, environmental_novelty
        )
        
        # Generate reasoning
        reasoning = self._generate_curiosity_reasoning(
            spatial_novelty, action_novelty, sensory_novelty, environmental_novelty, boredom_penalty
        )
        
        # Calculate urgency based on boredom level
        urgency = 0.2 + (self.current_boredom_level * 0.6)
        
        return ActionEvaluation(
            drive_name=self.name,
            action_score=novelty_score,
            confidence=confidence,
            reasoning=reasoning,
            urgency=urgency
        )
    
    def _calculate_spatial_novelty(self, action: Dict[str, float], context: MotivatorContext) -> float:
        """
        Calculate spatial novelty using memory-based approach.
        
        Uses the rich world_graph memory to determine how difficult it would be
        to find similar spatial experiences in memory.
        """
        # Use memory-based novelty if world_graph is available
        if hasattr(self, 'world_graph') and self.world_graph:
            return self._calculate_memory_based_spatial_novelty(action, context)
        else:
            # Fallback to improved familiarity-based approach
            return self._calculate_familiarity_based_spatial_novelty(action, context)
    
    def _calculate_memory_based_spatial_novelty(self, action: Dict[str, float], context: MotivatorContext) -> float:
        """
        Calculate spatial novelty based on memory search difficulty.
        
        High novelty = Hard to find similar spatial experiences in memory
        Low novelty = Easy to find similar spatial experiences in memory
        """
        current_pos = context.robot_position
        predicted_positions = self._predict_action_destinations(action, current_pos, context.robot_orientation)
        
        # Create spatial experience pattern that matches mental_context format (8 elements)
        # Use first 8 elements of current_sensory which includes position-aware data
        spatial_pattern = context.current_sensory[:8] if len(context.current_sensory) >= 8 else context.current_sensory
        
        # Use world_graph memory to calculate novelty based on search difficulty
        memory_based_novelty = self._get_memory_based_novelty(spatial_pattern)
        
        return memory_based_novelty
    
    def _calculate_familiarity_based_spatial_novelty(self, action: Dict[str, float], context: MotivatorContext) -> float:
        """
        Fallback spatial novelty calculation using familiarity counters.
        (Improved version of the original approach)
        """
        current_pos = context.robot_position
        
        # Predict where this action would take us
        predicted_positions = self._predict_action_destinations(action, current_pos, context.robot_orientation)
        
        # CRITICAL FIX: Use hybrid approach to prevent oscillation
        # Consider both current position and predicted position familiarity
        current_familiarity = self.spatial_familiarity.get(current_pos, 0.0)
        
        max_predicted_novelty = 0.0
        for pos in predicted_positions:
            # Calculate novelty based on familiarity
            familiarity = self.spatial_familiarity.get(pos, 0.0)
            novelty = 1.0 - familiarity
            max_predicted_novelty = max(max_predicted_novelty, novelty)
        
        # If we're staying in the same position (turning, braking, etc.)
        # use current position familiarity
        if len(predicted_positions) == 1 and predicted_positions[0] == current_pos:
            return 1.0 - current_familiarity
        
        # For movement actions, blend current and predicted novelty
        # This prevents oscillation by considering where we ARE and where we're GOING
        current_novelty = 1.0 - current_familiarity
        
        # Weight: 30% current position, 70% predicted position
        # This ensures we don't keep revisiting the same areas
        blended_novelty = (current_novelty * 0.3) + (max_predicted_novelty * 0.7)
        
        return blended_novelty
    
    def _create_spatial_experience_pattern(self, current_pos: Tuple[int, int], 
                                         predicted_positions: List[Tuple[int, int]],
                                         distance_sensors: List[float]) -> List[float]:
        """
        Create a spatial experience pattern for memory-based novelty detection.
        
        Combines position information with sensory context to create a rich
        spatial signature for memory search.
        """
        pattern = []
        
        # Current position (normalized)
        pattern.extend([current_pos[0] / 40.0, current_pos[1] / 40.0])
        
        # Primary predicted position (normalized)
        if predicted_positions:
            primary_pred = predicted_positions[0]
            pattern.extend([primary_pred[0] / 40.0, primary_pred[1] / 40.0])
        else:
            pattern.extend([0.0, 0.0])
        
        # Distance sensor context (obstacles around)
        pattern.extend(distance_sensors[:4])  # Front, left, right, back sensors
        
        # Movement direction encoding
        if predicted_positions and predicted_positions[0] != current_pos:
            dx = predicted_positions[0][0] - current_pos[0]
            dy = predicted_positions[0][1] - current_pos[1]
            # Normalize movement direction
            movement_magnitude = (dx*dx + dy*dy) ** 0.5
            if movement_magnitude > 0:
                pattern.extend([dx/movement_magnitude, dy/movement_magnitude])
            else:
                pattern.extend([0.0, 0.0])
        else:
            pattern.extend([0.0, 0.0])  # No movement
        
        return pattern
    
    def _get_memory_based_novelty(self, experience_pattern: List[float]) -> float:
        """
        Calculate memory-based novelty using the world_graph.
        
        FIXED: Properly accounts for the true number of similar memories,
        not just the capped search results.
        """
        if not self.world_graph or not self.world_graph.has_nodes():
            return 1.0  # Maximum novelty if no memory exists
        
        start_time = time.time()
        
        # First, do a proper similarity check at our threshold
        # Use a high max_results to get true count
        try:
            # Get actual count of similar nodes at our similarity threshold
            similar_at_threshold = self.world_graph.find_similar_nodes(
                experience_pattern, 
                similarity_threshold=0.35,  # Our actual threshold
                max_results=1000  # High limit to get true count
            )
            
            # If we have many similar experiences, this is very familiar
            similarity_ratio = len(similar_at_threshold) / max(1, self.world_graph.node_count())
            
            # Progressive search for nuanced novelty
            search_attempts = 0
            found_at_threshold = 0.35
            
            for threshold in [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.35]:
                search_attempts += 1
                similar_nodes = self.world_graph.find_similar_nodes(
                    experience_pattern,
                    similarity_threshold=threshold,
                    max_results=5  # Small sample for speed
                )
                if similar_nodes:
                    found_at_threshold = threshold
                    break
            
            search_time = time.time() - start_time
            
            # Calculate novelty based on multiple factors
            # 1. Similarity ratio - how many memories are similar
            familiarity = similarity_ratio  # 0 = novel, 1 = very familiar
            
            # 2. Threshold where we found matches - higher threshold = more familiar
            threshold_factor = found_at_threshold  # 0.9 = very familiar, 0.35 = less familiar
            
            # 3. Search difficulty - longer search = more novel
            time_factor = min(1.0, search_time * 1000)  # Convert to ms and cap at 1.0
            
            # Combine factors with emphasis on actual similarity count
            memory_based_novelty = 1.0 - (
                familiarity * 0.7 +          # Most weight on actual similarity ratio
                threshold_factor * 0.2 +      # Some weight on match threshold
                (1.0 - time_factor) * 0.1    # Small weight on search speed
            )
            
            return min(1.0, max(0.0, memory_based_novelty))
            
        except Exception as e:
            # If similarity search fails, return moderate novelty
            print(f"Warning: Memory-based novelty search failed: {e}")
            return 0.5
    
    def _calculate_action_novelty(self, action: Dict[str, float], context: MotivatorContext) -> float:
        """
        Calculate action novelty using memory-based approach.
        
        Uses the rich world_graph memory to determine how difficult it would be
        to find similar action experiences in memory.
        """
        # Use memory-based novelty if world_graph is available
        if hasattr(self, 'world_graph') and self.world_graph:
            return self._calculate_memory_based_action_novelty(action, context)
        else:
            # Fallback to improved familiarity-based approach
            return self._calculate_familiarity_based_action_novelty(action, context)
    
    def _calculate_memory_based_action_novelty(self, action: Dict[str, float], context: MotivatorContext) -> float:
        """
        Calculate action novelty based on memory search difficulty.
        
        High novelty = Hard to find similar action experiences in memory
        Low novelty = Easy to find similar action experiences in memory
        """
        # Create action experience pattern for memory search using same format as mental_context
        # Combine current sensory state with action for pattern matching
        action_pattern = context.current_sensory[:8] if len(context.current_sensory) >= 8 else context.current_sensory
        
        # Use world_graph memory to calculate novelty based on search difficulty
        memory_based_novelty = self._get_memory_based_novelty(action_pattern)
        
        return memory_based_novelty
    
    def _calculate_familiarity_based_action_novelty(self, action: Dict[str, float], context: MotivatorContext) -> float:
        """
        Fallback action novelty calculation using familiarity counters.
        (Improved version with coarser quantization)
        """
        # Create action signature
        action_sig = self._create_action_signature(action)
        
        # Calculate novelty based on familiarity
        familiarity = self.action_familiarity.get(action_sig, 0.0)
        return 1.0 - familiarity
    
    def _create_action_experience_pattern(self, action: Dict[str, float], context: MotivatorContext) -> List[float]:
        """
        Create an action experience pattern for memory-based novelty detection.
        
        Combines action parameters with contextual information to create a rich
        action signature for memory search.
        """
        pattern = []
        
        # Core motor action values
        pattern.append(action.get('forward_motor', 0.0))
        pattern.append(action.get('turn_motor', 0.0)) 
        pattern.append(action.get('brake_motor', 0.0))
        
        # Contextual sensory state (first 4 sensors - obstacles)
        pattern.extend(context.current_sensory[:4])
        
        # Robot state context
        pattern.extend([
            context.robot_health,
            context.robot_energy,
            float(context.robot_orientation) / 3.0  # Normalize orientation
        ])
        
        # Recent prediction error context (influences action selection)
        if context.prediction_errors:
            recent_error = context.prediction_errors[-1] if context.prediction_errors else 0.0
            pattern.append(recent_error)
        else:
            pattern.append(0.0)
        
        return pattern
    
    def _calculate_sensory_novelty(self, action: Dict[str, float], context: MotivatorContext) -> float:
        """Calculate sensory novelty using memory-based approach for consistency."""
        # Use memory-based novelty for consistency with other novelty calculations
        if self.memory_based_novelty_enabled and self.world_graph:
            # Use current sensory pattern for sensory memory search
            sensory_pattern = context.current_sensory[:8] if len(context.current_sensory) >= 8 else context.current_sensory
            memory_novelty = self._get_memory_based_novelty(sensory_pattern)
            return memory_novelty
        elif self.predictive_novelty_enabled and self.gpu_predictor:
            # Fallback to predictive assessment
            return self._calculate_predictive_sensory_novelty(action, context)
        else:
            # Fallback to reactive assessment
            return self._calculate_reactive_sensory_novelty(action, context)
    
    def _calculate_predictive_sensory_novelty(self, action: Dict[str, float], context: MotivatorContext) -> float:
        """Calculate novelty of PREDICTED sensory experience from taking this action."""
        try:
            # Predict what sensory experience this action would produce
            current_context = context.current_sensory[:8] if len(context.current_sensory) >= 8 else context.current_sensory
            prediction = self.gpu_predictor.predict_sensory_outcome(action, current_context)
            
            # Extract predicted sensory values
            predicted_sensory = []
            for i in range(8):  # Use first 8 sensors
                sensor_name = f"sensor_{i}"
                predicted_value = prediction.get_sensor_prediction(sensor_name, 0.0)
                predicted_sensory.append(predicted_value)
            
            # Calculate novelty of PREDICTED experience
            predicted_sig = tuple(predicted_sensory)
            predicted_familiarity = self.sensory_familiarity.get(predicted_sig, 0.0)
            
            # Weight novelty by prediction confidence
            base_novelty = 1.0 - predicted_familiarity
            confidence_weighted_novelty = base_novelty * prediction.confidence
            
            return confidence_weighted_novelty
            
        except Exception as e:
            # Fallback to reactive assessment if prediction fails
            return self._calculate_reactive_sensory_novelty(action, context)
    
    def _calculate_reactive_sensory_novelty(self, action: Dict[str, float], context: MotivatorContext) -> float:
        """Fallback reactive sensory novelty calculation."""
        # Create sensory signature from current context
        sensory_sig = tuple(context.current_sensory[:8]) if len(context.current_sensory) >= 8 else tuple(context.current_sensory)
        
        # Calculate novelty based on familiarity
        familiarity = self.sensory_familiarity.get(sensory_sig, 0.0)
        return 1.0 - familiarity
    
    def _calculate_environmental_novelty(self, action: Dict[str, float], context: MotivatorContext) -> float:
        """Calculate environmental novelty using memory-based approach for consistency."""
        # Use memory-based novelty for consistency with spatial and action novelty
        if self.memory_based_novelty_enabled and self.world_graph:
            # Use current sensory pattern for environmental memory search
            environmental_pattern = context.current_sensory[:8] if len(context.current_sensory) >= 8 else context.current_sensory
            memory_novelty = self._get_memory_based_novelty(environmental_pattern)
            return memory_novelty
        else:
            # Fallback to familiarity-based calculation
            familiarity = self.spatial_familiarity.get(context.robot_position, 0.0)
            sensory_familiarity = self.sensory_familiarity.get(tuple(context.current_sensory[:4]), 0.0)
            
            # Environmental novelty is inverse of combined familiarity
            combined_familiarity = (familiarity + sensory_familiarity) / 2.0
            return 1.0 - combined_familiarity
    
    def _calculate_boredom_penalty(self, context: MotivatorContext, base_novelty: float) -> float:
        """
        Calculate boredom penalty for familiar actions.
        
        When highly bored with the current situation, familiar actions (low novelty)
        become LESS attractive to encourage the robot to leave the boring area.
        """
        # Only apply penalty when significantly bored
        if self.current_boredom_level > self.boredom_threshold:
            # Higher penalty for more familiar actions (lower base novelty)
            familiarity = 1.0 - base_novelty
            boredom_penalty = familiarity * self.current_boredom_level * 0.4
            return boredom_penalty
        
        return 0.0
    
    def _calculate_boredom_from_experiences(self, world_graph: Optional[WorldGraph], context: MotivatorContext) -> float:
        """
        Calculate emergent boredom based on accumulated experiences.
        
        This is the core of the emergent boredom system - boredom arises naturally
        from having many similar experiences rather than from hardcoded rules.
        """
        if not world_graph or not hasattr(world_graph, 'nodes') or not world_graph.nodes:
            return 0.0  # No experiences yet - everything is novel
        
        # Get current sensory context
        current_context = context.current_sensory[:8] if len(context.current_sensory) >= 8 else context.current_sensory
        
        # Find similar experiences
        try:
            similar_experiences = world_graph.find_similar_nodes(
                current_context,
                similarity_threshold=0.6,  # Moderate threshold for spatial similarity
                max_results=15
            )
        except:
            return 0.0  # Fallback if world_graph not available
        
        if not similar_experiences:
            return 0.0  # No similar experiences - novel area
        
        # Calculate familiarity metrics
        experience_count = len(similar_experiences)
        
        # Calculate average prediction accuracy (high accuracy = familiar)
        total_accuracy = sum(max(0.0, 1.0 - exp.prediction_error) for exp in similar_experiences)
        avg_accuracy = total_accuracy / experience_count if experience_count > 0 else 0.0
        
        # Calculate boredom based on familiarity
        # More experiences + higher accuracy = higher boredom
        familiarity_factor = min(1.0, experience_count / 10.0)  # Normalize to 0-1
        accuracy_factor = avg_accuracy  # Already 0-1
        
        # Combine factors
        boredom_level = (familiarity_factor * 0.6 + accuracy_factor * 0.4)
        
        return min(1.0, boredom_level)
    
    def _calculate_novelty_confidence(self, spatial: float, action: float, sensory: float, environmental: float) -> float:
        """Calculate confidence based on novelty clarity."""
        # Higher confidence when novelty is clear and strong
        max_novelty = max(spatial, action, sensory, environmental)
        novelty_clarity = max_novelty - min(spatial, action, sensory, environmental)
        
        base_confidence = max_novelty * 0.6
        clarity_bonus = novelty_clarity * 0.4
        
        return min(1.0, base_confidence + clarity_bonus)
    
    def _generate_curiosity_reasoning(self, spatial: float, action: float, sensory: float, environmental: float, boredom_penalty: float) -> str:
        """Generate reasoning for curiosity evaluation."""
        max_novelty = max(spatial, action, sensory, environmental)
        
        if boredom_penalty > 0.2:
            return f"Bored with familiar actions - seeking change (penalty: {boredom_penalty:.2f})"
        elif max_novelty == spatial and spatial > 0.7:
            return f"High spatial novelty ({spatial:.2f}) - unexplored area"
        elif max_novelty == action and action > 0.7:
            return f"High action novelty ({action:.2f}) - new action combination"
        elif max_novelty == sensory and sensory > 0.7:
            prediction_type = "predicted" if self.predictive_novelty_enabled else "current"
            return f"High sensory novelty ({sensory:.2f}) - {prediction_type} sensory experience"
        elif max_novelty == environmental and environmental > 0.7:
            return f"High environmental novelty ({environmental:.2f}) - new situation"
        elif max_novelty > 0.5:
            return f"Moderate novelty opportunity ({max_novelty:.2f})"
        else:
            return f"Low novelty - familiar situation ({max_novelty:.2f})"
    
    def update_drive_state(self, context: MotivatorContext, world_graph: Optional[WorldGraph] = None) -> float:
        """
        Update curiosity drive state based on current context.
        
        This includes updating familiarity tracking and calculating emergent boredom.
        """
        # Initialize GPU predictor if world graph is available and not already initialized
        if world_graph and not self.predictive_novelty_enabled:
            self.initialize_gpu_predictor(world_graph)
        
        # Store world graph reference for memory-based novelty
        if world_graph and not self.memory_based_novelty_enabled:
            self.world_graph = world_graph
            self.memory_based_novelty_enabled = True
            print(f"🧠 CuriosityDrive: Memory-based novelty enabled using world_graph (threshold: {world_graph.similarity_threshold})")
        
        # Update familiarity tracking
        self._update_familiarity_tracking(context)
        
        # Calculate emergent boredom from experiences
        if world_graph:
            self.current_boredom_level = self._calculate_boredom_from_experiences(world_graph, context)
        
        # Calculate drive weight based on novelty potential
        novelty_potential = self._calculate_overall_novelty_potential(context)
        
        # Higher weight when there's more novelty potential or when bored
        boredom_multiplier = 1.0 + (self.current_boredom_level * 0.5)
        self.current_weight = self.base_weight * novelty_potential * boredom_multiplier
        
        return self.current_weight
    
    def _update_familiarity_tracking(self, context: MotivatorContext):
        """Update familiarity tracking for different experience types."""
        # Update spatial familiarity
        pos = context.robot_position
        self.spatial_familiarity[pos] = min(1.0, self.spatial_familiarity[pos] + 0.05)
        
        # Update sensory familiarity
        sensory_sig = tuple(context.current_sensory[:8]) if len(context.current_sensory) >= 8 else tuple(context.current_sensory)
        self.sensory_familiarity[sensory_sig] = min(1.0, self.sensory_familiarity[sensory_sig] + 0.03)
        
        # Decay familiarity over time to maintain some novelty
        self._decay_familiarity()
    
    def _decay_familiarity(self):
        """Gradually decay familiarity to maintain novelty over time."""
        # Decay spatial familiarity
        for pos in list(self.spatial_familiarity.keys()):
            self.spatial_familiarity[pos] *= self.novelty_decay_rate
            if self.spatial_familiarity[pos] < 0.01:
                del self.spatial_familiarity[pos]
        
        # Decay action familiarity
        for action_sig in list(self.action_familiarity.keys()):
            self.action_familiarity[action_sig] *= self.novelty_decay_rate
            if self.action_familiarity[action_sig] < 0.01:
                del self.action_familiarity[action_sig]
        
        # Decay sensory familiarity
        for sensory_sig in list(self.sensory_familiarity.keys()):
            self.sensory_familiarity[sensory_sig] *= self.novelty_decay_rate
            if self.sensory_familiarity[sensory_sig] < 0.01:
                del self.sensory_familiarity[sensory_sig]
    
    def _calculate_overall_novelty_potential(self, context: MotivatorContext) -> float:
        """Calculate overall novelty potential in current situation using memory-based approach."""
        # Use memory-based novelty for consistency with action evaluation
        if self.memory_based_novelty_enabled and self.world_graph:
            # Use current sensory pattern for memory search
            sensory_pattern = context.current_sensory[:8] if len(context.current_sensory) >= 8 else context.current_sensory
            memory_novelty = self._get_memory_based_novelty(sensory_pattern)
            novelty_potential = memory_novelty
        else:
            # Fallback to familiarity-based calculation
            spatial_familiarity = self.spatial_familiarity.get(context.robot_position, 0.0)
            sensory_sig = tuple(context.current_sensory[:8]) if len(context.current_sensory) >= 8 else tuple(context.current_sensory)
            sensory_familiarity = self.sensory_familiarity.get(sensory_sig, 0.0)
            
            # Novelty potential is inverse of familiarity
            avg_familiarity = (spatial_familiarity + sensory_familiarity) / 2.0
            novelty_potential = 1.0 - avg_familiarity
        
        # Boost potential when bored
        if self.current_boredom_level > 0.5:
            novelty_potential = min(1.0, novelty_potential + 0.3)
        
        return max(0.1, novelty_potential)  # Maintain minimum drive
    
    def _predict_action_destinations(self, action: Dict[str, float], current_pos: Tuple[int, int], orientation: int) -> List[Tuple[int, int]]:
        """Predict where the robot might end up after executing this action."""
        forward_motor = action.get('forward_motor', 0.0)
        turn_motor = action.get('turn_motor', 0.0)
        
        predicted_positions = []
        x, y = current_pos
        
        # Handle turning
        new_orientation = orientation
        if abs(turn_motor) > 0.3:
            if turn_motor > 0:
                new_orientation = (orientation + 1) % 4
            else:
                new_orientation = (orientation - 1) % 4
        
        # Handle movement
        if abs(forward_motor) > 0.2:
            # Direction vectors: [North, East, South, West]
            direction_vectors = [(0, -1), (1, 0), (0, 1), (-1, 0)]
            dx, dy = direction_vectors[new_orientation]
            
            if forward_motor > 0:
                # Forward
                predicted_pos = (x + dx, y + dy)
            else:
                # Backward
                predicted_pos = (x - dx, y - dy)
            
            predicted_positions.append(predicted_pos)
        else:
            # No movement - stay in place
            predicted_positions.append(current_pos)
        
        return predicted_positions
    
    def _create_action_signature(self, action: Dict[str, float]) -> str:
        """Create a signature for action familiarity tracking."""
        forward = action.get('forward_motor', 0.0)
        turn = action.get('turn_motor', 0.0)
        brake = action.get('brake_motor', 0.0)
        
        # COARSER quantization to prevent oscillation from tiny motor variations
        # Group similar actions together more aggressively
        forward_q = round(forward * 2) / 2  # Quantize to 0.5 steps (was 0.25)
        turn_q = round(turn * 2) / 2        # This groups more actions as "same"
        brake_q = round(brake * 2) / 2
        
        # Further classify into broad action types to reduce micro-novelty
        action_type = self._classify_action_type(forward_q, turn_q, brake_q)
        
        return f"{action_type}_{forward_q:.1f}_{turn_q:.1f}_{brake_q:.1f}"
    
    def _classify_action_type(self, forward: float, turn: float, brake: float) -> str:
        """Classify action into broad types to reduce micro-novelty."""
        if brake > 0.5:
            return "brake"
        elif abs(turn) > 0.5:
            return "turn"
        elif forward > 0.5:
            return "forward"
        elif forward < -0.5:
            return "backward"
        else:
            return "idle"
    
    def record_action_execution(self, action: Dict[str, float], context: MotivatorContext):
        """Record that an action was executed to update familiarity and memory."""
        # Update action familiarity (fallback system)
        action_sig = self._create_action_signature(action)
        self.action_familiarity[action_sig] = min(1.0, self.action_familiarity[action_sig] + 0.1)
        
        # OSCILLATION FIX: Pre-populate familiarity for predicted positions
        # This helps prevent immediate re-evaluation of the same moves
        predicted_positions = self._predict_action_destinations(action, context.robot_position, context.robot_orientation)
        for pos in predicted_positions:
            if pos != context.robot_position:  # Don't double-update current position
                # Add slight familiarity to predicted positions to reduce oscillation
                self.spatial_familiarity[pos] = min(1.0, self.spatial_familiarity[pos] + 0.02)
        
        # Memory-based novelty uses the world_graph directly (no separate storage needed)
        # The world_graph already stores all experiences with the lowered similarity threshold
        
        # Track novel experiences
        if self.action_familiarity[action_sig] < 0.3:  # Relatively novel action
            self.novel_experiences_count += 1
    
    def record_predicted_experience(self, predicted_sensory: List[float]):
        """Record familiarity of a predicted sensory experience."""
        # Update familiarity tracking for predicted experiences
        predicted_sig = tuple(predicted_sensory)
        self.sensory_familiarity[predicted_sig] = min(1.0, self.sensory_familiarity[predicted_sig] + 0.05)
    
    def evaluate_experience_valence(self, experience, context: MotivatorContext) -> float:
        """
        Curiosity drive's pain/pleasure evaluation.
        
        Novelty and discovery = PLEASURE
        Repetition and familiarity = MILD PAIN (boredom)
        """
        # Calculate novelty of this experience
        action_sig = self._create_action_signature(experience.action_taken)
        action_novelty = 1.0 - self.action_familiarity.get(action_sig, 0.0)
        
        spatial_novelty = 1.0 - self.spatial_familiarity.get(context.robot_position, 0.0)
        
        # High novelty = pleasure, low novelty = mild boredom pain
        avg_novelty = (action_novelty + spatial_novelty) / 2.0
        
        if avg_novelty > 0.7:
            return 0.6  # High pleasure from novel experiences
        elif avg_novelty > 0.4:
            return 0.2  # Moderate pleasure
        elif avg_novelty > 0.2:
            return 0.0  # Neutral
        else:
            return -0.3  # Mild boredom pain from repetition
    
    def get_current_mood_contribution(self, context: MotivatorContext) -> Dict[str, float]:
        """Curiosity drive's contribution to robot mood."""
        # Calculate novelty satisfaction
        current_novelty = self._calculate_overall_novelty_potential(context)
        
        # Satisfaction based on novelty availability
        if current_novelty > 0.7:
            satisfaction = 0.6  # High satisfaction with novel environment
        elif current_novelty > 0.4:
            satisfaction = 0.2  # Moderate satisfaction
        elif current_novelty > 0.2:
            satisfaction = -0.1  # Slight dissatisfaction
        else:
            satisfaction = -0.4  # Boredom dissatisfaction
        
        # Urgency based on boredom level
        urgency = self.current_boredom_level
        
        # Confidence based on novelty clarity
        confidence = current_novelty * 0.8
        
        return {
            'satisfaction': satisfaction,
            'urgency': urgency,
            'confidence': confidence
        }
    
    def get_curiosity_stats(self) -> Dict:
        """Get detailed curiosity drive statistics."""
        stats = self.get_drive_info()
        stats.update({
            'current_boredom_level': self.current_boredom_level,
            'spatial_locations_known': len(self.spatial_familiarity),
            'action_patterns_known': len(self.action_familiarity),
            'sensory_patterns_known': len(self.sensory_familiarity),
            'novel_experiences_count': self.novel_experiences_count,
            'total_experience_evaluations': self.total_experience_evaluations,
            'novelty_seeking_rate': self.novel_experiences_count / max(1, self.total_experience_evaluations),
            'predictive_novelty_enabled': self.predictive_novelty_enabled,
            'gpu_predictor_available': self.gpu_predictor is not None
        })
        return stats
    
    def reset_curiosity(self):
        """Reset curiosity drive (useful for new environments)."""
        self.spatial_familiarity.clear()
        self.action_familiarity.clear()
        self.sensory_familiarity.clear()
        self.current_boredom_level = 0.0
        self.novel_experiences_count = 0
        self.total_experience_evaluations = 0
        self.reset_drive()