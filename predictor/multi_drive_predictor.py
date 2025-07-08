"""
Multi-Drive Predictor - Integrates multiple competing drives with the prediction system.
Replaces single curiosity drive with a comprehensive motivation system.
"""

import time
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional
from core.world_graph import WorldGraph
from core.communication import PredictionPacket
from .single_traversal import SingleTraversal, TraversalResult
from .consensus_resolver import ConsensusResolver, ConsensusResult
from drives import create_default_motivation_system, DriveContext, MotivationSystem
from brain_prediction_profiler import profile_section


class MultiDrivePredictor:
    """
    Prediction engine that uses multiple competing drives to guide action selection.
    
    Combines the time-budgeted traversal system with multi-drive motivation
    to create sophisticated, emergent robot behavior.
    """
    
    def __init__(self, max_depth: int = 5, randomness_factor: float = 0.3,
                 action_similarity_threshold: float = 0.1, base_time_budget: float = 0.1,
                 motivation_system: Optional[MotivationSystem] = None, 
                 enable_parallel_traversals: bool = True, max_workers: Optional[int] = None):
        """
        Initialize multi-drive predictor.
        
        Args:
            max_depth: Maximum depth for traversals
            randomness_factor: Amount of randomness in node selection
            action_similarity_threshold: How similar actions need to be for consensus
            base_time_budget: Base thinking time budget in seconds
            motivation_system: Optional custom motivation system
            enable_parallel_traversals: Enable parallel graph traversals for CPU optimization
            max_workers: Maximum worker threads (defaults to CPU count)
        """
        self.base_time_budget = base_time_budget
        self.single_traversal = SingleTraversal(max_depth, randomness_factor)
        self.consensus_resolver = ConsensusResolver(action_similarity_threshold)
        
        # Threading configuration for CPU optimization
        self.enable_parallel_traversals = enable_parallel_traversals
        self.max_workers = max_workers or min(8, (os.cpu_count() or 1) + 1)  # Limit to 8 threads max
        
        print(f"MultiDrivePredictor: Parallel traversals {'enabled' if enable_parallel_traversals else 'disabled'}")
        if enable_parallel_traversals:
            print(f"  Max workers: {self.max_workers} (CPU cores: {os.cpu_count()})")
        
        # Initialize motivation system
        self.motivation_system = motivation_system or create_default_motivation_system()
        
        # Threat-responsive time scaling
        self.threat_multipliers = {
            "safe": 2.0,      # 200ms - can think longer when safe
            "normal": 1.0,    # 100ms - baseline
            "alert": 0.5,     # 50ms - need quick decisions  
            "danger": 0.2,    # 20ms - immediate action required
            "critical": 0.05  # 5ms - emergency reflexes
        }
        
        # Statistics tracking
        self.total_predictions = 0
        self.consensus_stats = {
            'perfect': 0,
            'strong': 0, 
            'weak': 0,
            'single': 0,
            'motivation_driven': 0,
            'no_consensus': 0
        }
        self.average_thinking_time = 0.0
        self.traversal_count_history = []
        self.drive_dominance_history = []
    
    def generate_prediction(self, current_context: List[float], 
                           world_graph: WorldGraph,
                           sequence_id: int = 0, threat_level: str = "normal",
                           robot_health: float = 1.0, robot_energy: float = 1.0,
                           robot_position: tuple = (0, 0), robot_orientation: int = 0,
                           step_count: int = 0) -> ConsensusResult:
        """
        Generate prediction using multi-drive motivation system.
        
        Args:
            current_context: Current mental context
            world_graph: Experience graph
            sequence_id: Sequence ID for prediction
            threat_level: Current threat assessment
            robot_health: Robot health [0.0-1.0]
            robot_energy: Robot energy [0.0-1.0] 
            robot_position: Robot position (x, y)
            robot_orientation: Robot orientation 0-3
            step_count: Current simulation step
            
        Returns:
            ConsensusResult with motivation-driven action selection
        """
        start_time = time.time()
        
        # Handle empty graph (bootstrap case)
        with profile_section("bootstrap_check"):
            if not world_graph.has_nodes():
                return self._bootstrap_prediction(sequence_id, threat_level, 
                                                robot_health, robot_energy, 
                                                robot_position, robot_orientation, step_count)
        
        # Calculate time budget based on threat level
        with profile_section("time_budget_calculation"):
            time_budget = self._calculate_time_budget(threat_level)
        
        # Run time-budgeted traversals to get experience-based insights
        with profile_section("traversals"):
            if self.enable_parallel_traversals:
                traversal_results, traversal_count = self._run_parallel_traversals(
                    current_context, world_graph, start_time, time_budget
                )
            else:
                traversal_results, traversal_count = self._run_sequential_traversals(
                    current_context, world_graph, start_time, time_budget
                )
        
        # Create drive context
        with profile_section("drive_context_creation"):
            drive_context = DriveContext(
                current_sensory=current_context,
                robot_health=robot_health,
                robot_energy=robot_energy,
                robot_position=robot_position,
                robot_orientation=robot_orientation,
                recent_experiences=list(world_graph.all_nodes())[-10:],  # Last 10 experiences
                prediction_errors=self._extract_recent_prediction_errors(world_graph),
                time_since_last_food=self._calculate_time_since_last_food(world_graph),
                time_since_last_damage=self._calculate_time_since_last_damage(world_graph),
                threat_level=threat_level,
                step_count=step_count
            )
        
        # Generate action candidates using motivation system
        with profile_section("action_candidate_generation"):
            action_candidates = self.motivation_system.generate_action_candidates(drive_context)
            
            # Add experience-based actions from traversals
            for result in traversal_results:
                if result.prediction:
                    action_candidates.append(result.prediction.motor_action)
        
        # Use motivation system to choose best action
        with profile_section("motivation_evaluation"):
            motivation_result = self.motivation_system.evaluate_action_candidates(
                action_candidates, drive_context
            )
        
        # Create prediction packet with chosen action
        with profile_section("prediction_packet_creation"):
            thinking_time = time.time() - start_time
        
        prediction = PredictionPacket(
            expected_sensory=self._predict_sensory_outcome(motivation_result.chosen_action, world_graph),
            motor_action=motivation_result.chosen_action,
            confidence=motivation_result.confidence,
            timestamp=start_time,
            sequence_id=sequence_id,
            thinking_depth=traversal_count
        )
        
        # Add metadata
        prediction.consensus_strength = "motivation_driven"
        prediction.traversal_paths = [f"drive:{motivation_result.dominant_drive}"]
        prediction.threat_level = threat_level
        prediction.traversal_count = traversal_count
        prediction.time_budget_used = thinking_time
        
        # Create consensus result
        consensus_result = ConsensusResult(
            prediction=prediction,
            consensus_strength="motivation_driven",
            agreement_count=len([d for d in motivation_result.drive_contributions.values() if d > 0.3]),
            total_traversals=traversal_count,
            reasoning=f"Multi-drive decision: {motivation_result.reasoning}"
        )
        
        # Update statistics
        self._update_statistics(consensus_result, thinking_time, motivation_result.dominant_drive)
        
        return consensus_result
    
    def _bootstrap_prediction(self, sequence_id: int, threat_level: str = "normal",
                            robot_health: float = 1.0, robot_energy: float = 1.0,
                            robot_position: tuple = (0, 0), robot_orientation: int = 0,
                            step_count: int = 0) -> ConsensusResult:
        """Generate bootstrap prediction using motivation system."""
        from datetime import datetime
        
        # Create minimal drive context for bootstrap
        drive_context = DriveContext(
            current_sensory=[],
            robot_health=robot_health,
            robot_energy=robot_energy,
            robot_position=robot_position,
            robot_orientation=robot_orientation,
            recent_experiences=[],
            prediction_errors=[],
            time_since_last_food=0,
            time_since_last_damage=0,
            threat_level=threat_level,
            step_count=step_count
        )
        
        # Generate action candidates for bootstrap
        action_candidates = self.motivation_system.generate_action_candidates(drive_context)
        
        # Use motivation system to choose bootstrap action
        motivation_result = self.motivation_system.evaluate_action_candidates(
            action_candidates, drive_context
        )
        
        # Create bootstrap prediction
        bootstrap_prediction = PredictionPacket(
            expected_sensory=[],  # Empty - brain doesn't assume sensory length
            motor_action=motivation_result.chosen_action,
            confidence=motivation_result.confidence,
            timestamp=datetime.now(),
            sequence_id=sequence_id,
            thinking_depth=0
        )
        
        bootstrap_prediction.consensus_strength = "bootstrap_motivation"
        bootstrap_prediction.traversal_paths = [f"bootstrap:{motivation_result.dominant_drive}"]
        bootstrap_prediction.threat_level = threat_level
        bootstrap_prediction.traversal_count = 0
        bootstrap_prediction.time_budget_used = 0.0
        
        # Update drive dominance tracking
        self.drive_dominance_history.append(motivation_result.dominant_drive)
        if len(self.drive_dominance_history) > 100:
            self.drive_dominance_history = self.drive_dominance_history[-50:]
        
        return ConsensusResult(
            prediction=bootstrap_prediction,
            consensus_strength="bootstrap_motivation",
            agreement_count=0,
            total_traversals=0,
            reasoning=f"Bootstrap multi-drive decision: {motivation_result.reasoning}"
        )
    
    def _calculate_time_budget(self, threat_level: str) -> float:
        """Calculate thinking time budget based on threat level."""
        multiplier = self.threat_multipliers.get(threat_level, 1.0)
        return self.base_time_budget * multiplier
    
    def _run_parallel_traversals(self, current_context: List[float], world_graph: WorldGraph, 
                                start_time: float, time_budget: float) -> tuple[List[TraversalResult], int]:
        """Run traversals in parallel using ThreadPoolExecutor for CPU optimization."""
        traversal_results = []
        
        # First phase: Run initial batch of traversals in parallel
        initial_batch_size = min(self.max_workers, 4)  # Start with 4 parallel traversals
        base_seed = int(time.time() * 1000)
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit initial batch
            initial_futures = []
            for i in range(initial_batch_size):
                future = executor.submit(
                    self._single_traversal_wrapper,
                    current_context, world_graph, base_seed + i
                )
                initial_futures.append(future)
            
            # Collect initial results
            for future in as_completed(initial_futures):
                try:
                    result = future.result(timeout=time_budget)
                    traversal_results.append(result)
                except Exception as e:
                    print(f"Traversal failed: {e}")
            
            # Second phase: Continue with more traversals if time allows
            elapsed = time.time() - start_time
            if elapsed < time_budget * 0.7:  # Use 70% of budget threshold
                additional_batch_size = min(self.max_workers, 2)  # Smaller second batch
                additional_futures = []
                
                for i in range(additional_batch_size):
                    future = executor.submit(
                        self._single_traversal_wrapper,
                        current_context, world_graph, base_seed + initial_batch_size + i
                    )
                    additional_futures.append(future)
                
                # Collect additional results with remaining time
                remaining_time = time_budget - (time.time() - start_time)
                for future in as_completed(additional_futures, timeout=max(0.01, remaining_time)):
                    try:
                        result = future.result(timeout=0.01)
                        traversal_results.append(result)
                    except Exception:
                        break  # Time budget exceeded, stop collecting
        
        return traversal_results, len(traversal_results)
    
    def _run_sequential_traversals(self, current_context: List[float], world_graph: WorldGraph,
                                  start_time: float, time_budget: float) -> tuple[List[TraversalResult], int]:
        """Run traversals sequentially (original behavior)."""
        traversal_results = []
        traversal_count = 0
        
        # First traversal (always do at least one)
        first_seed = int(time.time() * 1000)
        first_result = self.single_traversal.traverse(
            start_context=current_context,
            world_graph=world_graph,
            random_seed=first_seed
        )
        traversal_results.append(first_result)
        traversal_count = 1
        first_duration = time.time() - start_time
        
        # Continue thinking if we have time budget remaining
        while self._should_continue_thinking(start_time, time_budget, first_duration):
            seed = int(time.time() * 1000) + traversal_count
            result = self.single_traversal.traverse(
                start_context=current_context,
                world_graph=world_graph,
                random_seed=seed
            )
            traversal_results.append(result)
            traversal_count += 1
        
        return traversal_results, traversal_count
    
    def _single_traversal_wrapper(self, current_context: List[float], world_graph: WorldGraph, 
                                 seed: int) -> TraversalResult:
        """Thread-safe wrapper for single traversal execution."""
        return self.single_traversal.traverse(
            start_context=current_context,
            world_graph=world_graph,
            random_seed=seed
        )
    
    def _should_continue_thinking(self, start_time: float, time_budget: float, first_duration: float) -> bool:
        """Decide whether to start another traversal based on time remaining."""
        elapsed = time.time() - start_time
        remaining = time_budget - elapsed
        
        # Conservative estimate: need 1.2x first duration for next traversal
        estimated_next = first_duration * 1.2
        
        return remaining > estimated_next
    
    def _extract_recent_prediction_errors(self, world_graph: WorldGraph) -> List[float]:
        """Extract recent prediction errors from world graph."""
        errors = []
        for node in list(world_graph.all_nodes())[-10:]:  # Last 10 nodes
            errors.append(node.prediction_error)
        return errors
    
    def _calculate_time_since_last_food(self, world_graph: WorldGraph) -> int:
        """Calculate steps since last food consumption from experience graph."""
        if not world_graph or world_graph.node_count() == 0:
            return 10  # Default when no experiences
        
        # Look through recent experiences for food consumption
        # Food consumption shows as sensor value change: energy increases
        recent_nodes = world_graph.get_recent_nodes(20)  # Check last 20 experiences
        
        for i, node in enumerate(recent_nodes):
            if hasattr(node, 'actual_sensory') and len(node.actual_sensory) > 18:
                # Check if energy increased significantly (food consumption indicator)
                if hasattr(node, 'predicted_sensory') and len(node.predicted_sensory) > 18:
                    energy_change = node.actual_sensory[18] - node.predicted_sensory[18]
                    if energy_change > 0.1:  # Significant energy increase = food found
                        return i  # Steps since food
        
        return min(50, len(recent_nodes))  # Haven't found food recently
    
    def _calculate_time_since_last_damage(self, world_graph: WorldGraph) -> int:
        """Calculate steps since last damage taken from experience graph."""
        if not world_graph or world_graph.node_count() == 0:
            return 10  # Default when no experiences
        
        # Look through recent experiences for damage events
        # Damage shows as health decrease in sensory data
        recent_nodes = world_graph.get_recent_nodes(20)  # Check last 20 experiences
        
        for i, node in enumerate(recent_nodes):
            if hasattr(node, 'actual_sensory') and len(node.actual_sensory) > 17:
                # Check if health decreased significantly (damage indicator)
                if hasattr(node, 'predicted_sensory') and len(node.predicted_sensory) > 17:
                    health_change = node.actual_sensory[17] - node.predicted_sensory[17]
                    if health_change < -0.001:  # Any health decrease = damage taken
                        return i  # Steps since damage
        
        return min(50, len(recent_nodes))  # Haven't taken damage recently
    
    def _predict_sensory_outcome(self, action: Dict[str, float], world_graph: WorldGraph) -> List[float]:
        """Predict sensory outcome for the chosen action."""
        # Simple prediction based on recent experiences
        # In a real implementation, this could be more sophisticated
        if world_graph.has_nodes():
            latest_node = world_graph.get_latest_node()
            if latest_node:
                return latest_node.actual_sensory.copy()
        
        return []  # Empty prediction for bootstrap case
    
    def _update_statistics(self, consensus_result: ConsensusResult, thinking_time: float, dominant_drive: str):
        """Update internal statistics tracking."""
        self.total_predictions += 1
        
        # Update consensus statistics
        consensus_type = consensus_result.consensus_strength
        if consensus_type in self.consensus_stats:
            self.consensus_stats[consensus_type] += 1
        
        # Update thinking time average
        self.average_thinking_time = (
            (self.average_thinking_time * (self.total_predictions - 1) + thinking_time) / 
            self.total_predictions
        )
        
        # Track traversal counts
        self.traversal_count_history.append(consensus_result.total_traversals)
        if len(self.traversal_count_history) > 100:
            self.traversal_count_history = self.traversal_count_history[-50:]
        
        # Track drive dominance
        self.drive_dominance_history.append(dominant_drive)
        if len(self.drive_dominance_history) > 100:
            self.drive_dominance_history = self.drive_dominance_history[-50:]
    
    def get_predictor_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the multi-drive predictor."""
        stats = {
            'total_predictions': self.total_predictions,
            'average_thinking_time': self.average_thinking_time,
            'consensus_breakdown': self.consensus_stats.copy(),
            'base_time_budget': self.base_time_budget,
            'threat_multipliers': self.threat_multipliers.copy()
        }
        
        # Calculate consensus percentages
        if self.total_predictions > 0:
            for consensus_type, count in self.consensus_stats.items():
                percentage = (count / self.total_predictions) * 100
                stats[f'{consensus_type}_percentage'] = percentage
        
        # Recent traversal statistics
        if self.traversal_count_history:
            stats['average_traversals'] = sum(self.traversal_count_history) / len(self.traversal_count_history)
            stats['max_traversals'] = max(self.traversal_count_history)
            stats['min_traversals'] = min(self.traversal_count_history)
        
        # Drive dominance statistics
        if self.drive_dominance_history:
            drive_counts = {}
            for drive in self.drive_dominance_history:
                drive_counts[drive] = drive_counts.get(drive, 0) + 1
            
            stats['drive_dominance_counts'] = drive_counts
            stats['most_dominant_drive'] = max(drive_counts.keys(), key=lambda d: drive_counts[d])
        
        # Motivation system statistics
        stats['motivation_system'] = self.motivation_system.get_motivation_statistics()
        
        return stats
    
    def get_drive(self, drive_name: str):
        """Get a specific drive from the motivation system."""
        return self.motivation_system.get_drive(drive_name)
    
    def add_drive(self, drive):
        """Add a new drive to the motivation system."""
        self.motivation_system.add_drive(drive)
    
    def remove_drive(self, drive_name: str):
        """Remove a drive from the motivation system."""
        return self.motivation_system.remove_drive(drive_name)