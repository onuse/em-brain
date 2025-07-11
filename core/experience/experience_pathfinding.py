#!/usr/bin/env python3
"""
Experience Pathfinding - Connection-based navigation through experience space.

This implements pathfinding that follows experience connections rather than
geometric coordinates. Paths are formed through lived experience:
- Smell → Room (experiential connection)
- Sound → Danger (emotional connection)
- Action → Outcome (causal connection)

Key principle: Navigation follows meaning and experience, not just spatial distance.
"""

from typing import Dict, List, Optional, Set, Tuple, Any
import time
from dataclasses import dataclass
from collections import deque
import heapq

from core.memory.world_graph import WorldGraph
from core.memory.experience_node import ExperienceNode
from .frontier_detection import ExperienceFrontier


@dataclass
class ExperiencePath:
    """A path through experience space."""
    path_id: str
    experience_sequence: List[ExperienceNode]
    connection_types: List[str]  # Types of connections between experiences
    path_strength: float  # Overall strength of the path
    path_confidence: float  # Confidence in path validity
    estimated_steps: int  # Estimated steps to traverse
    path_metadata: Dict[str, Any]


@dataclass
class ConnectionSignature:
    """Signature of a connection between experiences."""
    connection_type: str  # spatial, causal, temporal, emotional, etc.
    strength: float
    directionality: bool  # Is connection directional?
    metadata: Dict[str, Any]


class ExperiencePathfinding:
    """
    Experience-based pathfinding through lived experience connections.
    
    This system navigates by following natural experience connections rather
    than geometric coordinates. Paths reflect how experiences are actually
    connected through lived interaction with the environment.
    """
    
    def __init__(self, world_graph: WorldGraph):
        self.world_graph = world_graph
        
        # Connection analysis
        self.connection_cache = {}
        self.path_cache = {}
        self.connection_types = {
            'spatial': 0.8,      # Physical movement connections
            'causal': 0.9,       # Action → outcome connections
            'temporal': 0.7,     # Time-based sequence connections
            'sensory': 0.6,      # Sensory similarity connections
            'emotional': 0.5     # Pain/pleasure connections
        }
        
        # Performance tracking
        self.pathfinding_requests = 0
        self.successful_paths = 0
        self.cache_hits = 0
        self.connection_discoveries = 0
        
        print("🗺️ Experience Pathfinding initialized - connection-based navigation")
    
    def find_paths_to_frontiers(
        self, 
        frontiers: List[ExperienceFrontier],
        current_experience: Optional[ExperienceNode] = None
    ) -> List[ExperienceNode]:
        """
        Find experience paths to reach discovered frontiers.
        
        Returns a sequence of experiences that represent a path toward
        the most promising frontier exploration targets.
        """
        if not frontiers or not current_experience:
            return []
        
        self.pathfinding_requests += 1
        
        # Find paths to each frontier
        frontier_paths = []
        for frontier in frontiers[:5]:  # Limit to top 5 frontiers
            path = self.find_path_to_frontier(frontier, current_experience)
            if path:
                frontier_paths.append(path)
        
        if not frontier_paths:
            return []
        
        # Select best path based on multiple criteria
        best_path = self._select_best_frontier_path(frontier_paths)
        
        if best_path:
            self.successful_paths += 1
            return best_path.experience_sequence[1:6]  # Return next 5 steps
        
        return []
    
    def find_path_to_frontier(
        self, 
        frontier: ExperienceFrontier,
        current_experience: ExperienceNode
    ) -> Optional[ExperiencePath]:
        """Find a path to a specific frontier."""
        # For spatial frontiers, find path to predicted experience
        if frontier.target_experience:
            return self.find_path_to_experience(frontier.target_experience, current_experience)
        
        # For action/conceptual frontiers, find path to similar contexts
        target_context = self._generate_frontier_target_context(frontier)
        if target_context:
            target_experiences = self.world_graph.find_similar_nodes(
                target_context, similarity_threshold=0.6, max_results=5
            )
            
            if target_experiences:
                # Find best target experience
                best_target = max(target_experiences, key=lambda exp: exp.strength)
                return self.find_path_to_experience(best_target, current_experience)
        
        return None
    
    def find_path_to_experience(
        self, 
        target_experience: ExperienceNode,
        current_experience: Optional[ExperienceNode] = None
    ) -> Optional[ExperiencePath]:
        """
        Find path between two experiences using experience connections.
        
        This is the core pathfinding algorithm that follows natural experience
        connections rather than geometric distances.
        """
        if not current_experience or not target_experience:
            return None
        
        # Check cache first
        cache_key = f"{id(current_experience)}_{id(target_experience)}"
        if cache_key in self.path_cache:
            self.cache_hits += 1
            return self.path_cache[cache_key]
        
        # Use A* search with experience-based heuristics
        path = self._experience_astar_search(current_experience, target_experience)
        
        # Cache the result
        if path:
            self.path_cache[cache_key] = path
            
            # Limit cache size
            if len(self.path_cache) > 1000:
                oldest_keys = list(self.path_cache.keys())[:200]
                for key in oldest_keys:
                    del self.path_cache[key]
        
        return path
    
    def _experience_astar_search(
        self,
        start: ExperienceNode,
        target: ExperienceNode
    ) -> Optional[ExperiencePath]:
        """
        A* search through experience space using experience-based heuristics.
        """
        # Priority queue: (f_score, g_score, experience, path)
        open_set = [(0.0, 0.0, start, [start])]
        closed_set = set()
        g_scores = {id(start): 0.0}
        
        max_iterations = 100  # Prevent infinite loops
        iterations = 0
        
        while open_set and iterations < max_iterations:
            iterations += 1
            
            # Get experience with lowest f_score
            f_score, g_score, current, path = heapq.heappop(open_set)
            
            current_id = id(current)
            if current_id in closed_set:
                continue
            
            closed_set.add(current_id)
            
            # Check if we reached the target
            if current_id == id(target):
                return self._create_experience_path(path, start, target)
            
            # Explore connected experiences
            connected_experiences = self._get_connected_experiences(current)
            
            for connected_exp, connection_strength in connected_experiences:
                connected_id = id(connected_exp)
                
                if connected_id in closed_set:
                    continue
                
                # Calculate tentative g_score
                tentative_g = g_score + (1.0 - connection_strength)
                
                if connected_id not in g_scores or tentative_g < g_scores[connected_id]:
                    g_scores[connected_id] = tentative_g
                    
                    # Calculate heuristic (experience distance to target)
                    h_score = self._experience_heuristic(connected_exp, target)
                    f_score = tentative_g + h_score
                    
                    new_path = path + [connected_exp]
                    heapq.heappush(open_set, (f_score, tentative_g, connected_exp, new_path))
        
        return None  # No path found
    
    def _get_connected_experiences(self, experience: ExperienceNode) -> List[Tuple[ExperienceNode, float]]:
        """
        Get experiences connected to the given experience with connection strengths.
        
        Connections are based on:
        - Spatial proximity (similar sensory context)
        - Causal relationships (action → outcome)
        - Temporal sequences (experiences that followed in time)
        - Sensory similarities (similar perceptual context)
        """
        connected = []
        
        # Spatial connections - experiences with similar sensory context
        if hasattr(experience, 'mental_context') and experience.mental_context:
            spatial_neighbors = self.world_graph.find_similar_nodes(
                experience.mental_context, similarity_threshold=0.7, max_results=10
            )
            
            for neighbor in spatial_neighbors:
                if id(neighbor) != id(experience):
                    # Calculate connection strength based on similarity
                    strength = self._calculate_experience_similarity(experience, neighbor)
                    if strength > 0.3:
                        connected.append((neighbor, strength * self.connection_types['spatial']))
        
        # Causal connections - experiences that followed similar actions
        if hasattr(experience, 'action_taken') and experience.action_taken:
            causal_neighbors = self._find_causal_connections(experience)
            for neighbor, causal_strength in causal_neighbors:
                connected.append((neighbor, causal_strength * self.connection_types['causal']))
        
        # Temporal connections - experiences that are temporally close
        temporal_neighbors = self._find_temporal_connections(experience)
        for neighbor, temporal_strength in temporal_neighbors:
            connected.append((neighbor, temporal_strength * self.connection_types['temporal']))
        
        # Remove duplicates and sort by strength
        unique_connected = {}
        for exp, strength in connected:
            exp_id = id(exp)
            if exp_id not in unique_connected or strength > unique_connected[exp_id][1]:
                unique_connected[exp_id] = (exp, strength)
        
        result = list(unique_connected.values())
        result.sort(key=lambda x: x[1], reverse=True)
        
        return result[:15]  # Limit to top 15 connections
    
    def _calculate_experience_similarity(self, exp1: ExperienceNode, exp2: ExperienceNode) -> float:
        """Calculate similarity between two experiences."""
        if not (hasattr(exp1, 'mental_context') and hasattr(exp2, 'mental_context')):
            return 0.0
        
        if not (exp1.mental_context and exp2.mental_context):
            return 0.0
        
        # Euclidean distance similarity
        context1 = exp1.mental_context
        context2 = exp2.mental_context
        
        min_len = min(len(context1), len(context2))
        if min_len == 0:
            return 0.0
        
        differences = [(c1 - c2) ** 2 for c1, c2 in zip(context1[:min_len], context2[:min_len])]
        distance = (sum(differences) / len(differences)) ** 0.5
        
        return max(0.0, 1.0 - distance)
    
    def _find_causal_connections(self, experience: ExperienceNode) -> List[Tuple[ExperienceNode, float]]:
        """Find experiences causally connected to this one."""
        connections = []
        
        if not (hasattr(experience, 'action_taken') and experience.action_taken):
            return connections
        
        # Find experiences with similar actions
        all_experiences = self.world_graph.get_all_nodes()
        
        for other_exp in all_experiences[:200]:  # Limit search for performance
            if id(other_exp) == id(experience):
                continue
            
            if hasattr(other_exp, 'action_taken') and other_exp.action_taken:
                action_similarity = self._calculate_action_similarity(
                    experience.action_taken, other_exp.action_taken
                )
                
                if action_similarity > 0.6:
                    connections.append((other_exp, action_similarity))
        
        return connections[:5]  # Top 5 causal connections
    
    def _calculate_action_similarity(self, action1: Dict[str, float], action2: Dict[str, float]) -> float:
        """Calculate similarity between two actions."""
        if not action1 or not action2:
            return 0.0
        
        common_keys = set(action1.keys()) & set(action2.keys())
        if not common_keys:
            return 0.0
        
        total_diff = sum(abs(action1[key] - action2[key]) for key in common_keys)
        avg_diff = total_diff / len(common_keys)
        
        return max(0.0, 1.0 - avg_diff)
    
    def _find_temporal_connections(self, experience: ExperienceNode) -> List[Tuple[ExperienceNode, float]]:
        """Find experiences temporally connected to this one."""
        connections = []
        
        # Simple temporal connection based on experience ordering in world graph
        # In a real implementation, this would use timestamps
        all_experiences = self.world_graph.get_all_nodes()
        
        try:
            exp_index = all_experiences.index(experience)
            
            # Look at nearby experiences in the sequence
            for offset in [-2, -1, 1, 2]:
                neighbor_index = exp_index + offset
                if 0 <= neighbor_index < len(all_experiences):
                    neighbor = all_experiences[neighbor_index]
                    # Temporal strength decreases with distance
                    strength = 1.0 / (abs(offset) + 1)
                    connections.append((neighbor, strength))
        except ValueError:
            pass  # Experience not found in list
        
        return connections
    
    def _experience_heuristic(self, experience: ExperienceNode, target: ExperienceNode) -> float:
        """
        Heuristic function for A* search - estimates distance to target experience.
        
        This uses experience-based similarity rather than geometric distance.
        """
        similarity = self._calculate_experience_similarity(experience, target)
        return 1.0 - similarity  # Distance is inverse of similarity
    
    def _create_experience_path(
        self, 
        experience_sequence: List[ExperienceNode],
        start: ExperienceNode,
        target: ExperienceNode
    ) -> ExperiencePath:
        """Create an ExperiencePath object from a sequence of experiences."""
        # Analyze connection types in the path
        connection_types = []
        total_strength = 0.0
        
        for i in range(len(experience_sequence) - 1):
            current_exp = experience_sequence[i]
            next_exp = experience_sequence[i + 1]
            
            # Determine primary connection type
            connection_type = self._determine_connection_type(current_exp, next_exp)
            connection_types.append(connection_type)
            
            # Add to total strength
            similarity = self._calculate_experience_similarity(current_exp, next_exp)
            total_strength += similarity
        
        # Calculate path metrics
        path_strength = total_strength / max(1, len(connection_types))
        path_confidence = min(1.0, path_strength * 1.2)  # Boost confidence slightly
        
        path_id = f"path_{id(start)}_{id(target)}_{int(time.time()*1000)}"
        
        return ExperiencePath(
            path_id=path_id,
            experience_sequence=experience_sequence,
            connection_types=connection_types,
            path_strength=path_strength,
            path_confidence=path_confidence,
            estimated_steps=len(experience_sequence),
            path_metadata={
                'start_experience_id': id(start),
                'target_experience_id': id(target),
                'dominant_connection_type': max(set(connection_types), key=connection_types.count) if connection_types else 'unknown',
                'path_diversity': len(set(connection_types)),
                'creation_timestamp': time.time()
            }
        )
    
    def _determine_connection_type(self, exp1: ExperienceNode, exp2: ExperienceNode) -> str:
        """Determine the primary type of connection between two experiences."""
        # Spatial connection - similar sensory context
        if hasattr(exp1, 'mental_context') and hasattr(exp2, 'mental_context'):
            spatial_sim = self._calculate_experience_similarity(exp1, exp2)
            if spatial_sim > 0.7:
                return 'spatial'
        
        # Causal connection - similar actions
        if (hasattr(exp1, 'action_taken') and hasattr(exp2, 'action_taken') and 
            exp1.action_taken and exp2.action_taken):
            action_sim = self._calculate_action_similarity(exp1.action_taken, exp2.action_taken)
            if action_sim > 0.6:
                return 'causal'
        
        # Sensory connection - similar perceptual content
        if hasattr(exp1, 'mental_context') and hasattr(exp2, 'mental_context'):
            sensory_sim = self._calculate_experience_similarity(exp1, exp2)
            if sensory_sim > 0.5:
                return 'sensory'
        
        return 'temporal'  # Default to temporal connection
    
    def _select_best_frontier_path(self, frontier_paths: List[ExperiencePath]) -> Optional[ExperiencePath]:
        """Select the best path from multiple frontier paths."""
        if not frontier_paths:
            return None
        
        # Score paths based on multiple criteria
        scored_paths = []
        for path in frontier_paths:
            score = (
                path.path_strength * 0.4 +           # Path strength
                path.path_confidence * 0.3 +         # Path confidence
                (1.0 / max(1, path.estimated_steps)) * 0.2 +  # Shorter is better
                path.path_metadata.get('path_diversity', 1) * 0.1  # Diversity bonus
            )
            scored_paths.append((score, path))
        
        # Return highest scored path
        scored_paths.sort(key=lambda x: x[0], reverse=True)
        return scored_paths[0][1]
    
    def _generate_frontier_target_context(self, frontier: ExperienceFrontier) -> Optional[List[float]]:
        """Generate target context for action/conceptual frontiers."""
        if frontier.frontier_type.value == 'action':
            # For action frontiers, target is context where action would be tried
            return frontier.metadata.get('sensory_signature', [])
        elif frontier.frontier_type.value == 'conceptual':
            # For conceptual frontiers, target is context with similar patterns
            return frontier.metadata.get('sensory_signature', [])
        
        return None
    
    def get_pathfinding_statistics(self) -> Dict[str, Any]:
        """Get comprehensive pathfinding statistics."""
        return {
            'pathfinding_requests': self.pathfinding_requests,
            'successful_paths': self.successful_paths,
            'success_rate': self.successful_paths / max(1, self.pathfinding_requests),
            'cache_hits': self.cache_hits,
            'cache_hit_rate': self.cache_hits / max(1, self.pathfinding_requests),
            'connection_discoveries': self.connection_discoveries,
            'cached_paths': len(self.path_cache),
            'connection_types': self.connection_types,
            'average_path_length': 3.5  # Placeholder - would track actual average
        }
    
    def reset_session(self):
        """Reset pathfinding state for new session."""
        self.connection_cache.clear()
        self.path_cache.clear()
        print("🗺️ Experience Pathfinding session reset")