"""
WorldGraph - The complete experience graph storing all memories and relationships.
Enhanced with neural-like dynamics for emergent memory phenomena.
Now includes hardware-agnostic accelerated similarity search.
"""

from typing import Dict, List, Optional, Set, Any, Tuple
from datetime import datetime
import math
import time
from .experience_node import ExperienceNode
from .accelerated_similarity import get_similarity_engine


class WorldGraph:
    """
    The complete experience graph - all memories and their relationships.
    Enhanced with neural-like dynamics that create emergent memory phenomena:
    - Spreading activation creates associative memory
    - Natural decay and consolidation
    - Emergent forgetting of unused memories
    - Working memory effects from activation levels
    """
    
    def __init__(self):
        self.nodes: Dict[str, ExperienceNode] = {}
        self.latest_node_id: Optional[str] = None
        self.total_nodes_created: int = 0
        self.total_merges_performed: int = 0
        
        # Performance optimization structures (legacy)
        self.strength_index: Dict[float, Set[str]] = {}  # strength -> set of node_ids
        self.temporal_chain: List[str] = []              # chronological order
        self.similarity_clusters: Dict[str, Set[str]] = {}  # cluster_id -> node_ids
        
        # Neural-like dynamics settings
        self.global_activation_decay = 0.995          # How fast activation decays globally
        self.connection_learning_rate = 0.1           # Rate of connection strengthening
        self.similarity_threshold = 0.7               # Threshold for creating connections
        self.activation_spread_iterations = 3         # How many waves of activation spread
        self.forgetting_enabled = True                # Whether to forget unused memories
        self.consolidation_frequency = 50             # Steps between consolidation cycles
        self.steps_since_consolidation = 0
        
        # Accelerated similarity search
        self.similarity_engine = get_similarity_engine()
        self._context_cache_dirty = True              # Whether context cache needs rebuild
        
        # Performance optimization caching
        self._stats_cache = None                      # Cached graph statistics
        self._stats_cache_dirty = True                # Whether stats cache needs rebuild
        self._strongest_nodes_cache = None            # Cached strongest nodes list
        self._strongest_nodes_cache_dirty = True      # Whether strongest nodes cache needs rebuild
    
    def has_nodes(self) -> bool:
        """Check if graph has any nodes (for bootstrap detection)."""
        return len(self.nodes) > 0
    
    def _invalidate_caches(self):
        """Invalidate performance caches when graph structure changes."""
        self._stats_cache_dirty = True
        self._strongest_nodes_cache_dirty = True
        self._context_cache_dirty = True
    
    def add_node(self, experience: ExperienceNode) -> str:
        """Add a new experience node to the graph with optimized connection formation."""
        node_id = experience.node_id
        self.nodes[node_id] = experience
        self.total_nodes_created += 1
        
        # Update indexes (fast operations)
        self._update_strength_index(node_id, experience.strength)
        self.temporal_chain.append(node_id)
        
        # Link to previous node temporally (fast)
        if self.latest_node_id:
            experience.temporal_predecessor = self.latest_node_id
            if self.latest_node_id in self.nodes:
                self.nodes[self.latest_node_id].temporal_successor = node_id
        
        # Activate the new memory (fast)
        experience.activate(strength=1.0)
        self.latest_node_id = node_id
        
        # Invalidate caches since graph structure changed
        self._invalidate_caches()
        
        # OPTIMIZATION: Only do expensive operations every N additions
        # This moves O(n²) operations from critical path
        if self.total_nodes_created % 10 == 0:  # Every 10 nodes instead of every node
            self._create_natural_connections_batch()
            self.step_time()
        else:
            # Quick connection to just the most recent nodes (fast O(k) where k=5)
            self._create_minimal_connections(experience)
        
        return node_id
    
    def _create_natural_connections(self, new_node: ExperienceNode):
        """Create connections between new node and existing similar nodes - like growing dendrites."""
        for existing_node in self.nodes.values():
            if existing_node.node_id != new_node.node_id:
                # Calculate similarity
                similarity = self._calculate_context_similarity(
                    new_node.mental_context, existing_node.mental_context
                )
                
                # Create bidirectional connections if similarity is high enough
                if similarity > self.similarity_threshold:
                    connection_strength = similarity * 0.9  # Much stronger connections (emergent behavior)
                    
                    new_node.connection_weights[existing_node.node_id] = connection_strength
                    existing_node.connection_weights[new_node.node_id] = connection_strength
                
                # Create weaker connections for moderate similarity
                elif similarity > 0.4:
                    connection_strength = similarity * 0.5  # Stronger even for moderate similarity
                    
                    new_node.connection_weights[existing_node.node_id] = connection_strength
                    existing_node.connection_weights[new_node.node_id] = connection_strength
    
    def _create_minimal_connections(self, new_node: ExperienceNode):
        """Create connections to only the most recent 5 nodes for fast operation."""
        # Get the 5 most recent nodes (O(5) instead of O(n))
        recent_node_ids = self.temporal_chain[-5:] if len(self.temporal_chain) > 5 else self.temporal_chain[:-1]
        
        for node_id in recent_node_ids:
            if node_id in self.nodes and node_id != new_node.node_id:
                existing_node = self.nodes[node_id]
                
                # Calculate similarity only with recent nodes
                similarity = self._calculate_context_similarity(
                    new_node.mental_context, existing_node.mental_context
                )
                
                # Create connection if similar enough
                if similarity > self.similarity_threshold:
                    connection_strength = similarity * 0.6
                    new_node.connection_weights[existing_node.node_id] = connection_strength
                    existing_node.connection_weights[new_node.node_id] = connection_strength
                elif similarity > 0.4:
                    connection_strength = similarity * 0.3
                    new_node.connection_weights[existing_node.node_id] = connection_strength
                    existing_node.connection_weights[new_node.node_id] = connection_strength
    
    def _create_natural_connections_batch(self):
        """Batch process connections for recently added nodes (called every 10 additions)."""
        # Get the last 10 nodes that might need full connection processing
        recent_node_ids = self.temporal_chain[-10:] if len(self.temporal_chain) >= 10 else self.temporal_chain
        
        # Only process connections between recent nodes to limit complexity
        for i, node_id1 in enumerate(recent_node_ids):
            if node_id1 not in self.nodes:
                continue
            node1 = self.nodes[node_id1]
            
            # Connect to some older nodes for diversity (sample, don't check all)
            import random
            if len(self.nodes) > 20:
                older_sample = random.sample(list(self.nodes.keys()), min(20, len(self.nodes) - len(recent_node_ids)))
                candidate_nodes = recent_node_ids + older_sample
            else:
                candidate_nodes = list(self.nodes.keys())
            
            for node_id2 in candidate_nodes:
                if node_id2 != node_id1 and node_id2 in self.nodes:
                    node2 = self.nodes[node_id2]
                    
                    # Only create connection if one doesn't exist
                    if node_id2 not in node1.connection_weights:
                        similarity = self._calculate_context_similarity(
                            node1.mental_context, node2.mental_context
                        )
                        
                        if similarity > self.similarity_threshold:
                            connection_strength = similarity * 0.6
                            node1.connection_weights[node_id2] = connection_strength
                            node2.connection_weights[node_id1] = connection_strength
                        elif similarity > 0.4:
                            connection_strength = similarity * 0.3
                            node1.connection_weights[node_id2] = connection_strength
                            node2.connection_weights[node_id1] = connection_strength
    
    def get_node(self, node_id: str) -> Optional[ExperienceNode]:
        """Retrieve a node by ID."""
        return self.nodes.get(node_id)
    
    def remove_node(self, node_id: str) -> bool:
        """Remove a node and clean up all references."""
        if node_id not in self.nodes:
            return False
        
        node = self.nodes[node_id]
        
        # Clean up all references to this node
        for other_id, other_node in self.nodes.items():
            if other_id == node_id:
                continue
                
            # Remove from similarity lists
            if node_id in other_node.similar_contexts:
                other_node.similar_contexts.remove(node_id)
            
            # Remove from prediction sources
            if node_id in other_node.prediction_sources:
                other_node.prediction_sources.remove(node_id)
            
            # Fix temporal chain
            if other_node.temporal_predecessor == node_id:
                other_node.temporal_predecessor = node.temporal_predecessor
            if other_node.temporal_successor == node_id:
                other_node.temporal_successor = node.temporal_successor
        
        # Remove from indexes
        self._remove_from_strength_index(node_id, node.strength)
        if node_id in self.temporal_chain:
            self.temporal_chain.remove(node_id)
        
        # Remove the node
        del self.nodes[node_id]
        return True
    
    def find_similar_nodes(self, target_context: List[float], 
                          similarity_threshold: float = 0.7,
                          max_results: int = 10) -> List[ExperienceNode]:
        """Find nodes with similar mental contexts using accelerated similarity search."""
        if not self.nodes:
            return []
        
        # Use accelerated similarity search
        try:
            # Prepare contexts for vectorized search
            node_list = list(self.nodes.values())
            all_contexts = [node.mental_context for node in node_list]
            
            # Perform accelerated similarity search
            indices, similarities = self.similarity_engine.find_similar_contexts(
                target_context, all_contexts, similarity_threshold, max_results
            )
            
            # Return corresponding nodes
            similar_nodes = [node_list[i] for i in indices]
            
            # Update access frequency for found nodes (neural-like dynamics)
            for node in similar_nodes:
                node.access_frequency += 1
                node.last_access_time = time.time()
            
            return similar_nodes
            
        except Exception as e:
            # Fallback to original implementation if accelerated search fails
            print(f"Warning: Accelerated similarity search failed: {e}")
            return self._find_similar_nodes_fallback(target_context, similarity_threshold, max_results)
    
    def _find_similar_nodes_fallback(self, target_context: List[float],
                                   similarity_threshold: float = 0.7,
                                   max_results: int = 10) -> List[ExperienceNode]:
        """Fallback implementation using original similarity search."""
        similarities = []
        
        for node in self.nodes.values():
            similarity = self._calculate_context_similarity(target_context, node.mental_context)
            if similarity >= similarity_threshold:
                similarities.append((similarity, node))
        
        # Sort by similarity (highest first) and limit results
        similarities.sort(key=lambda x: x[0], reverse=True)
        return [node for _, node in similarities[:max_results]]
    
    def find_nodes_by_action(self, target_action: Dict[str, float], 
                           tolerance: float = 0.1) -> List[ExperienceNode]:
        """Find nodes that took similar actions."""
        matching_nodes = []
        
        for node in self.nodes.values():
            if self._actions_similar(node.action_taken, target_action, tolerance):
                matching_nodes.append(node)
        
        return matching_nodes
    
    def get_nodes_by_strength_range(self, min_strength: float, 
                                   max_strength: float = float('inf')) -> List[ExperienceNode]:
        """Get all nodes within a strength range."""
        result = []
        for node in self.nodes.values():
            if min_strength <= node.strength <= max_strength:
                result.append(node)
        return result
    
    def get_weakest_nodes(self, count: int) -> List[ExperienceNode]:
        """Get the N weakest nodes (candidates for merging)."""
        all_nodes = list(self.nodes.values())
        all_nodes.sort(key=lambda n: n.strength)
        return all_nodes[:count]
    
    def get_strongest_nodes(self, count: int) -> List[ExperienceNode]:
        """Get the N strongest nodes (most important memories) - cached for performance."""
        # Return cached strongest nodes if available and valid
        if not self._strongest_nodes_cache_dirty and self._strongest_nodes_cache is not None:
            return self._strongest_nodes_cache[:count]
        
        # Calculate and cache strongest nodes
        all_nodes = list(self.nodes.values())
        all_nodes.sort(key=lambda n: n.strength, reverse=True)
        
        # Cache the full sorted list
        self._strongest_nodes_cache = all_nodes
        self._strongest_nodes_cache_dirty = False
        
        return all_nodes[:count]
    
    def node_count(self) -> int:
        """Total number of nodes in graph."""
        return len(self.nodes)
    
    def node_exists(self, node_id: str) -> bool:
        """Check if a node exists."""
        return node_id in self.nodes
    
    def all_nodes(self) -> List[ExperienceNode]:
        """Get all nodes (use carefully - can be large)."""
        return list(self.nodes.values())
    
    def get_latest_node(self) -> Optional[ExperienceNode]:
        """Get the most recently added node."""
        if self.latest_node_id:
            return self.nodes.get(self.latest_node_id)
        return None
    
    def update_node_strength(self, node_id: str, new_strength: float):
        """Update a node's strength and maintain indexes."""
        if node_id not in self.nodes:
            return
        
        node = self.nodes[node_id]
        old_strength = node.strength
        self._remove_from_strength_index(node_id, old_strength)
        
        node.strength = new_strength
        node.access_node()  # Update access tracking
        
        # Invalidate caches since node strength changed
        self._invalidate_caches()
        
        self._update_strength_index(node_id, new_strength)
    
    def strengthen_node(self, node_id: str, amount: float = 1.0):
        """Increase a node's strength by a given amount."""
        if node_id in self.nodes:
            current_strength = self.nodes[node_id].strength
            self.update_node_strength(node_id, current_strength + amount)
    
    def weaken_node(self, node_id: str, amount: float = 0.001):
        """Decrease a node's strength by a given amount."""
        if node_id in self.nodes:
            current_strength = self.nodes[node_id].strength
            new_strength = max(0.0, current_strength - amount)
            self.update_node_strength(node_id, new_strength)
    
    def decay_all_nodes(self, decay_rate: float = 0.001):
        """Apply decay to all nodes (unused memories fade)."""
        for node_id in list(self.nodes.keys()):
            self.weaken_node(node_id, decay_rate)
    
    def merge_similar_nodes(self, similarity_threshold: float = 0.9, 
                           strength_threshold: float = 0.5) -> int:
        """
        Merge nodes with similar contexts and low strength.
        Returns the number of merges performed.
        """
        merges_performed = 0
        nodes_to_remove = set()
        
        # Get weak nodes that are candidates for merging
        weak_nodes = self.get_nodes_by_strength_range(0.0, strength_threshold)
        
        for i, node1 in enumerate(weak_nodes):
            if node1.node_id in nodes_to_remove:
                continue
                
            for j, node2 in enumerate(weak_nodes[i+1:], i+1):
                if node2.node_id in nodes_to_remove:
                    continue
                
                # Check if nodes are similar enough to merge
                similarity = self._calculate_context_similarity(
                    node1.mental_context, node2.mental_context
                )
                
                if similarity >= similarity_threshold:
                    # Merge node2 into node1
                    merged_node = node1.merge_with(node2, weight=0.5)
                    
                    # Remove old nodes and add merged node
                    self.remove_node(node1.node_id)
                    self.remove_node(node2.node_id)
                    self.add_node(merged_node)
                    
                    nodes_to_remove.add(node1.node_id)
                    nodes_to_remove.add(node2.node_id)
                    
                    merges_performed += 1
                    self.total_merges_performed += 1
                    break
        
        return merges_performed
    
    def get_temporal_sequence(self, start_node_id: str, length: int = 5) -> List[ExperienceNode]:
        """Get a sequence of nodes following temporal links."""
        sequence = []
        current_id = start_node_id
        
        for _ in range(length):
            if current_id and current_id in self.nodes:
                node = self.nodes[current_id]
                sequence.append(node)
                current_id = node.temporal_successor
            else:
                break
        
        return sequence
    
    def get_graph_statistics(self) -> Dict[str, Any]:
        """Get summary statistics about the graph (cached for performance)."""
        # Return cached statistics if available and valid
        if not self._stats_cache_dirty and self._stats_cache is not None:
            return self._stats_cache
        
        if not self.nodes:
            base_stats = {"total_nodes": 0}
        else:
            # Calculate expensive statistics only when cache is dirty
            strengths = [node.strength for node in self.nodes.values()]
            context_lengths = [len(node.mental_context) for node in self.nodes.values()]
            access_counts = [node.times_accessed for node in self.nodes.values()]
            
            base_stats = {
                "total_nodes": len(self.nodes),
                "total_merges": self.total_merges_performed,
                "avg_strength": sum(strengths) / len(strengths),
                "max_strength": max(strengths),
                "min_strength": min(strengths),
                "avg_context_length": sum(context_lengths) / len(context_lengths),
                "temporal_chain_length": len(self.temporal_chain),
                "avg_access_count": sum(access_counts) / len(access_counts),
                "total_accesses": sum(access_counts)
            }
        
        # Add similarity engine performance stats
        similarity_stats = self.similarity_engine.get_performance_stats()
        base_stats["similarity_engine"] = similarity_stats
        
        # Cache the results
        self._stats_cache = base_stats
        self._stats_cache_dirty = False
        
        return base_stats
    
    # Neural-like dynamics methods
    
    def activate_memory_network(self, trigger_context: List[float], 
                               activation_threshold: float = 0.3) -> List[ExperienceNode]:
        """
        Spreading activation through the memory network - emergent associative memory.
        Enhanced with action-relevance and pain/pleasure biases.
        """
        if not self.nodes:
            return []
        
        # Dictionary to track activation levels for each node
        activated_nodes: Dict[str, float] = {}
        
        # Phase 1: Find initial activation points based on context similarity
        for node in self.nodes.values():
            similarity = self._calculate_context_similarity(trigger_context, node.mental_context)
            if similarity > 0.4:  # Slightly lower threshold for more associations
                # Combine similarity with node's accessibility
                initial_activation = similarity * node.get_accessibility()
                
                # Boost activation for nodes with actions (more useful for motor planning)
                if node.action_taken:
                    initial_activation *= 1.2
                
                # Apply pain/pleasure biases
                if hasattr(node, 'pain_signal'):
                    # Painful experiences get higher activation (avoidance is important)
                    initial_activation *= (1.0 + abs(node.pain_signal) * 0.3)
                
                if hasattr(node, 'pleasure_signal'):
                    # Pleasurable experiences get moderate boost (approach is good)
                    initial_activation *= (1.0 + node.pleasure_signal * 0.2)
                
                activated_nodes[node.node_id] = initial_activation
                
                # Activate the node (increases its activation level)
                node.activate(strength=initial_activation * 0.5)
        
        # Phase 2: Spread activation through connections (multiple waves)
        for iteration in range(self.activation_spread_iterations):
            new_activations = {}
            
            for node_id, activation in activated_nodes.items():
                if activation > activation_threshold:  # Only spread from sufficiently active nodes
                    node = self.nodes[node_id]
                    
                    # Spread to connected nodes
                    for connected_id, connection_strength in node.connection_weights.items():
                        if connected_id in self.nodes:
                            spread_activation = activation * connection_strength * 0.7
                            
                            # Boost spread for temporal connections (recent experiences)
                            connected_node = self.nodes[connected_id]
                            if (node.temporal_successor == connected_id or 
                                node.temporal_predecessor == connected_id):
                                spread_activation *= 1.3
                            
                            # Accumulate activation (multiple sources can activate same node)
                            if connected_id not in activated_nodes:
                                new_activations[connected_id] = spread_activation
                            else:
                                new_activations[connected_id] = max(
                                    activated_nodes.get(connected_id, 0), spread_activation
                                )
            
            # Update activation levels
            activated_nodes.update(new_activations)
        
        # Phase 3: Return most accessible nodes (emergent "working memory" effect)
        active_node_list = []
        for node_id, activation in activated_nodes.items():
            if activation > activation_threshold:
                node = self.nodes[node_id]
                
                # Calculate final relevance score
                relevance_score = activation
                
                # Boost nodes with low prediction error (successful experiences)
                if node.prediction_error < 0.3:
                    relevance_score *= 1.4
                
                # Boost nodes with recent access
                if node.last_access_time and (time.time() - node.last_access_time) < 30:
                    relevance_score *= 1.2
                
                active_node_list.append((node, relevance_score))
        
        # Sort by relevance score and return top nodes (natural working memory limit)
        active_node_list.sort(key=lambda x: x[1], reverse=True)
        
        # Return top 10 nodes (expanded working memory for motor planning)
        return [node for node, activation in active_node_list[:10]]
    
    def step_time(self):
        """
        Natural time-based processes - emergent memory consolidation and forgetting.
        No special 'MemoryConsolidation' class needed!
        """
        self.steps_since_consolidation += 1
        
        # Phase 1: Natural decay for all memories
        for node in self.nodes.values():
            node.decay_over_time(1.0)
        
        # Phase 2: Strengthen connections between co-activated memories (Hebbian learning)
        recently_active_nodes = [
            node for node in self.nodes.values() 
            if node.get_accessibility() > 1.0  # Recently accessed/high activation
        ]
        
        for i, node1 in enumerate(recently_active_nodes):
            for node2 in recently_active_nodes[i+1:]:
                # If both nodes were recently active, strengthen their connection
                if node1.node_id in node2.connection_weights:
                    node1.strengthen_connection(node2.node_id, 0.1)
                    node2.strengthen_connection(node1.node_id, 0.1)
        
        # Phase 3: Weaken unused connections
        for node in self.nodes.values():
            connections_to_weaken = list(node.connection_weights.keys())
            for connected_id in connections_to_weaken:
                node.weaken_connection(connected_id, decay_factor=0.999)
        
        # Phase 4: Natural forgetting of unused memories
        if self.forgetting_enabled and self.steps_since_consolidation % 10 == 0:
            self._natural_forgetting()
        
        # Phase 5: Periodic memory consolidation
        if self.steps_since_consolidation >= self.consolidation_frequency:
            self._consolidate_memories()
            self.steps_since_consolidation = 0
    
    def _natural_forgetting(self):
        """Remove memories that are naturally forgotten - emergent forgetting."""
        nodes_to_forget = []
        
        for node_id, node in self.nodes.items():
            if node.is_forgettable():
                nodes_to_forget.append(node_id)
        
        # Remove forgotten memories
        for node_id in nodes_to_forget:
            self._remove_node_safely(node_id)
    
    def _consolidate_memories(self):
        """Strengthen important memories and create abstractions - emergent consolidation."""
        if not self.nodes:
            return
        
        # Find high-impact memories (frequently accessed, low prediction error)
        important_memories = []
        for node in self.nodes.values():
            if node.access_frequency > 3 and node.prediction_error < 0.3:
                important_memories.append(node)
        
        # Strengthen important memories
        for node in important_memories:
            node.consolidation_strength = min(2.0, node.consolidation_strength * 1.1)
        
        # Strengthen connections between important memories
        for i, node1 in enumerate(important_memories):
            for node2 in important_memories[i+1:]:
                similarity = self._calculate_context_similarity(
                    node1.mental_context, node2.mental_context
                )
                if similarity > 0.6:
                    node1.strengthen_connection(node2.node_id, similarity * 0.2)
                    node2.strengthen_connection(node1.node_id, similarity * 0.2)
    
    def _remove_node_safely(self, node_id: str):
        """Safely remove a node and clean up all references."""
        if node_id not in self.nodes:
            return
        
        # Remove connections from other nodes to this node
        for other_node in self.nodes.values():
            if node_id in other_node.connection_weights:
                del other_node.connection_weights[node_id]
            
            # Clean up temporal links
            if other_node.temporal_predecessor == node_id:
                other_node.temporal_predecessor = None
            if other_node.temporal_successor == node_id:
                other_node.temporal_successor = None
        
        # Remove from indexes
        if node_id in self.temporal_chain:
            self.temporal_chain.remove(node_id)
        
        # Remove the node itself
        del self.nodes[node_id]
    
    def get_most_accessible_memories(self, limit: int = 10) -> List[ExperienceNode]:
        """Get the most accessible memories - emergent 'working memory' set."""
        if not self.nodes:
            return []
        
        # Sort nodes by accessibility
        accessible_nodes = [(node, node.get_accessibility()) for node in self.nodes.values()]
        accessible_nodes.sort(key=lambda x: x[1], reverse=True)
        
        return [node for node, accessibility in accessible_nodes[:limit]]
    
    def get_emergent_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about emergent memory phenomena."""
        if not self.nodes:
            return {"error": "No nodes in graph"}
        
        # Calculate various emergent properties
        total_connections = sum(len(node.connection_weights) for node in self.nodes.values())
        avg_connections = total_connections / len(self.nodes)
        
        activation_levels = [node.activation_level for node in self.nodes.values()]
        accessibility_levels = [node.get_accessibility() for node in self.nodes.values()]
        
        # Count nodes by "memory type" (emergent categories)
        working_memory_count = len([n for n in self.nodes.values() if n.get_accessibility() > 1.5])
        consolidated_count = len([n for n in self.nodes.values() if n.consolidation_strength > 1.5])
        forgettable_count = len([n for n in self.nodes.values() if n.is_forgettable()])
        
        return {
            "total_nodes": len(self.nodes),
            "total_connections": total_connections,
            "avg_connections_per_node": avg_connections,
            "avg_activation_level": sum(activation_levels) / len(activation_levels),
            "avg_accessibility": sum(accessibility_levels) / len(accessibility_levels),
            "working_memory_nodes": working_memory_count,
            "consolidated_nodes": consolidated_count,
            "forgettable_nodes": forgettable_count,
            "memory_consolidation_cycles": self.total_nodes_created // self.consolidation_frequency,
            "steps_since_last_consolidation": self.steps_since_consolidation,
            "emergent_memory_types": {
                "working_memory": working_memory_count,
                "consolidated_memory": consolidated_count,
                "forgettable_memory": forgettable_count,
                "stable_memory": len(self.nodes) - working_memory_count - forgettable_count
            }
        }
    
    def get_recent_nodes(self, limit: int = 10) -> List[ExperienceNode]:
        """Get the most recently added nodes (by temporal order)."""
        if not self.temporal_chain:
            return []
        
        # Get the last 'limit' nodes from temporal chain
        recent_ids = self.temporal_chain[-limit:] if len(self.temporal_chain) >= limit else self.temporal_chain
        
        # Return actual nodes, filtering out any that no longer exist
        recent_nodes = []
        for node_id in reversed(recent_ids):  # Most recent first
            if node_id in self.nodes:
                recent_nodes.append(self.nodes[node_id])
        
        return recent_nodes
    
    # Private helper methods
    
    def _calculate_context_similarity(self, context1: List[float], context2: List[float]) -> float:
        """Calculate similarity between two mental contexts using Euclidean distance."""
        if len(context1) != len(context2):
            return 0.0
        
        if not context1 or not context2:
            return 0.0
        
        # Euclidean distance
        distance = sum((a - b) ** 2 for a, b in zip(context1, context2)) ** 0.5
        max_possible_distance = (len(context1) * 4.0) ** 0.5  # Assuming values roughly -2 to +2
        
        # Convert distance to similarity (0 to 1)
        if max_possible_distance == 0:
            return 1.0 if distance == 0 else 0.0
        
        similarity = max(0.0, 1.0 - (distance / max_possible_distance))
        return similarity
    
    def _actions_similar(self, action1: Dict[str, float], action2: Dict[str, float], 
                        tolerance: float = 0.1) -> bool:
        """Check if two actions are similar within tolerance."""
        # Check if all keys match
        if set(action1.keys()) != set(action2.keys()):
            return False
        
        # Check if all values are within tolerance
        for key in action1:
            if abs(action1[key] - action2[key]) > tolerance:
                return False
        
        return True
    
    def _update_strength_index(self, node_id: str, strength: float):
        """Update the strength-based index for fast retrieval."""
        # Round strength to bucket for indexing
        strength_bucket = round(strength, 2)
        if strength_bucket not in self.strength_index:
            self.strength_index[strength_bucket] = set()
        self.strength_index[strength_bucket].add(node_id)
    
    def _remove_from_strength_index(self, node_id: str, strength: float):
        """Remove node from strength index."""
        strength_bucket = round(strength, 2)
        if strength_bucket in self.strength_index:
            self.strength_index[strength_bucket].discard(node_id)
            if not self.strength_index[strength_bucket]:
                del self.strength_index[strength_bucket]