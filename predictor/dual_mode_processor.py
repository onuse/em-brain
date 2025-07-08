"""
Dual-Mode Cognitive Processing System.

Implements two layers of thinking:
1. Immediate Mode: Fast, time-bounded predictions for real-time decisions
2. Background Mode: Continuous exploration, consolidation, and pattern discovery

The background mode "bubbles up" insights to immediate mode through:
- Pre-activation of relevant nodes
- Context priming and attention biasing  
- Pattern consolidation and memory organization
- Predictive caching of likely scenarios
"""

import time
import threading
import queue
import random
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict
from enum import Enum

from core.world_graph import WorldGraph
from core.experience_node import ExperienceNode
from core.communication import PredictionPacket
from .auto_tuned_energy_traversal import AutoTunedEnergyTraversal, AutoTunedTraversalResult


class BackgroundTaskType(Enum):
    """Types of background processing tasks."""
    PATTERN_DISCOVERY = "pattern_discovery"
    MEMORY_CONSOLIDATION = "memory_consolidation"
    CONTEXT_MAPPING = "context_mapping"
    PREDICTIVE_CACHING = "predictive_caching"
    CONNECTION_OPTIMIZATION = "connection_optimization"


@dataclass
class PreActivation:
    """Pre-activation state for nodes that might be relevant soon."""
    node_id: str
    activation_level: float  # 0.0 to 1.0
    reason: str             # Why this node was pre-activated
    expires_at: float       # Timestamp when this expires
    confidence: float       # Confidence in the relevance


@dataclass
class ContextMap:
    """Map of contexts to pre-computed starting nodes."""
    context_signature: str  # Hash/signature of context pattern
    starting_nodes: List[str]  # Pre-computed good starting nodes
    success_rate: float     # Historical success rate of this mapping
    last_updated: float     # When this mapping was last updated
    usage_count: int = 0    # How often this mapping has been used


@dataclass
class BackgroundInsight:
    """Insight discovered by background processing."""
    insight_type: str       # Type of insight discovered
    content: Dict[str, Any] # The actual insight data
    confidence: float       # Confidence in this insight
    discovered_at: float    # When this was discovered
    relevance_contexts: List[str]  # Contexts where this might be relevant


@dataclass
class AttentionBias:
    """Bias weights for attention allocation."""
    context_type: str       # Type of context this applies to
    node_biases: Dict[str, float]  # node_id -> bias_weight
    connection_biases: Dict[str, float]  # connection_type -> bias_weight
    temporal_decay: float   # How quickly this bias decays
    created_at: float       # When this bias was created


class BackgroundProcessor:
    """
    Continuous background processing system that discovers patterns,
    consolidates memory, and prepares insights for immediate mode.
    """
    
    def __init__(self, world_graph: WorldGraph):
        self.world_graph = world_graph
        
        # Background processing state
        self.is_running = False
        self.background_thread = None
        self.task_queue = queue.Queue()
        
        # Pre-activation system
        self.pre_activations: Dict[str, PreActivation] = {}
        self.max_pre_activations = 100
        
        # Context mapping system
        self.context_maps: Dict[str, ContextMap] = {}
        self.max_context_maps = 50
        
        # Attention biasing system
        self.attention_biases: List[AttentionBias] = []
        self.max_attention_biases = 20
        
        # Background insights
        self.background_insights: List[BackgroundInsight] = []
        self.max_insights = 100
        
        # Processing configuration
        self.processing_interval = 0.1  # 100ms between background tasks
        self.consolidation_interval = 10.0  # 10 seconds between memory consolidation
        self.pattern_discovery_interval = 5.0  # 5 seconds between pattern discovery
        
        # Performance tracking
        self.tasks_processed = 0
        self.insights_discovered = 0
        self.pre_activations_used = 0
        self.context_maps_used = 0
        
        # Last processing times
        self.last_consolidation = 0.0
        self.last_pattern_discovery = 0.0
        self.last_context_mapping = 0.0
    
    def start_background_processing(self):
        """Start continuous background processing."""
        if self.is_running:
            return
        
        self.is_running = True
        self.background_thread = threading.Thread(target=self._background_loop, daemon=True)
        self.background_thread.start()
        print("🧠 Background processing started")
    
    def stop_background_processing(self):
        """Stop background processing."""
        self.is_running = False
        if self.background_thread:
            self.background_thread.join(timeout=1.0)
        print("🧠 Background processing stopped")
    
    def _background_loop(self):
        """Main background processing loop."""
        while self.is_running:
            try:
                current_time = time.time()
                
                # Process any queued tasks
                self._process_queued_tasks()
                
                # Periodic tasks based on intervals
                if current_time - self.last_consolidation > self.consolidation_interval:
                    self._schedule_task(BackgroundTaskType.MEMORY_CONSOLIDATION)
                    self.last_consolidation = current_time
                
                if current_time - self.last_pattern_discovery > self.pattern_discovery_interval:
                    self._schedule_task(BackgroundTaskType.PATTERN_DISCOVERY)
                    self.last_pattern_discovery = current_time
                
                if current_time - self.last_context_mapping > self.pattern_discovery_interval * 1.5:
                    self._schedule_task(BackgroundTaskType.CONTEXT_MAPPING)
                    self.last_context_mapping = current_time
                
                # Clean up expired pre-activations and biases
                self._cleanup_expired_state()
                
                # Sleep until next processing cycle
                time.sleep(self.processing_interval)
                
            except Exception as e:
                print(f"Background processing error: {e}")
                # Continue processing despite errors
    
    def _schedule_task(self, task_type: BackgroundTaskType, data: Dict = None):
        """Schedule a background processing task."""
        task = {
            'type': task_type,
            'data': data or {},
            'scheduled_at': time.time()
        }
        self.task_queue.put(task)
    
    def _process_queued_tasks(self):
        """Process all queued background tasks."""
        while not self.task_queue.empty():
            try:
                task = self.task_queue.get_nowait()
                self._execute_task(task)
                self.tasks_processed += 1
            except queue.Empty:
                break
            except Exception as e:
                print(f"Error processing background task: {e}")
    
    def _execute_task(self, task: Dict):
        """Execute a specific background task."""
        task_type = task['type']
        
        if task_type == BackgroundTaskType.MEMORY_CONSOLIDATION:
            self._perform_memory_consolidation()
        elif task_type == BackgroundTaskType.PATTERN_DISCOVERY:
            self._perform_pattern_discovery()
        elif task_type == BackgroundTaskType.CONTEXT_MAPPING:
            self._perform_context_mapping()
        elif task_type == BackgroundTaskType.PREDICTIVE_CACHING:
            self._perform_predictive_caching()
        elif task_type == BackgroundTaskType.CONNECTION_OPTIMIZATION:
            self._perform_connection_optimization()
    
    def _perform_memory_consolidation(self):
        """Consolidate memory by strengthening important connections."""
        # Find nodes that have been accessed frequently
        all_nodes = self.world_graph.all_nodes()
        
        # Sort by access frequency (or strength as proxy)
        frequent_nodes = sorted(all_nodes, key=lambda n: n.strength, reverse=True)[:20]
        
        # Strengthen connections between frequently accessed nodes
        for i, node1 in enumerate(frequent_nodes):
            for node2 in frequent_nodes[i+1:i+6]:  # Connect to next 5 nodes
                # Calculate connection strength based on context similarity
                similarity = self.world_graph._calculate_context_similarity(
                    node1.mental_context, node2.mental_context
                )
                
                if similarity > 0.7:  # High similarity threshold
                    # Strengthen bidirectional connection
                    current_weight1 = node1.connection_weights.get(node2.node_id, 0.0)
                    current_weight2 = node2.connection_weights.get(node1.node_id, 0.0)
                    
                    new_weight = min(1.0, max(current_weight1, current_weight2) + 0.1)
                    node1.connection_weights[node2.node_id] = new_weight
                    node2.connection_weights[node1.node_id] = new_weight
        
        # Create insight about consolidation
        insight = BackgroundInsight(
            insight_type="memory_consolidation",
            content={"nodes_processed": len(frequent_nodes), "connections_strengthened": True},
            confidence=0.8,
            discovered_at=time.time(),
            relevance_contexts=["all"]
        )
        self._add_insight(insight)
    
    def _perform_pattern_discovery(self):
        """Discover patterns in recent experiences."""
        # Look for patterns in recent node accesses
        all_nodes = self.world_graph.all_nodes()
        
        # Group nodes by similar contexts
        context_clusters = defaultdict(list)
        for node in all_nodes:
            # Create a rough context signature
            context_sig = self._create_context_signature(node.mental_context)
            context_clusters[context_sig].append(node)
        
        # Find interesting clusters (multiple nodes with high strength)
        interesting_clusters = []
        for sig, nodes in context_clusters.items():
            if len(nodes) >= 3:  # At least 3 nodes
                avg_strength = sum(n.strength for n in nodes) / len(nodes)
                if avg_strength > 50.0:  # High average strength
                    interesting_clusters.append((sig, nodes, avg_strength))
        
        # Create pre-activations for nodes in interesting clusters
        for sig, nodes, avg_strength in interesting_clusters[:5]:  # Top 5 clusters
            for node in nodes[:3]:  # Top 3 nodes per cluster
                self._add_pre_activation(
                    node.node_id,
                    activation_level=min(0.8, avg_strength / 100.0),
                    reason=f"pattern_cluster_{sig}",
                    duration=300.0  # 5 minutes
                )
        
        # Create insight about discovered patterns
        if interesting_clusters:
            insight = BackgroundInsight(
                insight_type="pattern_discovery",
                content={
                    "clusters_found": len(interesting_clusters),
                    "pattern_signatures": [sig for sig, _, _ in interesting_clusters]
                },
                confidence=0.7,
                discovered_at=time.time(),
                relevance_contexts=[sig for sig, _, _ in interesting_clusters]
            )
            self._add_insight(insight)
            self.insights_discovered += 1
    
    def _perform_context_mapping(self):
        """Build context maps for common scenarios."""
        # Analyze recent successful traversals to build context maps
        all_nodes = self.world_graph.all_nodes()
        
        # Group nodes by context patterns
        context_groups = defaultdict(list)
        for node in all_nodes:
            if node.strength > 30.0:  # Only consider reasonably strong nodes
                context_sig = self._create_context_signature(node.mental_context)
                context_groups[context_sig].append(node)
        
        # Create context maps for groups with multiple strong nodes
        for context_sig, nodes in context_groups.items():
            if len(nodes) >= 2:
                # Sort by strength and take top nodes as starting points
                top_nodes = sorted(nodes, key=lambda n: n.strength, reverse=True)[:5]
                
                context_map = ContextMap(
                    context_signature=context_sig,
                    starting_nodes=[n.node_id for n in top_nodes],
                    success_rate=0.8,  # Initial success rate estimate
                    last_updated=time.time()
                )
                
                self.context_maps[context_sig] = context_map
        
        # Limit number of context maps
        if len(self.context_maps) > self.max_context_maps:
            # Remove oldest maps
            oldest_maps = sorted(self.context_maps.items(), 
                               key=lambda x: x[1].last_updated)
            for sig, _ in oldest_maps[:len(self.context_maps) - self.max_context_maps]:
                del self.context_maps[sig]
    
    def _perform_predictive_caching(self):
        """Cache predictions for likely future scenarios."""
        # This could pre-compute predictions for common contexts
        # For now, create attention biases for likely scenarios
        
        current_time = time.time()
        
        # Create biases toward recent successful patterns
        recent_insights = [i for i in self.background_insights 
                          if current_time - i.discovered_at < 600.0]  # Last 10 minutes
        
        for insight in recent_insights:
            if insight.insight_type == "pattern_discovery":
                # Create attention bias for discovered patterns
                bias = AttentionBias(
                    context_type=insight.insight_type,
                    node_biases={},  # Could be populated with specific nodes
                    connection_biases={"similarity": 1.2, "temporal": 1.1},
                    temporal_decay=0.99,  # Slow decay
                    created_at=current_time
                )
                self._add_attention_bias(bias)
    
    def _perform_connection_optimization(self):
        """Optimize connections in the graph for better traversal."""
        # Find weak connections that could be strengthened
        all_nodes = self.world_graph.all_nodes()
        
        for node in all_nodes:
            if node.strength > 40.0:  # Only optimize strong nodes
                # Look for similar nodes that aren't well connected
                similar_nodes = self.world_graph.find_similar_nodes(
                    node.mental_context, similarity_threshold=0.6, max_results=5
                )
                
                for similar_node in similar_nodes:
                    if similar_node.node_id != node.node_id:
                        # Check if connection exists and is weak
                        current_weight = node.connection_weights.get(similar_node.node_id, 0.0)
                        if current_weight < 0.5:  # Weak or missing connection
                            # Strengthen the connection
                            new_weight = min(0.8, current_weight + 0.2)
                            node.connection_weights[similar_node.node_id] = new_weight
                            similar_node.connection_weights[node.node_id] = new_weight
    
    def _create_context_signature(self, context: List[float]) -> str:
        """Create a rough signature for a context."""
        # Discretize context values into bins
        bins = []
        for value in context:
            if value < 0.3:
                bins.append('L')  # Low
            elif value < 0.7:
                bins.append('M')  # Medium
            else:
                bins.append('H')  # High
        return ''.join(bins)
    
    def _add_pre_activation(self, node_id: str, activation_level: float, 
                          reason: str, duration: float):
        """Add a pre-activation for a node."""
        expires_at = time.time() + duration
        pre_activation = PreActivation(
            node_id=node_id,
            activation_level=activation_level,
            reason=reason,
            expires_at=expires_at,
            confidence=0.8
        )
        
        self.pre_activations[node_id] = pre_activation
        
        # Limit number of pre-activations
        if len(self.pre_activations) > self.max_pre_activations:
            # Remove oldest pre-activations
            oldest = min(self.pre_activations.values(), key=lambda p: p.expires_at)
            del self.pre_activations[oldest.node_id]
    
    def _add_insight(self, insight: BackgroundInsight):
        """Add a background insight."""
        self.background_insights.append(insight)
        
        # Limit number of insights
        if len(self.background_insights) > self.max_insights:
            self.background_insights.pop(0)
    
    def _add_attention_bias(self, bias: AttentionBias):
        """Add an attention bias."""
        self.attention_biases.append(bias)
        
        # Limit number of biases
        if len(self.attention_biases) > self.max_attention_biases:
            self.attention_biases.pop(0)
    
    def _cleanup_expired_state(self):
        """Clean up expired pre-activations and biases."""
        current_time = time.time()
        
        # Remove expired pre-activations
        expired_pre_activations = [
            node_id for node_id, pre_act in self.pre_activations.items()
            if current_time > pre_act.expires_at
        ]
        for node_id in expired_pre_activations:
            del self.pre_activations[node_id]
        
        # Remove old insights
        self.background_insights = [
            insight for insight in self.background_insights
            if current_time - insight.discovered_at < 3600.0  # Keep for 1 hour
        ]
    
    def get_pre_activated_nodes(self, context: List[float]) -> List[Tuple[str, float]]:
        """Get pre-activated nodes relevant to the current context."""
        context_sig = self._create_context_signature(context)
        current_time = time.time()
        
        relevant_pre_activations = []
        
        # Check for direct pre-activations
        for node_id, pre_act in self.pre_activations.items():
            if current_time <= pre_act.expires_at:
                # Check if this pre-activation is relevant to current context
                if (pre_act.reason == "pattern_cluster_" + context_sig or
                    "all" in pre_act.reason):
                    relevant_pre_activations.append((node_id, pre_act.activation_level))
        
        return relevant_pre_activations
    
    def get_context_map(self, context: List[float]) -> Optional[ContextMap]:
        """Get a context map for the given context."""
        context_sig = self._create_context_signature(context)
        context_map = self.context_maps.get(context_sig)
        
        if context_map:
            context_map.usage_count += 1
            self.context_maps_used += 1
        
        return context_map
    
    def get_attention_biases(self, context: List[float]) -> List[AttentionBias]:
        """Get attention biases relevant to the current context."""
        # For now, return all recent biases
        # Could be filtered by context in the future
        current_time = time.time()
        
        relevant_biases = []
        for bias in self.attention_biases:
            # Apply temporal decay
            age = current_time - bias.created_at
            if age < 600.0:  # Within 10 minutes
                relevant_biases.append(bias)
        
        return relevant_biases
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get background processing statistics."""
        return {
            "is_running": self.is_running,
            "tasks_processed": self.tasks_processed,
            "insights_discovered": self.insights_discovered,
            "pre_activations_active": len(self.pre_activations),
            "context_maps_cached": len(self.context_maps),
            "attention_biases_active": len(self.attention_biases),
            "pre_activations_used": self.pre_activations_used,
            "context_maps_used": self.context_maps_used,
            "background_insights": len(self.background_insights)
        }


class DualModeProcessor:
    """
    Dual-mode cognitive processing system that combines immediate and background thinking.
    
    Immediate Mode: Fast, energy-based traversal for real-time decisions
    Background Mode: Continuous exploration and pattern discovery
    """
    
    def __init__(self, world_graph: WorldGraph):
        self.world_graph = world_graph
        
        # Initialize immediate mode (auto-tuned energy traversal)
        self.immediate_mode = AutoTunedEnergyTraversal(
            initial_energy=100.0,
            energy_drain_rate=12.0,  # Slightly conservative
            interest_threshold=0.5,   # Lower threshold for background-assisted search
            enable_auto_tuning=True,
            tuning_frequency=10
        )
        
        # Initialize background mode
        self.background_mode = BackgroundProcessor(world_graph)
        
        # Dual-mode statistics
        self.immediate_predictions = 0
        self.background_assisted_predictions = 0
        self.total_pre_activations_used = 0
        self.total_context_maps_used = 0
    
    def start_background_processing(self):
        """Start background processing."""
        self.background_mode.start_background_processing()
    
    def stop_background_processing(self):
        """Stop background processing."""
        self.background_mode.stop_background_processing()
    
    def predict(self, start_context: List[float], 
               context_info: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Perform dual-mode prediction combining immediate and background thinking.
        
        Returns:
            Comprehensive prediction result with both immediate and background info
        """
        self.immediate_predictions += 1
        
        # 1. Get background assistance
        background_assistance = self._get_background_assistance(start_context)
        
        # 2. Apply background assistance to immediate mode
        self._apply_background_assistance(background_assistance)
        
        # 3. Perform immediate mode prediction
        immediate_result = self.immediate_mode.traverse(
            start_context, self.world_graph, context=context_info
        )
        
        # 4. Check if background assistance was used
        background_used = any([
            background_assistance['pre_activated_nodes'],
            background_assistance['context_map'],
            background_assistance['attention_biases']
        ])
        
        if background_used:
            self.background_assisted_predictions += 1
        
        # 5. Update background processing based on immediate result
        self._feedback_to_background(immediate_result, start_context)
        
        return {
            'immediate_result': immediate_result,
            'background_assistance': background_assistance,
            'background_used': background_used,
            'dual_mode_stats': self.get_dual_mode_stats()
        }
    
    def _get_background_assistance(self, context: List[float]) -> Dict[str, Any]:
        """Get assistance from background processing."""
        # Get pre-activated nodes
        pre_activated_nodes = self.background_mode.get_pre_activated_nodes(context)
        
        # Get context map
        context_map = self.background_mode.get_context_map(context)
        
        # Get attention biases
        attention_biases = self.background_mode.get_attention_biases(context)
        
        # Get recent insights
        recent_insights = [
            insight for insight in self.background_mode.background_insights
            if time.time() - insight.discovered_at < 300.0  # Last 5 minutes
        ]
        
        return {
            'pre_activated_nodes': pre_activated_nodes,
            'context_map': context_map,
            'attention_biases': attention_biases,
            'recent_insights': recent_insights
        }
    
    def _apply_background_assistance(self, assistance: Dict[str, Any]):
        """Apply background assistance to immediate mode processing."""
        # Track usage
        if assistance['pre_activated_nodes']:
            self.total_pre_activations_used += len(assistance['pre_activated_nodes'])
        
        if assistance['context_map']:
            self.total_context_maps_used += 1
        
        # The assistance could modify the energy traversal parameters
        # or provide hints about good starting nodes
        # For now, this is conceptual - the actual integration would require
        # modifying the energy traversal to use background hints
    
    def _feedback_to_background(self, immediate_result: AutoTunedTraversalResult, 
                              context: List[float]):
        """Provide feedback to background processing based on immediate results."""
        # If the immediate result was very good, reinforce the patterns used
        if immediate_result.goldilocks_score > 0.8:
            # Schedule pattern reinforcement
            self.background_mode._schedule_task(
                BackgroundTaskType.PATTERN_DISCOVERY,
                {'successful_context': context, 'score': immediate_result.goldilocks_score}
            )
        
        # If performance was poor, trigger exploration
        elif immediate_result.goldilocks_score < 0.4:
            self.background_mode._schedule_task(
                BackgroundTaskType.CONNECTION_OPTIMIZATION,
                {'problematic_context': context}
            )
    
    def get_dual_mode_stats(self) -> Dict[str, Any]:
        """Get comprehensive dual-mode processing statistics."""
        background_stats = self.background_mode.get_processing_stats()
        immediate_stats = self.immediate_mode.get_tuning_status()
        
        return {
            'immediate_mode': {
                'total_predictions': self.immediate_predictions,
                'background_assisted_predictions': self.background_assisted_predictions,
                'background_assistance_rate': (
                    self.background_assisted_predictions / max(1, self.immediate_predictions)
                ),
                'tuning_status': immediate_stats
            },
            'background_mode': background_stats,
            'dual_mode_efficiency': {
                'pre_activations_used': self.total_pre_activations_used,
                'context_maps_used': self.total_context_maps_used
            }
        }
    
    def get_system_overview(self) -> Dict[str, Any]:
        """Get high-level overview of the dual-mode system."""
        stats = self.get_dual_mode_stats()
        
        # Calculate efficiency metrics
        assistance_rate = stats['immediate_mode']['background_assistance_rate']
        background_productivity = stats['background_mode']['insights_discovered']
        
        return {
            'system_status': {
                'immediate_mode_active': True,
                'background_mode_active': stats['background_mode']['is_running'],
                'integration_level': assistance_rate
            },
            'performance_summary': {
                'total_predictions': stats['immediate_mode']['total_predictions'],
                'background_assistance_rate': f"{assistance_rate:.1%}",
                'insights_discovered': background_productivity,
                'system_efficiency': min(1.0, assistance_rate + background_productivity / 100.0)
            },
            'cognitive_architecture': {
                'dual_mode_active': True,
                'meta_cognition_enabled': stats['immediate_mode']['tuning_status']['auto_tuning_enabled'],
                'background_insights_active': len(self.background_mode.background_insights) > 0,
                'pre_activation_system_active': len(self.background_mode.pre_activations) > 0
            }
        }