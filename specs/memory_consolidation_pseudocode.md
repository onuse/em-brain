# Memory Consolidation Process Pseudocode

## Main Consolidation Loop
```python
def memory_consolidation_process(world_graph, consolidation_config):
    """
    Background process that continuously optimizes the experience graph
    Runs every second to avoid hitching the main mental loop
    
    Abstractions emerge naturally through the merging process - no explicit creation needed
    """
    while robot_is_active():
        # === STRENGTH DECAY ===
        apply_global_strength_decay(world_graph, decay_rate=0.001)
        
        # === NODE MERGING ===
        weak_nodes = find_nodes_below_threshold(world_graph, 
                                               threshold=consolidation_config.merge_threshold)
        
        for weak_node in weak_nodes:
            merge_with_similar_node(weak_node, world_graph)
        
        # === GRAPH CLEANUP ===
        remove_orphaned_nodes(world_graph)
        optimize_similarity_links(world_graph)
        
        # Wait before next consolidation cycle
        sleep(1.0)  # Run every second
```

## Strength Management
```python
def apply_global_strength_decay(world_graph, decay_rate):
    """
    Reduce strength of all nodes by decay_rate
    """
    for node in world_graph.all_nodes():
        node.strength = max(0.0, node.strength - decay_rate)
        
        # Track decay for debugging
        if node.strength == 0.0:
            mark_for_potential_removal(node)

def find_nodes_below_threshold(world_graph, threshold):
    """
    Find nodes weak enough to be merged
    """
    weak_nodes = []
    
    for node in world_graph.all_nodes():
        if node.strength < threshold:
            weak_nodes.append(node)
    
    # Sort by strength (weakest first)
    weak_nodes.sort(key=lambda n: n.strength)
    
    return weak_nodes
```

## Node Merging System
```python
def merge_with_similar_node(weak_node, world_graph):
    """
    Find the most similar node and merge the weak node into it
    """
    # Find most similar node that's stronger
    candidate_nodes = world_graph.all_nodes()
    candidate_nodes = [n for n in candidate_nodes if n.strength > weak_node.strength]
    
    if not candidate_nodes:
        return  # No stronger nodes to merge with
    
    # Calculate similarity to all candidates
    best_similarity = -1
    best_candidate = None
    
    for candidate in candidate_nodes:
        similarity = calculate_total_similarity(weak_node, candidate, world_graph)
        
        if similarity > best_similarity:
            best_similarity = similarity
            best_candidate = candidate
    
    # Only merge if similarity is high enough
    if best_similarity > 0.7:  # Similarity threshold
        perform_node_merge(weak_node, best_candidate, world_graph)

def perform_node_merge(source_node, target_node, world_graph):
    """
    Merge source_node into target_node and update all connections
    
    This process naturally creates abstractions - merged nodes become more general
    and can match broader patterns in future predictions
    """
    # === DATA MERGING ===
    
    # Weighted average of contexts (by strength)
    total_strength = source_node.strength + target_node.strength
    source_weight = source_node.strength / total_strength
    target_weight = target_node.strength / total_strength
    
    # Merge mental contexts - this creates natural generalization
    merged_context = []
    for i in range(len(target_node.mental_context)):
        merged_value = (source_node.mental_context[i] * source_weight + 
                       target_node.mental_context[i] * target_weight)
        merged_context.append(merged_value)
    
    target_node.mental_context = merged_context
    
    # Merge sensory data (average) - creates broader pattern matching
    target_node.predicted_sensory = average_vectors(
        source_node.predicted_sensory, 
        target_node.predicted_sensory
    )
    target_node.actual_sensory = average_vectors(
        source_node.actual_sensory, 
        target_node.actual_sensory
    )
    
    # Average prediction errors - merged nodes represent broader uncertainty
    target_node.prediction_error = (source_node.prediction_error + target_node.prediction_error) / 2.0
    
    # Combine strengths - merged nodes become stronger and more influential
    target_node.strength += source_node.strength
    
    # === CONNECTION MERGING ===
    
    # Merge all connection lists - inherit relationships from both nodes
    target_node.similar_contexts.extend(source_node.similar_contexts)
    target_node.prediction_sources.extend(source_node.prediction_sources)
    
    # Remove duplicates
    target_node.similar_contexts = list(set(target_node.similar_contexts))
    target_node.prediction_sources = list(set(target_node.prediction_sources))
    
    # === UPDATE ALL REFERENCES ===
    
    # Find all nodes that reference the source node
    referencing_nodes = find_nodes_referencing(source_node.id, world_graph)
    
    for ref_node in referencing_nodes:
        # Replace source_node references with target_node
        replace_node_references(ref_node, source_node.id, target_node.id)
    
    # === CLEANUP ===
    
    # Remove the source node from graph
    world_graph.remove_node(source_node.id)
    
    # Log the merge for debugging
    log_node_merge(source_node.id, target_node.id, "Natural abstraction formed through merging")
```

## Graph Optimization
```python
def remove_orphaned_nodes(world_graph):
    """
    Remove nodes that have no connections (shouldn't happen normally)
    """
    orphaned_nodes = []
    
    for node in world_graph.all_nodes():
        if is_orphaned(node):
            orphaned_nodes.append(node)
    
    for orphan in orphaned_nodes:
        world_graph.remove_node(orphan.id)
        log_orphan_removal(orphan.id)

def optimize_similarity_links(world_graph):
    """
    Clean up and optimize similarity connections
    """
    for node in world_graph.all_nodes():
        # Remove similarity links to nodes that no longer exist
        valid_links = []
        for link_id in node.similar_contexts:
            if world_graph.node_exists(link_id):
                valid_links.append(link_id)
        
        node.similar_contexts = valid_links
        
        # Limit number of similarity connections to prevent explosion
        max_similarity_links = 20
        if len(node.similar_contexts) > max_similarity_links:
            # Keep only the strongest connections
            linked_nodes = [world_graph.get_node(id) for id in node.similar_contexts]
            linked_nodes.sort(key=lambda n: n.strength, reverse=True)
            node.similar_contexts = [n.id for n in linked_nodes[:max_similarity_links]]

def calculate_total_similarity(node1, node2, world_graph):
    """
    Enhanced similarity calculation for merging decisions
    """
    # Direct context similarity
    context_sim = euclidean_similarity(node1.mental_context, node2.mental_context)
    
    # Action similarity
    action_sim = euclidean_similarity(node1.action_taken, node2.action_taken)
    
    # Sensory similarity
    sensory_sim = euclidean_similarity(node1.actual_sensory, node2.actual_sensory)
    
    # Neighborhood similarity
    neighbor_sim = neighborhood_similarity(node1, node2, world_graph)
    
    # Weighted combination
    total_similarity = (context_sim * 0.4 + 
                       action_sim * 0.2 + 
                       sensory_sim * 0.2 + 
                       neighbor_sim * 0.2)
    
    return total_similarity
```

## Helper Functions
```python
def average_vectors(vec1, vec2):
    """
    Calculate element-wise average of two vectors
    This creates natural generalization when merging experiences
    """
    if len(vec1) != len(vec2):
        return vec1  # Fallback to first vector
    
    return [(v1 + v2) / 2.0 for v1, v2 in zip(vec1, vec2)]

def is_orphaned(node):
    """
    Check if a node has no meaningful connections
    """
    has_temporal = node.temporal_predecessor is not None
    has_predictions = len(node.prediction_sources) > 0
    has_similar = len(node.similar_contexts) > 0
    
    return not (has_temporal or has_predictions or has_similar)

def find_nodes_referencing(target_id, world_graph):
    """
    Find all nodes that reference the target node ID
    """
    referencing = []
    
    for node in world_graph.all_nodes():
        if (target_id in node.similar_contexts or 
            target_id in node.prediction_sources or
            node.temporal_predecessor == target_id):
            referencing.append(node)
    
    return referencing

def replace_node_references(node, old_id, new_id):
    """
    Replace all references to old_id with new_id in a node's connections
    """
    # Replace in similar contexts
    node.similar_contexts = [new_id if id == old_id else id 
                            for id in node.similar_contexts]
    
    # Replace in prediction sources
    node.prediction_sources = [new_id if id == old_id else id 
                              for id in node.prediction_sources]
    
    # Replace temporal predecessor
    if node.temporal_predecessor == old_id:
        node.temporal_predecessor = new_id
```
```