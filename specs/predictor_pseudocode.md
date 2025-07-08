# Predictor Process Pseudocode

## Main Predictor Function
```python
def generate_prediction(current_mental_context, world_graph, max_traversal_depth, adaptive_systems=None):
    """
    Runs multiple parallel thought traversals and returns consensus prediction
    Uses adaptive systems to optimize the prediction process
    """
    # Determine number of traversals to run (adaptive or default to 3)
    if adaptive_systems and 'consensus' in adaptive_systems:
        num_traversals = adaptive_systems['consensus'].traversal_count
    else:
        num_traversals = 3
    
    predictions = []
    traversal_info = []
    
    # Run parallel traversals with different random seeds
    for i in range(num_traversals):
        random_seed = generate_random_seed()
        
        prediction, path_info = single_traversal_with_info(
            current_mental_context, 
            world_graph, 
            max_traversal_depth, 
            random_seed,
            adaptive_systems
        )
        
        predictions.append(prediction)
        traversal_info.append(path_info)
        
    # Find consensus or handle ties
    final_prediction, consensus_strength = resolve_consensus_with_info(predictions, traversal_info)
    
    # Store information for adaptive learning
    final_prediction['traversal_paths'] = [info['path'] for info in traversal_info]
    final_prediction['similarity_components_used'] = traversal_info[0].get('similarity_components', {})
    
    return final_prediction, consensus_strength
```

## Single Traversal Process
```python
def single_traversal_with_info(start_context, world_graph, max_depth, random_seed, adaptive_systems):
    """
    Traverse the experience graph to find a prediction, with detailed tracking for adaptation
    """
    # Find starting node using adaptive similarity if available
    if adaptive_systems and 'similarity' in adaptive_systems:
        current_node, similarity_info = find_most_similar_node_adaptive(
            start_context, world_graph, adaptive_systems['similarity']
        )
    else:
        current_node = find_most_similar_node(start_context, world_graph)
        similarity_info = {}
    
    traversal_path = [current_node]
    
    # Bounded depth traversal
    for depth in range(max_depth):
        # Get neighboring nodes (connected via similarity, temporal, causal links)
        neighbors = get_all_neighbors(current_node, world_graph)
        
        if not neighbors:
            break  # Dead end
            
        # Select next node using strength + randomness
        next_node = weighted_random_selection(neighbors, random_seed)
        traversal_path.append(next_node)
        current_node = next_node
    
    # Strengthen all nodes in the traversal path
    strengthen_path(traversal_path)
    
    # Find terminal node that best matches recent experience pattern
    terminal_node = find_best_terminal_match(traversal_path, start_context)
    
    # Generate prediction from terminal node
    prediction = extract_prediction(terminal_node)
    
    # Package information for adaptive learning
    path_info = {
        'path': [node.node_id for node in traversal_path],
        'similarity_components': similarity_info,
        'depth_reached': len(traversal_path),
        'terminal_node_strength': terminal_node.strength
    }
    
    return prediction, path_info
```

## Supporting Functions
```python
def find_most_similar_node(target_context, world_graph):
    """
    Find node with most similar mental context using neighborhood similarity
    """
    best_node = None
    best_similarity = -1
    
    for node in world_graph.all_nodes():
        # Direct context similarity
        direct_sim = euclidean_similarity(target_context, node.mental_context)
        
        # Neighborhood similarity (compare surrounding nodes)
        neighbor_sim = neighborhood_similarity(target_context, node, world_graph)
        
        # Combined similarity score
        total_similarity = direct_sim + neighbor_sim
        
        if total_similarity > best_similarity:
            best_similarity = total_similarity
            best_node = node
            
    return best_node

def weighted_random_selection(neighbors, random_seed):
    """
    Select next node based on strength with randomness
    """
    # Create probability distribution based on node strengths
    weights = [node.strength for node in neighbors]
    
    # Add randomness factor
    random_factors = [random(random_seed) * randomness_coefficient for _ in neighbors]
    adjusted_weights = [w + r for w, r in zip(weights, random_factors)]
    
    # Select using weighted random choice
    selected_node = weighted_random_choice(neighbors, adjusted_weights)
    return selected_node

def strengthen_path(traversal_path):
    """
    Increase strength of all nodes in the traversal path
    """
    for node in traversal_path:
        node.strength += 1.0
        
        # Also strengthen neighboring nodes (spillover effect)
        for neighbor in node.similar_contexts:
            neighbor.strength += 0.1

def find_most_similar_node_adaptive(target_context, world_graph, adaptive_similarity):
    """
    Find node with most similar mental context using learned similarity weights
    """
    best_node = None
    best_similarity = -1
    similarity_components = {}
    
    for node in world_graph.all_nodes():
        # Calculate similarity using adaptive weights
        total_similarity, components = adaptive_similarity.calculate_similarity(
            {'mental_context': target_context}, 
            node, 
            world_graph
        )
        
        if total_similarity > best_similarity:
            best_similarity = total_similarity
            best_node = node
            similarity_components = components
            
    return best_node, similarity_components

def resolve_consensus_with_info(predictions, traversal_info):
    """
    Handle consensus between multiple predictions with detailed tracking
    """
    # Check for exact matches
    if len(predictions) >= 3:
        if predictions[0] == predictions[1] == predictions[2]:
            return predictions[0], 'perfect_consensus'  # All agree
        elif predictions[0] == predictions[1]:
            return predictions[0], 'strong_consensus'  # 2 out of 3
        elif predictions[0] == predictions[2]:
            return predictions[0], 'strong_consensus'
        elif predictions[1] == predictions[2]:
            return predictions[1], 'strong_consensus'
        else:
            # All different - use the one from strongest traversal path
            strongest_idx = max(range(len(traversal_info)), 
                              key=lambda i: traversal_info[i]['terminal_node_strength'])
            return predictions[strongest_idx], 'weak_consensus'
    
    elif len(predictions) == 2:
        if predictions[0] == predictions[1]:
            return predictions[0], 'strong_consensus'
        else:
            # Use stronger path
            stronger_idx = 0 if traversal_info[0]['terminal_node_strength'] > traversal_info[1]['terminal_node_strength'] else 1
            return predictions[stronger_idx], 'weak_consensus'
    
    else:
        # Single prediction
        return predictions[0], 'single_traversal'

def extract_prediction(terminal_node):
    """
    Generate prediction from terminal node
    """
    return {
        'expected_sensory': terminal_node.actual_sensory,  # What happened before
        'motor_action': terminal_node.action_taken,         # Action to take
        'confidence': terminal_node.strength               # How sure we are
    }
```

## Graph Structure Access
```python
def get_all_neighbors(node, world_graph):
    """
    Get all connected nodes (temporal, causal, similarity)
    """
    neighbors = []
    
    # Add temporally connected nodes
    if node.temporal_predecessor:
        neighbors.append(world_graph.get_node(node.temporal_predecessor))
    
    # Add prediction source nodes
    for source_id in node.prediction_sources:
        neighbors.append(world_graph.get_node(source_id))
    
    # Add similar context nodes
    for similar_id in node.similar_contexts:
        neighbors.append(world_graph.get_node(similar_id))
    
    return neighbors

def neighborhood_similarity(target_context, candidate_node, world_graph):
    """
    Compare neighborhoods of nodes for better similarity matching
    """
    target_neighbors = get_context_neighbors(target_context, world_graph)
    candidate_neighbors = get_all_neighbors(candidate_node, world_graph)
    
    similarity_score = 0
    for target_neighbor in target_neighbors:
        for candidate_neighbor in candidate_neighbors:
            neighbor_sim = euclidean_similarity(target_neighbor.mental_context, 
                                              candidate_neighbor.mental_context)
            similarity_score += neighbor_sim
    
    return similarity_score / (len(target_neighbors) * len(candidate_neighbors))
```