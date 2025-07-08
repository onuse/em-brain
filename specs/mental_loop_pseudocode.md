# Main Mental Loop Pseudocode

## Core Mental Loop
```python
def main_mental_loop(world_graph, brainstem_connection, genome, adaptive_systems):
    """
    The continuous cycle of prediction → action → learning with adaptive parameters
    """
    current_mental_context = initialize_mental_context()
    pending_prediction = None
    recent_prediction_accuracies = []
    
    while robot_is_active():
        # === SENSORY INPUT PHASE ===
        actual_sensory = receive_sensory_data(brainstem_connection)
        
        # === EXPERIENCE NODE CREATION ===
        if pending_prediction is not None:
            # We have a prediction to validate against reality
            new_experience = create_experience_node(
                mental_context=current_mental_context,
                prediction=pending_prediction,
                actual_sensory=actual_sensory
            )
            
            # Add to graph and establish connections
            add_experience_to_graph(new_experience, world_graph)
            
            # Learn from prediction outcome
            prediction_accuracy = 1.0 - new_experience.prediction_error
            recent_prediction_accuracies.append(prediction_accuracy)
            
            # Update adaptive systems based on prediction success
            update_adaptive_systems(adaptive_systems, prediction_accuracy, 
                                   pending_prediction, new_experience)
            
            # Keep only recent history for adaptive learning
            if len(recent_prediction_accuracies) > 100:
                recent_prediction_accuracies = recent_prediction_accuracies[-50:]
            
        # === MENTAL CONTEXT UPDATE ===
        current_mental_context = update_mental_context(
            previous_context=current_mental_context,
            new_experience=new_experience if pending_prediction else None,
            recent_sensory=actual_sensory
        )
        
        # === PREDICTION GENERATION ===
        if world_graph.has_nodes():
            # Use adaptive predictor when we have experience to draw from
            prediction_result = generate_prediction_with_adaptation(
                current_mental_context, 
                world_graph, 
                adaptive_systems,
                recent_prediction_accuracies
            )
        else:
            # Bootstrap: random action when graph is empty
            prediction_result = generate_random_action(brainstem_connection)
        
        # === ACTION EXECUTION ===
        motor_commands = prediction_result['motor_action']
        send_motor_commands(brainstem_connection, motor_commands)
        
        # Store prediction for next cycle validation
        pending_prediction = {
            'expected_sensory': prediction_result['expected_sensory'],
            'motor_action': motor_commands,
            'confidence': prediction_result.get('confidence', 0.0),
            'consensus_strength': prediction_result.get('consensus_strength', 'unknown'),
            'thinking_depth': prediction_result.get('thinking_depth', 0),
            'timestamp': current_time()
        }
        
        # === TIMING ===
        wait_for_next_cycle()  # Control loop frequency
```

## Supporting Functions

### Experience Node Creation
```python
def create_experience_node(mental_context, prediction, actual_sensory):
    """
    Create a new experience node from prediction vs reality
    """
    prediction_error = calculate_prediction_error(
        prediction['expected_sensory'], 
        actual_sensory
    )
    
    experience = ExperienceNode(
        mental_context=mental_context.copy(),
        action_taken=prediction['motor_action'],
        predicted_sensory=prediction['expected_sensory'],
        actual_sensory=actual_sensory,
        prediction_error=prediction_error,
        strength=1.0,  # Initial strength
        timestamp=current_time(),
        
        # Will be populated during graph insertion
        temporal_predecessor=None,
        prediction_sources=[],
        similar_contexts=[]
    )
    
    return experience
```

### Graph Integration
```python
def add_experience_to_graph(new_experience, world_graph):
    """
    Add experience node and establish all graph connections
    """
    # Add the node
    node_id = world_graph.add_node(new_experience)
    
    # === TEMPORAL LINKING ===
    latest_node = world_graph.get_latest_node()
    if latest_node:
        new_experience.temporal_predecessor = latest_node.id
        # Also add forward temporal link
        latest_node.temporal_successor = node_id
    
    # === PREDICTION SOURCE LINKING ===
    # Find nodes that contributed to this prediction
    # (This would come from the predictor's traversal path)
    prediction_sources = get_last_prediction_sources()
    new_experience.prediction_sources = prediction_sources
    
    # === SIMILARITY LINKING ===
    # Find nodes with similar mental contexts
    similar_nodes = find_similar_context_nodes(
        new_experience.mental_context, 
        world_graph,
        similarity_threshold=0.8
    )
    
    new_experience.similar_contexts = [node.id for node in similar_nodes]
    
    # Add reverse links
    for similar_node in similar_nodes:
        similar_node.similar_contexts.append(node_id)
```

### Mental Context Management
```python
def update_mental_context(previous_context, new_experience, recent_sensory):
    """
    Update the brain's current mental state
    """
    # Mental context represents "what the brain is thinking about"
    # This could be a rolling window of recent experiences + current sensory state
    
    context_vector = []
    
    # Add current sensory state (compressed/processed)
    sensory_features = extract_sensory_features(recent_sensory)
    context_vector.extend(sensory_features)
    
    # Add recent experience pattern
    if new_experience:
        experience_features = extract_experience_features(new_experience)
        context_vector.extend(experience_features)
    
    # Add previous context (with decay)
    if previous_context:
        decayed_previous = apply_temporal_decay(previous_context, decay_rate=0.1)
        context_vector.extend(decayed_previous)
    
    # Normalize to fixed size
    normalized_context = normalize_context_vector(context_vector, target_size=256)
    
    return normalized_context

def calculate_thinking_depth(mental_context, genome):
    """
    Determine how deep to search based on current state and drives
    """
    base_depth = 5  # Default thinking depth
    
    # Increase depth when uncertain (high prediction errors recently)
    uncertainty_factor = get_recent_uncertainty(mental_context)
    
    # Genome influence on thinking style
    deliberation_drive = genome.get_drive('deliberation', default=1.0)
    
    # Calculate dynamic depth
    thinking_depth = int(base_depth * deliberation_drive * (1 + uncertainty_factor))
    
    # Clamp to reasonable bounds
    return max(2, min(thinking_depth, 20))
```

### Bootstrap Handling
```python
def generate_random_action():
    """
    Generate random motor commands when no experience exists
    """
    # Get available actuators from brainstem
    available_actuators = get_actuator_list()
    
    random_action = {}
    for actuator in available_actuators:
        # Random value within safe operating range
        random_value = random_float(actuator.min_value, actuator.max_value)
        random_action[actuator.id] = random_value
    
    return {
        'motor_action': random_action,
        'expected_sensory': None,  # No prediction possible
        'confidence': 0.0
    }

def initialize_mental_context():
    """
    Create initial mental context for startup
    """
    # Start with basic "I exist" context
    initial_context = [0.0] * 256  # Fixed size vector
    
    # Maybe add some basic genome-driven biases
    # initial_context[0] = 1.0  # "awake" signal
    
    return initial_context
```

### Communication Layer
```python
def receive_sensory_data(brainstem_connection):
    """
    Get current sensor readings from brainstem
    """
    sensory_packet = brainstem_connection.receive()
    
    return {
        'data_points': sensory_packet.sensor_values,  # [val1, val2, val3, ...]
        'actuator_states': sensory_packet.actuator_positions,
        'timestamp': sensory_packet.timestamp,
        'sequence_id': sensory_packet.sequence_id
    }

def send_motor_commands(brainstem_connection, motor_commands):
    """
    Send motor commands to brainstem
    """
    command_packet = {
        'actuator_commands': motor_commands,
        'timestamp': current_time(),
        'sequence_id': generate_sequence_id()
    }
    
    brainstem_connection.send(command_packet)
```