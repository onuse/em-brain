# Bootstrap and Startup Sequence

## Overview
This document describes the complete startup process from power-on to autonomous learning. The system transitions from complete ignorance to pattern-based behavior through pure experimentation.

## Phase 1: Hardware Initialization

### Brainstem Startup (Raspberry Pi)
```python
def brainstem_startup():
    """
    Initialize brainstem hardware and prepare for brain connection
    """
    print("=== BRAINSTEM STARTUP ===")
    
    # 1. System initialization
    initialize_gpio_pins()
    setup_signal_handlers()  # Graceful shutdown on SIGTERM
    
    # 2. Hardware discovery
    print("Discovering hardware...")
    sensors = discover_and_initialize_sensors()
    actuators = discover_and_initialize_actuators()
    
    print(f"Found {len(sensors)} sensors: {[s['id'] for s in sensors]}")
    print(f"Found {len(actuators)} actuators: {[a['id'] for a in actuators]}")
    
    # 3. Safety check - move all actuators to safe positions
    print("Moving actuators to safe positions...")
    safe_commands = get_safe_default_commands(actuators)
    execute_motor_commands(actuators, safe_commands)
    time.sleep(1.0)  # Wait for movement to complete
    
    # 4. Network setup
    print("Setting up network connection...")
    brain_connection = wait_for_brain_connection()
    
    # 5. Send hardware capabilities to brain
    print("Sending hardware manifest to brain...")
    send_hardware_capabilities(brain_connection, sensors, actuators)
    
    # 6. Wait for brain ready signal
    print("Waiting for brain ready signal...")
    wait_for_brain_ready(brain_connection)
    
    print("=== BRAINSTEM READY ===")
    return brain_connection, sensors, actuators
```

### Brain Startup (Laptop)
```python
def brain_startup():
    """
    Initialize brain software and establish connection to brainstem
    """
    print("=== BRAIN STARTUP ===")
    
    # 1. Initialize core systems
    world_graph = WorldGraph()
    mental_context = MentalContext(context_vector=[0.0] * 256)  # Blank mind
    genome = GenomeData()  # Default drives and parameters
    
    # 2. Initialize adaptive systems
    adaptive_systems = {
        'similarity': AdaptiveSimilarity(),
        'decay': AdaptiveDecay(), 
        'merging': AdaptiveMerging(),
        'thinking': AdaptiveThinking(),
        'consensus': AdaptiveConsensus()
    }
    
    # 3. Network connection to brainstem (or simulation)
    print("Connecting to brainstem...")
    brainstem_connection = connect_to_brainstem_or_simulation()
    
    # 4. Receive hardware capabilities
    print("Receiving hardware manifest...")
    hardware_manifest = receive_hardware_capabilities(brainstem_connection)
    
    # 5. Initialize sensor/actuator mappings
    sensor_mapping = create_sensor_mapping(hardware_manifest['sensors'])
    actuator_mapping = create_actuator_mapping(hardware_manifest['actuators'])
    
    print(f"Brain will process {len(sensor_mapping)} sensor inputs")
    print(f"Brain will control {len(actuator_mapping)} actuators")
    
    # 6. Start background processes
    print("Starting memory consolidation process...")
    start_memory_consolidation(world_graph, adaptive_systems['decay'], adaptive_systems['merging'])
    
    # 7. Signal ready to brainstem
    send_brain_ready_signal(brainstem_connection)
    
    print("=== BRAIN READY ===")
    return {
        'world_graph': world_graph,
        'mental_context': mental_context,
        'genome': genome,
        'brainstem_connection': brainstem_connection,
        'sensor_mapping': sensor_mapping,
        'actuator_mapping': actuator_mapping,
        'adaptive_systems': adaptive_systems
    }

def connect_to_brainstem_or_simulation():
    """
    Connect to either real brainstem or simulation based on configuration
    """
    import os
    
    if os.getenv('ROBOT_MODE') == 'simulation':
        print("Using Grid World simulation")
        from brainstem_interface import SimulationBrainstemInterface
        return SimulationBrainstemInterface()
    else:
        print("Connecting to real brainstem over network")
        from brainstem_interface import NetworkBrainstemInterface
        return NetworkBrainstemInterface("192.168.1.100", 5001)
```

## Phase 2: First Contact

### Initial Sensory Flood
```python
def handle_first_sensory_input(brain_state):
    """
    Process the very first sensory data when robot powers on
    """
    print("=== FIRST SENSORY CONTACT ===")
    
    # Wait for first sensory packet from brainstem
    first_sensory = receive_sensory_data(brain_state['brainstem_connection'])
    
    print(f"Received first sensory data: {len(first_sensory['sensor_values'])} values")
    print(f"Sample values: {first_sensory['sensor_values'][:5]}...")
    
    # Update mental context with initial sensory state
    brain_state['mental_context'].context_vector = first_sensory['sensor_values'].copy()
    
    # Since world graph is empty, no prediction is possible
    # Generate first random action
    first_action = generate_first_random_action(brain_state['actuator_mapping'])
    
    print(f"Generating first random action: {first_action}")
    
    # Send first motor commands
    send_motor_commands(brain_state['brainstem_connection'], first_action)
    
    # Store as pending prediction (will become first experience node)
    pending_prediction = PredictionPacket(
        expected_sensory=[0.0] * len(first_sensory['sensor_values']),  # No prediction
        motor_action=first_action,
        confidence=0.0,  # Zero confidence
        timestamp=datetime.now(),
        sequence_id=1,
        traversal_paths=[],  # No graph traversal
        consensus_strength='random',
        thinking_depth=0
    )
    
    return pending_prediction

def generate_first_random_action(actuator_mapping):
    """
    Generate completely random motor commands for first action
    """
    random_action = {}
    
    for actuator_id, actuator_info in actuator_mapping.items():
        # Random value within safe range
        min_val, max_val = actuator_info['range']
        
        # Be conservative on first action - stay near center
        center = (min_val + max_val) / 2.0
        range_size = (max_val - min_val) * 0.3  # Use only 30% of range initially
        
        random_value = center + random.uniform(-range_size/2, range_size/2)
        random_action[actuator_id] = random_value
    
    return random_action
```

### First Experience Node Creation
```python
def create_first_experience_node(brain_state, pending_prediction):
    """
    Create the very first experience node from first action-outcome pair
    """
    print("=== CREATING FIRST EXPERIENCE ===")
    
    # Wait for sensory response to first action
    second_sensory = receive_sensory_data(brain_state['brainstem_connection'])
    
    # Calculate prediction error (should be large since no real prediction)
    prediction_error = calculate_prediction_error(
        pending_prediction.expected_sensory,
        second_sensory['sensor_values']
    )
    
    # Create first experience node
    first_experience = ExperienceNode(
        mental_context=brain_state['mental_context'].context_vector.copy(),
        action_taken=pending_prediction.motor_action,
        predicted_sensory=pending_prediction.expected_sensory,
        actual_sensory=second_sensory['sensor_values'],
        prediction_error=prediction_error,
        strength=1.0,
        timestamp=datetime.now()
    )
    
    # Add to world graph (first node, no connections)
    node_id = brain_state['world_graph'].add_node(first_experience)
    
    print(f"Created first experience node: {node_id}")
    print(f"Prediction error: {prediction_error:.3f}")
    
    # Update mental context with new experience
    brain_state['mental_context'].update_from_experience(first_experience)
    
    return first_experience
```

## Phase 3: Random Exploration

### Early Learning Loop
```python
def run_bootstrap_learning_phase(brain_state, num_iterations=100):
    """
    Run pure random exploration to build initial experience base
    """
    print(f"=== BOOTSTRAP LEARNING ({num_iterations} iterations) ===")
    
    pending_prediction = None
    
    for iteration in range(num_iterations):
        cycle_start = time.time()
        
        # === SENSORY INPUT ===
        current_sensory = receive_sensory_data(brain_state['brainstem_connection'])
        
        # === EXPERIENCE CREATION ===
        if pending_prediction is not None:
            # Create experience from previous prediction
            experience = create_experience_node(
                brain_state['mental_context'].context_vector,
                pending_prediction,
                current_sensory['sensor_values']
            )
            
            brain_state['world_graph'].add_node(experience)
            brain_state['mental_context'].update_from_experience(experience)
        
        # === ACTION GENERATION ===
        if brain_state['world_graph'].node_count() < 10:
            # Pure random for first 10 experiences
            action = generate_random_action(brain_state['actuator_mapping'])
            confidence = 0.0
            consensus_strength = 'random'
            
        else:
            # Start using simple predictions based on accumulated experience
            try:
                prediction_result = generate_prediction(
                    brain_state['mental_context'].context_vector,
                    brain_state['world_graph'],
                    max_traversal_depth=2  # Shallow thinking during bootstrap
                )
                action = prediction_result['motor_action']
                confidence = prediction_result['confidence']
                consensus_strength = prediction_result.get('consensus_strength', 'weak')
                
            except Exception as e:
                # Fall back to random if prediction fails
                print(f"Prediction failed (iteration {iteration}): {e}")
                action = generate_random_action(brain_state['actuator_mapping'])
                confidence = 0.0
                consensus_strength = 'random'
        
        # === SEND COMMANDS ===
        send_motor_commands(brain_state['brainstem_connection'], action)
        
        # === PREPARE NEXT PREDICTION ===
        pending_prediction = PredictionPacket(
            expected_sensory=current_sensory['sensor_values'].copy(),  # Simple expectation: same as now
            motor_action=action,
            confidence=confidence,
            timestamp=datetime.now(),
            sequence_id=iteration + 2,
            consensus_strength=consensus_strength,
            thinking_depth=2 if brain_state['world_graph'].node_count() >= 10 else 0
        )
        
        # === PROGRESS REPORTING ===
        if iteration % 10 == 0:
            stats = brain_state['world_graph'].get_graph_statistics()
            print(f"Iteration {iteration}: {stats['total_nodes']} experiences, "
                  f"avg strength: {stats['avg_strength']:.2f}, confidence: {confidence:.2f}")
        
        # === TIMING ===
        cycle_time = time.time() - cycle_start
        if cycle_time < 0.1:  # Maintain ~10Hz during bootstrap
            time.sleep(0.1 - cycle_time)
    
    print("=== BOOTSTRAP LEARNING COMPLETE ===")
    final_stats = brain_state['world_graph'].get_graph_statistics()
    print(f"Final stats: {final_stats}")

def generate_random_action(actuator_mapping):
    """
    Generate random motor commands within safe ranges
    """
    action = {}
    
    for actuator_id, actuator_info in actuator_mapping.items():
        min_val, max_val = actuator_info['range']
        action[actuator_id] = random.uniform(min_val, max_val)
    
    return action
```

## Phase 4: Transition to Autonomous Learning

### Pattern Recognition Emergence
```python
def transition_to_autonomous_learning(brain_state):
    """
    Switch from bootstrap mode to full autonomous learning
    """
    print("=== TRANSITIONING TO AUTONOMOUS LEARNING ===")
    
    # Analyze accumulated experiences for basic patterns
    analyze_initial_patterns(brain_state['world_graph'])
    
    # Switch to full mental loop with adaptive parameters
    enable_adaptive_systems(brain_state['adaptive_systems'])
    
    # Increase thinking depth now that we have experience
    brain_state['thinking_depth'] = 5  # Default autonomous depth
    
    # Enable memory consolidation at full rate
    enable_full_memory_consolidation(brain_state['world_graph'])
    
    print("Robot is now running in full autonomous mode")
    print("Learning will continue through experience...")

def analyze_initial_patterns(world_graph):
    """
    Look for basic patterns in the initial random experiences
    """
    print("Analyzing initial patterns...")
    
    all_nodes = world_graph.all_nodes()
    if len(all_nodes) < 10:
        print("Not enough experiences for pattern analysis")
        return
    
    # Find most common actuator patterns
    actuator_patterns = {}
    for node in all_nodes:
        for actuator_id, value in node.action_taken.items():
            if actuator_id not in actuator_patterns:
                actuator_patterns[actuator_id] = []
            actuator_patterns[actuator_id].append(value)
    
    for actuator_id, values in actuator_patterns.items():
        avg_value = sum(values) / len(values)
        print(f"  {actuator_id}: avg={avg_value:.3f}, range=[{min(values):.3f}, {max(values):.3f}]")
    
    # Find basic cause-effect relationships
    print("Looking for cause-effect patterns...")
    
    for i in range(len(all_nodes) - 1):
        current_node = all_nodes[i]
        next_node = all_nodes[i + 1]
        
        # Simple pattern: did any actuator command correlate with sensor change?
        for actuator_id, actuator_value in current_node.action_taken.items():
            if abs(actuator_value) > 0.1:  # Significant action
                sensor_change = sum(abs(a - b) for a, b in 
                                  zip(next_node.actual_sensory, current_node.actual_sensory))
                if sensor_change > 0.5:  # Significant sensor change
                    print(f"  Pattern: {actuator_id}={actuator_value:.3f} → sensor_change={sensor_change:.3f}")
```

## Phase 5: Continuous Operation

### Main Autonomous Loop
```python
def run_autonomous_learning_loop(brain_state):
    """
    Main loop for continuous autonomous learning and behavior
    Fully integrated with adaptive systems
    """
    print("=== STARTING AUTONOMOUS LEARNING LOOP ===")
    
    pending_prediction = None
    iteration = 0
    
    while True:  # Run indefinitely
        try:
            cycle_start = time.time()
            
            # === MAIN MENTAL LOOP ===
            current_sensory = receive_sensory_data(brain_state['brainstem_connection'])
            
            # Create experience from previous prediction
            if pending_prediction is not None:
                experience = create_experience_node(
                    brain_state['mental_context'].context_vector,
                    pending_prediction,
                    current_sensory['sensor_values']
                )
                
                brain_state['world_graph'].add_node(experience)
                brain_state['mental_context'].update_from_experience(experience)
                
                # Learn from prediction outcome
                prediction_accuracy = 1.0 - experience.prediction_error
                update_adaptive_systems(brain_state['adaptive_systems'], 
                                       prediction_accuracy, pending_prediction, experience)
            
            # Update mental context
            update_mental_context(brain_state['mental_context'], current_sensory)
            
            # Generate new prediction using adaptive systems
            if brain_state['world_graph'].node_count() > 0:
                # Calculate adaptive thinking depth
                recent_accuracies = getattr(brain_state, 'recent_accuracies', [])
                thinking_depth = brain_state['adaptive_systems']['thinking'].calculate_thinking_depth(
                    brain_state['mental_context'].context_vector, recent_accuracies[-10:]
                )
                
                # Generate prediction with adaptive consensus
                prediction_result, consensus_strength = generate_prediction(
                    brain_state['mental_context'].context_vector,
                    brain_state['world_graph'],
                    thinking_depth,
                    brain_state['adaptive_systems']
                )
                
                prediction_result['consensus_strength'] = consensus_strength
                prediction_result['thinking_depth'] = thinking_depth
                
            else:
                # Bootstrap mode - random actions
                prediction_result = generate_random_action(brain_state['brainstem_connection'])
            
            # Send motor commands
            send_motor_commands(brain_state['brainstem_connection'], 
                              prediction_result['motor_action'])
            
            # Prepare next prediction
            pending_prediction = {
                'expected_sensory': prediction_result.get('expected_sensory', []),
                'motor_action': prediction_result['motor_action'],
                'confidence': prediction_result.get('confidence', 0.0),
                'consensus_strength': prediction_result.get('consensus_strength', 'random'),
                'thinking_depth': prediction_result.get('thinking_depth', 0),
                'similarity_components_used': prediction_result.get('similarity_components_used', {}),
                'timestamp': datetime.now(),
                'sequence_id': iteration
            }
            
            # === PERIODIC REPORTING ===
            iteration += 1
            if iteration % 100 == 0:
                report_learning_progress(brain_state, iteration)
            
            # === TIMING ===
            cycle_time = time.time() - cycle_start
            if cycle_time < 0.05:  # Maintain ~20Hz for autonomous operation
                time.sleep(0.05 - cycle_time)
                
        except KeyboardInterrupt:
            print("\n=== SHUTDOWN REQUESTED ===")
            break
        except Exception as e:
            print(f"Error in autonomous loop: {e}")
            # Continue operation but log error
            continue
    
    shutdown_gracefully(brain_state)

def report_learning_progress(brain_state, iteration):
    """
    Report current learning state and statistics
    """
    stats = brain_state['world_graph'].get_graph_statistics()
    
    print(f"\n=== LEARNING PROGRESS (Iteration {iteration}) ===")
    print(f"Total experiences: {stats['total_nodes']}")
    print(f"Memory merges: {stats['total_merges']}")
    print(f"Average node strength: {stats['avg_strength']:.3f}")
    print(f"Strongest memory: {stats['max_strength']:.3f}")
    
    # Report adaptive system states
    adaptive = brain_state['adaptive_systems']
    print(f"Current similarity weights: {adaptive['similarity'].component_weights}")
    print(f"Current decay rate: {adaptive['decay'].current_decay_rate:.6f}")
    print(f"Average thinking depth: {adaptive['thinking'].traversal_count}")
    
    # Estimate learning metrics
    recent_nodes = brain_state['world_graph'].get_nodes_by_strength_range(0.5)
    if recent_nodes:
        avg_prediction_error = sum(node.prediction_error for node in recent_nodes) / len(recent_nodes)
        print(f"Recent prediction accuracy: {1.0 - avg_prediction_error:.3f}")
    
    print("=" * 50)
```

## Error Handling and Recovery

### Graceful Degradation
```python
def handle_system_errors(brain_state, error_type, error_info):
    """
    Handle various system errors gracefully
    """
    print(f"=== HANDLING ERROR: {error_type} ===")
    
    if error_type == "network_disconnection":
        print("Lost connection to brainstem - attempting reconnection...")
        brain_state['brainstem_connection'] = attempt_brainstem_reconnection()
        
    elif error_type == "memory_overflow":
        print("Memory usage too high - forcing aggressive consolidation...")
        force_memory_consolidation(brain_state['world_graph'])
        
    elif error_type == "prediction_failure":
        print("Prediction system failed - falling back to random actions...")
        brain_state['fallback_mode'] = True
        
    elif error_type == "hardware_failure":
        print("Hardware problem detected - entering safe mode...")
        enter_safe_mode(brain_state)
        
    else:
        print(f"Unknown error type: {error_type}")
        print("Continuing with degraded functionality...")

def shutdown_gracefully(brain_state):
    """
    Clean shutdown sequence
    """
    print("=== GRACEFUL SHUTDOWN ===")
    
    # Stop all motors
    safe_commands = {actuator_id: 0.0 for actuator_id in brain_state['actuator_mapping']}
    send_motor_commands(brain_state['brainstem_connection'], safe_commands)
    
    # Save world graph if desired
    # save_world_graph(brain_state['world_graph'], "final_world_model.pkl")
    
    # Close connections
    brain_state['brainstem_connection'].close()
    
    # Final statistics
    final_stats = brain_state['world_graph'].get_graph_statistics()
    print(f"Final learning statistics: {final_stats}")
    
    print("=== SHUTDOWN COMPLETE ===")
```

## Summary

This bootstrap sequence takes the robot from complete ignorance to autonomous learning:

1. **Hardware Discovery** - Find and initialize all sensors and actuators
2. **First Contact** - Process initial sensory flood with random actions  
3. **Random Exploration** - Build initial experience base through pure experimentation
4. **Pattern Emergence** - Simple predictions begin as graph grows
5. **Autonomous Learning** - Full mental loop with adaptive systems

The beauty of this approach is that no behaviors are programmed - everything emerges from the robot's drive to build better predictive models of its world through embodied interaction.