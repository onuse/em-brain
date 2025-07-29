# Field Brain Architecture

## Core Paradigm

This architecture implements continuous field-based intelligence as an alternative to discrete AI approaches. Cognitive functions emerge from the topology and dynamics of a dynamically-sized continuous field organized by physics principles rather than sensory modalities. The field dimensions adapt to each robot's capabilities.

## Project Structure

```
brain/
├── CLAUDE.md                   # Development instructions  
├── README.md                   # Project overview
├── demo.py                     # Interactive demo launcher
│
├── server/                     # Complete brain implementation
│   ├── brain.py                # Main server entry point
│   ├── settings.json           # Brain configuration
│   │
│   ├── src/                    # Core brain systems
│   │   ├── core/               # Dynamic architecture components
│   │   │   ├── dynamic_brain_factory.py # Creates brains on-demand
│   │   │   ├── brain_service.py        # Session management
│   │   │   ├── adapters.py             # Robot-brain translation
│   │   │   └── robot_registry.py       # Robot capability profiles
│   │   │
│   │   ├── brains/field/       # Field-native brain implementation
│   │   │   ├── dynamic_unified_brain_full.py # Primary brain implementation
│   │   │   ├── pattern_based_motor.py  # Pattern-to-motor without coordinates
│   │   │   ├── pattern_based_attention.py # Salience-based attention
│   │   │   ├── emergent_spatial_dynamics.py # Coordinate-free navigation
│   │   │   ├── emergent_robot_interface.py  # Pattern-based sensing
│   │   │   ├── spontaneous_dynamics.py # Autonomous field activity
│   │   │   ├── blended_reality.py     # Fantasy/reality blending
│   │   │   ├── enhanced_field_dynamics.py # Phase transitions, attractors
│   │   │   └── dynamics/               # Field dynamics systems
│   │   │       ├── constraint_field_nd.py      # N-dimensional constraints
│   │   │       └── temporal_field_dynamics.py   # Temporal patterns
│   │   │
│   │   ├── persistence/        # Brain state management
│   │   │   ├── integrated_persistence.py    # Main persistence system
│   │   │   └── dynamic_persistence_adapter.py # State serialization
│   │   │
│   │   ├── utils/              # Core utilities
│   │   │   ├── cognitive_autopilot.py   # Adaptive processing intensity
│   │   │   ├── hardware_adaptation.py  # Dynamic device selection
│   │   │   ├── brain_logger.py         # Brain state logging
│   │   │   └── async_logger.py         # Non-blocking logging
│   │   │
│   │   └── communication/      # Network architecture
│   │       ├── protocol.py             # Communication protocol
│   │       ├── tcp_server.py           # Multi-client server
│   │       └── monitoring_server.py   # Real-time monitoring
│   │
│   ├── tools/testing/          # Development tools
│   │   └── behavioral_test_framework.py # Intelligence assessment
│   │
│   └── tests/                  # Comprehensive test suite
│       ├── integration/        # End-to-end tests
│       └── unit/               # Component tests
│
├── client_picarx/              # Robot-specific implementation
├── demos/                      # Interactive demonstrations
├── validation/                 # Scientific validation
└── tools/runners/             # Test orchestration
```

## Cognitive Pipeline

The complete flow from sensory input to motor output:

```
1. Sensory Input (24D + reward)
      ↓
2. Energy Assessment (field activity → exploration drive)
      ↓
3. Pattern Recognition (novelty detection via cosine similarity)
      ↓
4. Reward Processing (significant rewards → topology deformation)
      ↓
5. Field Imprinting (strength modulated by energy state)
      ↓
6. Attention Selection (pattern salience: novelty + surprise + importance)
      ↓
7. Field Evolution (spontaneous + sensory + topology influence)
      ↓
8. Action Prediction (preview multiple candidates)
      ↓
9. Value Estimation (score predicted outcomes)
      ↓
10. Action Selection (value + uncertainty exploration)
      ↓
11. Motor Generation (pattern → motor mapping)
      ↓
12. Learning (update predictions from outcomes)
```

### Key Feedback Loops
- **Energy ↔ Behavior**: Low energy drives exploration, which finds patterns that increase energy
- **Prediction ↔ Action**: Actions chosen by predicted outcomes, errors improve predictions
- **Reward ↔ Topology**: Rewards shape field topology, topology guides future behavior toward rewards
- **Confidence ↔ Processing**: High confidence enables fantasy, low confidence demands reality
- **Attention ↔ Memory**: Salient patterns form stronger memories

## Core Architecture

### Dynamic Brain Architecture

The brain creates adaptive continuous fields on-demand based on robot capabilities:

1. **Robot connects** with sensor/actuator specifications
2. **System calculates** optimal field dimensions using logarithmic scaling
3. **Brain is created** with dimensions matching robot complexity
4. **Adapters translate** between robot and brain coordinate spaces

#### Field Organization by Physics Families
```python
# Dimensions organized by physical dynamics, not sensory modalities
# Actual dimensions scale with robot complexity:
SPATIAL:     # Position, orientation, scale, time
OSCILLATORY: # Frequencies, rhythms, neural patterns  
FLOW:        # Gradients, motion, attention flows
TOPOLOGY:    # Stable configurations, boundaries
ENERGY:      # Motor, cognitive, sensory, emotional
COUPLING:    # Correlations, associations, binding
EMERGENCE:   # Novelty, creativity, phase transitions

# Field size = log2(sensory_dims) × complexity_factor × physics_families
```

#### Key Field Operations
- **Field Evolution**: Constraint-guided topology optimization with energy dissipation
- **Action Generation**: Motor commands from multi-dimensional field gradients
- **Memory Formation**: Persistent field topology patterns
- **Learning**: Field adaptation through constraint discovery
- **Maintenance**: Periodic energy dissipation and topology cleanup (off hot path)

### Integrated Field Attention

#### Pattern-Based Attention (Primary)
```python
# Attention emerges from pattern salience, not coordinates
pattern_attention = PatternBasedAttention(field_shape)
attention_state = pattern_attention.process_field_patterns(
    field, sensory_patterns
)
# Salience = novelty + surprise + importance
```
- Pattern salience detection (novelty, surprise, importance)
- Limited attention capacity (5 slots) mimics biology
- Cross-modal binding through temporal synchrony
- No spatial coordinates or gradients
- Prioritizes sensory patterns over field patterns

#### Gradient-Based Attention (Legacy)
```python
# Alternative: Attention from field activation gradients
attention_focus = field.compute_activation_gradients()
processed_input = field.apply_attention_weighting(sensors)
```
- Emerges from field activation gradients
- Integrated into N-dimensional field processing
- Follows field energy gradients for focus
- Computed during field evolution
- Scales with field dimensions

### Multi-Scale Field Processing

#### Hierarchical Processing
```python
# Multi-scale processing emerges from field structure
field_response = field.evolve_across_scales(sensory_input)
hierarchical_output = field.extract_multi_scale_features()
```
- Emerges from N-dimensional field structure
- Scale dimensions built into field organization
- Single field evolution handles all scales
- Integrated multi-scale dynamics

### Memory Systems

#### Field-Native Memory
```python
# Memory as persistent field topology rather than discrete storage
memory_pattern = field.discover_stable_topology(experience)
recall = field.resonate_with_pattern(memory_pattern)
```
- Patterns stored as stable field configurations
- Different decay rates for different memory types
- Consolidation strengthens important memories
- Working memory emerges from temporal field dynamics

#### Universal Pattern Memory
- Cross-modal memory formation for any signal type
- Sparse distributed representations for efficiency
- Temporal correlation tracking across modalities

### Constraint-Based Field Dynamics

#### N-Dimensional Constraint System
```python
# Constraints operate across all dimensions, not just spatial/temporal
constraint_field = ConstraintFieldND(field_shape, dimension_names)
discovered_constraints = constraint_field.discover_constraints(field, gradients)
constraint_forces = constraint_field.enforce_constraints(field)
```

#### Constraint Discovery
```python
# Constraints emerge from field topology across all dimension families
discovered_constraints = field.analyze_topology_patterns()
field.apply_constraint_satisfaction(discovered_constraints)
```
Constraint Types:
- Gradient flow constraints (smoothness)
- Topology boundary constraints (stability)
- Activation threshold constraints (sparsity)
- Temporal momentum constraints (continuity)
- Scale coupling constraints (hierarchy)
- Pattern coherence constraints (binding)
- Cross-dimensional coupling (full field interactions)

#### Self-Organization
- Constraint satisfaction guides field evolution without external programming
- Constraints discovered across all field dimensions
- Efficient sparse constraint representation
- Dimension-aware constraint types

### Field Intelligence Control

#### Intrinsic Field Dynamics
```python
# Field evolution naturally adapts processing intensity
field_confidence = field.assess_pattern_familiarity(input)
processing_intensity = field.adapt_evolution_rate(confidence)
```
- Field adapts processing intensity based on familiarity
- Intelligence emerges from field dynamics
- Confidence-driven processing allocation

#### Learning System
```python
# Intrinsic motivation emerges from field evolution patterns
improvement_rate = field.track_prediction_improvements()
field.modulate_exploration_based_on_learning(improvement_rate)
```
- Field seeks prediction improvements
- Automatic exploration/exploitation balance
- Intrinsic motivation from prediction accuracy
- Motivation emerges from field topology

### Spontaneous Dynamics and Blended Reality

#### Spontaneous Field Activity
```python
# Field maintains autonomous activity without sensory input
spontaneous = SpontaneousDynamics(field_shape)
spontaneous_activity = spontaneous.generate_spontaneous_activity(
    field, sensory_gating=confidence
)
```
- Traveling waves create coherent internal patterns
- Local recurrence maintains persistent activity
- Homeostatic drive balances field energy
- Critical dynamics at edge of chaos
- Generates motor commands autonomously

#### Blended Reality System
```python
# Seamless blending of spontaneous dynamics (fantasy) with sensory input (reality)
blended_reality = BlendedReality()
spontaneous_weight = blended_reality.calculate_spontaneous_weight()
imprint_strength = blended_reality.calculate_imprint_strength(
    base_intensity, has_sensory_input
)
```
- **High confidence** → More fantasy (83% spontaneous, 17% sensory)
- **Low confidence** → More reality (26% spontaneous, 74% sensory)
- **Dream mode** → Pure fantasy (95% spontaneous after 100 idle cycles)
- Smooth transitions with temporal confidence smoothing
- Variable sensory imprint strength based on confidence

#### Cognitive Modes
```python
# Brain autonomously adjusts processing based on prediction confidence
autopilot_state = cognitive_autopilot.update_cognitive_state(
    prediction_confidence, prediction_error, brain_state
)
```
- **AUTOPILOT** (>90% confidence): 20% sensor attention, fantasy-dominated
- **FOCUSED** (70-90% confidence): 50% sensor attention, balanced blend
- **DEEP_THINK** (<70% confidence): 90% sensor attention, reality-focused
- Sensors suppressed but never ignored (max 80% suppression)

#### Prediction-Based Confidence
```python
# Field evolution serves as prediction mechanism
predicted_field = field.evolve()
actual_field = field.after_sensory_input()
prediction_error = compare(predicted, actual)
confidence = 1.0 / (1.0 + prediction_error * 5000)
```
- Field evolution naturally predicts next state
- Prediction errors drive learning
- Confidence based on actual prediction accuracy
- Creates genuine curiosity and surprise detection

#### Value Learning System
```python
# 25th sensory dimension carries reward signal
reward = sensory_input[24]  # -1.0 to +1.0
field.map_reward_to_energy_dimensions(reward)
field.strengthen_memories_by_importance(reward)
```
- External rewards shape field dynamics
- Positive rewards create stronger memories
- Negative rewards create aversive patterns
- Value gradients emerge in field topology

### Emergent Spatial Navigation

#### Coordinate-Free Navigation
```python
# Places emerge from stable field configurations
place = spatial_dynamics.process_spatial_experience(
    current_field, sensory_pattern, reward
)
# Navigation through field tension, not gradients
spatial_dynamics.navigate_to_place(target_place_id)
```
- Places are stable field attractors, not coordinates
- Recognition through field similarity (cosine similarity)
- Navigation via field tension between current and target states
- Transitions learned through experience

#### Pattern-Based Sensing
```python
# Sensory patterns create field impressions directly
experience = robot_interface.sensory_pattern_to_field_experience(
    sensory_input  # No coordinate mapping!
)
```
- Pattern features (symmetry, rhythm, gradient) activate field dimensions
- Different features map to different dynamics families
- No fixed spatial coordinates in sensory processing

#### Motor Emergence from Field Evolution
```python
# Movement emerges from field dynamics patterns
action = spatial_dynamics.compute_motor_emergence(
    current_field, field_evolution
)
```
- Oscillatory patterns → forward/backward motion
- Flow patterns → turning behavior
- Energy patterns → speed/urgency
- Field tension drives goal-directed movement

#### Pattern-Based Motor Generation
```python
# Motor commands from field evolution patterns, not gradients
pattern_motor = PatternBasedMotorGenerator(field_shape, motor_dim)
motor_action = pattern_motor.generate_motor_action(
    field, field_evolution, attention_state
)
```
- Extracts motor tendencies from field evolution patterns
- Maps field dynamics to motor commands without coordinates
- Integrates with pattern-based attention for focused action
- No spatial gradients or coordinate systems

#### Hardware Adaptation
```python
# Dynamic device selection and resource management
if field_dimensions > 16 and device == 'mps':
    device = 'cpu'  # MPS limitation workaround
cognitive_limits = adapt_to_hardware(available_memory, cpu_cores)
```
- Automatic GPU/MPS/CPU selection
- Dynamic memory limits based on available hardware
- Performance monitoring and adaptive scaling

### Persistence System

#### Brain State Management
```python
# Robust cross-session learning with state recovery
brain_state = serialize_complete_brain(brain)
save_incremental_delta(brain_state, previous_state)
recovered_state = recover_brain_state_at_startup()
```
- Complete field state serialization
- Delta compression for efficient updates
- Background memory consolidation
- State recovery with corruption detection
- Cross-session learning continuity

## System Integration

### BrainFactory
```python
# Direct UnifiedFieldBrain wrapper - no brain type complexity
brain = BrainFactory(config)  # Always creates UnifiedFieldBrain
action, brain_state = brain.process_sensory_input(sensors)
```
- Single UnifiedFieldBrain implementation
- No brain type switching complexity
- Automatic hardware device selection
- Integrated state management

### Communication Architecture
```python
# Direct TCP communication with UnifiedFieldBrain
# Sensor input -> N-dimensional field processing -> Motor output
# Input: 24D sensors + 1D reward signal
action, state = brain.process_sensory_input(sensor_vector)
```
- Sensor input → adaptive field processing → motor output
- Input includes 24D sensors + 1D reward signal
- Internal adaptive field intelligence
- TCP server for robot connections
- Designed for real-time operation

## Intelligence Assessment

### Behavioral Test Framework
```python
# Test actual intelligent behaviors rather than technical functionality
assessment = framework.run_intelligence_assessment(brain, intelligence_profile)
```
Intelligence Metrics:
- Prediction accuracy improvement over time
- Exploration vs exploitation balance
- Field energy efficiency and stability
- Pattern recognition and discrimination
- Goal-directed behavior
- Biological memory consolidation
- Computational efficiency metrics

## Key Design Principles

### Continuous Field Intelligence
- Everything emerges from continuous field dynamics
- Dimensions organized by physical dynamics, not sensory categories
- Intelligence emerges from field topology optimization

### Emergent Properties
- Memory: Persistent field topology patterns
- Attention: Pattern salience (novelty, surprise, importance)
- Motor Control: Action-outcome prediction guides behavior
- Learning: Continuous field evolution from prediction errors
- Reasoning: Field dynamics and constraint satisfaction
- Prediction: Actions preview outcomes before execution
- Value: Anticipated outcomes guide action selection
- Curiosity: Uncertainty-driven exploration
- Imagination: Preview multiple action candidates internally
- Dreaming: Pure fantasy states during extended idle
- Anticipation: Field evolution predicts sensory consequences
- Navigation: Places emerge as stable field configurations
- Spatial Understanding: Topological relationships, not coordinates
- Cross-Modal Binding: Temporal synchrony creates unified percepts
- Energy Flow: Exploration↔Consolidation emerges from field activity

### Architectural Simplicity
- Single UnifiedFieldBrain implementation
- No brain type switching complexity
- All features integrated into field dynamics
- Streamlined codebase

### Production Considerations
- Single brain implementation for deployment
- Automatic hardware device selection
- State persistence and recovery
- Maintainable architecture

## Architecture Benefits

1. Single implementation reduces complexity
2. Maintainable codebase with fewer abstractions
3. Adaptive continuous field dynamics as intelligence substrate
4. Automatic hardware device selection
5. No brain type switching complexity
6. Direct field brain interfaces

## Implementation Details

### Gradient Computation
- **Local Region Optimization**: Gradients computed in 3x3x3 local regions around robot position
- **Full Field Compatibility**: Maintains complete field structure for distributed actuators
- **Caching System**: Gradient results cached when field remains stable

### Hardware Adaptation
- **Automatic Resolution Selection**: Spatial resolution determined by hardware benchmarking
  - High performance hardware: 5³ resolution
  - Medium performance hardware: 4³ resolution  
  - Low performance hardware: 3³ resolution
- **Dynamic Limits**: Working memory and search limits scale with hardware capabilities

### Memory Formation
- **Topology Region Discovery**: Activation > 0.02, variance < 0.5
- **Region Persistence**: Removal only when activation < 0.001
- **Baseline Field Value**: 0.0001 (prevents zero without interfering)
- **Reward Modulation**: Positive rewards increase field intensity 0.5-1.0

### Energy Management
- **Organic Energy System**: Energy emerges from field activity intensity
- **No Modes**: Continuous behavioral influence, no hard thresholds
- **Pattern Memory**: Cosine similarity replaces hashing
- **Natural Flow**: Low energy → explore → patterns → high energy → consolidate

### Predictive Action Selection
- **Action Candidates**: Generate exploit, explore, and random options
- **Outcome Preview**: Field evolution simulates action consequences
- **Value Estimation**: Predicted states scored for desirability
- **Uncertainty Bonus**: Unknown outcomes encourage exploration
- **Adaptive Frequency**: Full prediction every 10 cycles or when exploring

### Reward Topology Shaping (Emergent Intentionality)
- **No Goals**: System has no explicit goal representation or planning
- **Field Deformation**: Rewards create persistent "impressions" in field topology
- **Attractors**: Positive rewards → field naturally flows toward rewarded states
- **Repulsors**: Negative rewards → field naturally flows away from punished states
- **Emergence**: Goal-seeking behavior emerges from topology alone
- **Persistence**: Deformations decay slowly (95% persistence factor)
- **Key Insight**: Like gravity warping spacetime, rewards warp field topology

## System Configuration

**Architecture**: Single UnifiedFieldBrain implementation  
**Processing**: Sensor → Energy → Prediction → Action → Motor  
**Hardware**: Automatic GPU/MPS/CPU selection with adaptive spatial resolution  
**Configuration**: settings.json for brain parameters and network settings

### Performance Targets
**Development**: <750ms cycle time acceptable (M1 MacBook)  
**Production**: <150ms target (10x faster hardware)  
**Testing**: <500ms for CI environments  
**Embedded**: <100ms for resource-constrained systems

### Implementation Status

#### Fully Integrated Features
- Pattern-based motor generation (coordinate-free)
- Pattern-based attention (salience-based, no gradients)
- Organic energy system (emerges from field activity)
- Predictive action selection (imagine before acting)
- Reward topology shaping (emergent goal-seeking)
- Persistence system (cross-session learning)
- Blended reality (integrated into field evolution)
- Spontaneous dynamics (always active)

#### Removed Systems (Simplified)
- Enhanced Dynamics: Replaced by organic energy + reward topology
- Developmental Confidence: Exploration emerges from energy alone
- Cognitive Autopilot: Behavior emerges from field dynamics
- BrainFieldAdapter: Direct integration is cleaner
- Configuration Flags: Committed to core design choices

#### Known Limitations
- Topology region formation requires parameter tuning for different resolutions
- GPU processing limited to CPU due to MPS tensor dimension constraints
- Unified field persistence creates large files (100+ MB) without compression
- Hardware deployment interface (picarx_brainstem.py) requires updating
- Predictive action selection adds ~300ms on development hardware (30ms on production)