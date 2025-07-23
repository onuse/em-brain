# Field Brain Architecture

## Core Paradigm

This architecture implements continuous field-based intelligence as an alternative to discrete AI approaches. Cognitive functions emerge from the topology and dynamics of a unified 37-dimensional field organized by physics principles rather than sensory modalities.

## Project Structure

```
brain/
├── CLAUDE.md                   # Development instructions  
├── README.md                   # Project overview
├── demo.py                     # Interactive demo launcher
│
├── server/                     # Complete brain implementation
│   ├── brain_server.py         # Main server entry point
│   ├── settings.json           # Brain configuration
│   │
│   ├── src/                    # Core brain systems
│   │   ├── brain_factory.py    # Simplified UnifiedFieldBrain wrapper
│   │   │
│   │   ├── brains/field/       # Field-native brain implementation
│   │   │   ├── core_brain.py           # 37D field intelligence
│   │   │   ├── memory.py               # Field topology memory
│   │   │   └── dynamics/               # Field dynamics systems
│   │   │       ├── constraint_field_dynamics.py # 4D constraint discovery (legacy)
│   │   │       ├── constraint_field_nd.py      # N-dimensional constraints
│   │   │       └── temporal_field_dynamics.py   # Temporal patterns
│   │   │
│   │   ├── persistence/        # Brain state management
│   │   │   └── brain_serializer.py     # Simplified state save/load
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

## Core Architecture

### UnifiedFieldBrain

The brain is organized as a 37-dimensional continuous field where intelligence emerges from field topology and dynamics:

#### Field Organization by Physics Families
```python
# Dimensions organized by physical dynamics, not sensory modalities:
SPATIAL (5D):     # Position, orientation, scale, time
OSCILLATORY (6D): # Frequencies, rhythms, neural gamma/theta  
FLOW (8D):        # Gradients, motion, attention flows
TOPOLOGY (6D):    # Stable configurations, boundaries
ENERGY (4D):      # Motor, cognitive, sensory, emotional
COUPLING (5D):    # Correlations, associations, binding
EMERGENCE (3D):   # Novelty, creativity, phase transitions
```

#### Key Field Operations
- **Field Evolution**: Constraint-guided topology optimization with energy dissipation
- **Action Generation**: Motor commands from multi-dimensional field gradients
- **Memory Formation**: Persistent field topology patterns
- **Learning**: Field adaptation through constraint discovery
- **Maintenance**: Periodic energy dissipation and topology cleanup (off hot path)

### Integrated Field Attention

#### Field-Native Attention
```python
# Attention emerges from field activation gradients
attention_focus = field.compute_activation_gradients()
processed_input = field.apply_attention_weighting(sensors)
```
- Emerges from field activation gradients
- Integrated into 37D field processing
- Follows field energy gradients for focus
- Computed during field evolution

### Multi-Scale Field Processing

#### Hierarchical Processing
```python
# Multi-scale processing emerges from field structure
field_response = field.evolve_across_scales(sensory_input)
hierarchical_output = field.extract_multi_scale_features()
```
- Emerges from 37D field structure
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
# Constraints operate across all 37 dimensions, not just spatial/temporal
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
- Cross-dimensional coupling (37D interactions)

#### Self-Organization
- Constraint satisfaction guides field evolution without external programming
- Constraints discovered across all 37 dimensions
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
# 24D sensor input -> 37D field processing -> 4D motor output
action, state = brain.process_sensory_input(sensor_vector)
```
- 24D sensor input → 37D field processing → 4D motor output
- Internal 37D field intelligence
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
- Attention: Field activation gradients
- Learning: Continuous field evolution
- Reasoning: Field gradient following and constraint satisfaction

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
3. 37D continuous field dynamics as intelligence substrate
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

## System Configuration

**Architecture**: Single UnifiedFieldBrain implementation  
**Processing**: 24D sensor input → 37D field dynamics → 4D motor output  
**Hardware**: Automatic GPU/MPS/CPU selection with adaptive spatial resolution  
**Configuration**: settings.json for brain parameters and network settings

### Current Limitations
- Constraint enforcement disabled (dimension indexing incompatibility)
- Persistence system disabled (pending state management refactor)
- Topology region formation requires parameter tuning for different resolutions
- GPU processing limited to CPU due to MPS tensor dimension constraints