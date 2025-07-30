# Field Brain Architecture

## Core Paradigm

This architecture implements continuous field-based intelligence as an alternative to discrete AI approaches. Cognitive functions emerge from the topology and dynamics of a continuous field. The brain adapts to each robot's sensory and motor capabilities through dynamic dimensioning.

## System Architecture

### 4D Tensor Architecture

The brain uses a simplified 4D tensor structure:
- **Dimensions**: [32, 32, 32, 64] - spatial volume with feature depth
- **Total Elements**: 2M parameters
- **Hardware**: Optimized for GPU/MPS/CPU execution
- **Performance**: ~250ms cycle time on M1 MacBook

Key principles:
- No semantic encoding in tensor dimensions
- All cognitive properties emerge from field dynamics
- Single unified tensor for all processing
- Hardware-optimized tensor operations

### Cognitive Pipeline

Processing flow from sensory input to motor output:

```
1. Sensory Input (NxD + reward)
      ↓
2. Unified Field Dynamics (energy + confidence + novelty)
      ↓
3. Pattern Recognition (cosine similarity in pattern space)
      ↓
4. Reward Topology (persistent field deformations)
      ↓
5. Field Imprinting (modulated sensory integration)
      ↓
6. Field Evolution (unified dynamics + spontaneous activity)
      ↓
7. Action Prediction (preview candidate outcomes)
      ↓
8. Action Selection (value + exploration balance)
      ↓
9. Motor Generation (pattern-based mapping)
      ↓
10. Learning (prediction error updates)
```

### Core Systems

#### Unified Field Dynamics
Combines energy, confidence, and behavioral modulation into one coherent system:
- **Energy**: Field activity intensity (low = hungry/explore, high = satiated/consolidate)
- **Confidence**: Prediction accuracy tracking
- **Modulation**: Continuous balance between internal dynamics and external input
- **Dream States**: Pure internal dynamics after extended idle
- **Temporal Persistence**: Differential decay rates create working memory (spatial features decay fast, temporal features persist)

#### Evolved Field Dynamics (Core System)
The field encodes its own evolution rules:
- **Dynamic Parameters**: Last 16 features encode local decay, diffusion, coupling, plasticity
- **Topology-Driven Evolution**: Each region develops specialized dynamics
- **Meta-Learning**: System learns how to learn through experience
- **Gradual Emergence**: Starts with minimal self-modification, naturally increases
- **True Autonomy**: No fixed parameters - everything evolves

#### Pattern-Based Processing
All cognitive functions operate on patterns without coordinates:
- **Attention**: Pattern salience (novelty + surprise + importance)
- **Motor Generation**: Field evolution patterns → motor commands
- **Sensory Processing**: Pattern features → field impressions
- **Memory**: Stable field configurations

#### Predictive Action System
Actions selected through outcome imagination:
- Generate candidate actions (exploit/explore/random)
- Preview outcomes via field evolution
- Select based on predicted value + uncertainty bonus
- Learn from prediction errors

#### Reward Topology Shaping
Goal-seeking emerges from field topology:
- Rewards create persistent "impressions" in field
- Positive rewards → attractors in field space
- Negative rewards → repulsors in field space
- Behavior naturally flows toward rewarded states

#### Spontaneous Dynamics
Autonomous field activity without input:
- Traveling waves maintain coherent patterns
- Local recurrence creates persistent activity
- Homeostatic balance of field energy
- Critical dynamics at edge of chaos

#### Topology Region System
Stable patterns form persistent structures:
- **Region Detection**: Identifies stable field configurations
- **Causal Tracking**: Links temporal sequences of activations
- **Abstraction Formation**: Composes regions into higher-level concepts
- **Memory Consolidation**: Strengthens important regions during idle

#### Consolidation System
Advanced learning during rest periods:
- **Pattern Replay**: Reactivates important patterns
- **Dream Generation**: Creates novel pattern combinations
- **Topology Refinement**: Optimizes field organization
- **Cross-scale Integration**: Links patterns across resolutions

#### Emergent Sensory Mapping
Patterns find their natural place in the field:
- **Resonance Detection**: Patterns locate based on field resonance
- **Correlation Clustering**: Similar patterns become neighbors
- **Reward-based Importance**: Critical patterns claim prime locations
- **Self-Organization**: No fixed mappings - organization emerges

### Dynamic Brain Creation

1. **Robot Connection**: Robot provides sensory/motor dimensions via handshake
2. **Brain Creation**: System creates brain adapted to robot's capabilities
3. **Dimension Locking**: Brain locks to first robot's dimensions
4. **Adapter Mapping**: Translates between robot space and field space

### Persistence System

Cross-session learning through state management:
- Complete field state serialization
- Delta compression for incremental updates
- Background consolidation during idle
- Corruption detection and recovery

## Project Structure

```
brain/
├── server/                     # Brain implementation
│   ├── brain.py               # Main server entry
│   ├── settings.json          # Configuration
│   │
│   ├── src/
│   │   ├── core/              # Architecture components
│   │   │   ├── simplified_brain_factory.py
│   │   │   ├── brain_service.py
│   │   │   ├── adapters.py
│   │   │   └── robot_registry.py
│   │   │
│   │   ├── brains/field/      # Field brain implementation
│   │   │   ├── simplified_unified_brain.py
│   │   │   ├── evolved_field_dynamics.py
│   │   │   ├── unified_pattern_system.py
│   │   │   ├── pattern_motor_adapter.py
│   │   │   ├── pattern_attention_adapter.py
│   │   │   ├── predictive_action_system.py
│   │   │   ├── reward_topology_shaping.py
│   │   │   ├── topology_region_system.py
│   │   │   ├── consolidation_system.py
│   │   │   ├── emergent_sensory_mapping.py
│   │   │   └── motor_cortex.py
│   │   │
│   │   ├── persistence/       # State management
│   │   └── communication/     # Network protocol
│   │
│   └── tests/                 # Test suite
│
├── client_picarx/             # Robot implementation
├── demos/                     # Interactive demos
├── validation/                # Scientific validation
└── tools/                     # Development tools
```

## Communication Protocol

TCP-based protocol with capability negotiation:

```
Handshake:
Client → Server: [robot_version, sensory_dim, motor_dim, hardware_type]
Server → Client: [brain_version, field_dimensions, capabilities]

Runtime:
Client → Server: [sensor_data...] 
Server → Client: [motor_data...]
```

## Key Design Principles

### Emergence Over Engineering
- Complexity emerges from simple dynamics
- No explicit goals, modes, or thresholds
- Continuous functions replace discrete states
- Minimal code for maximum capability

### Field-Native Intelligence
- All computation through field dynamics
- No coordinate systems or spatial mappings
- Pattern-based processing throughout
- Memory as field topology

### Biological Inspiration
- Energy as metabolic state
- Confidence as prediction accuracy
- Attention as limited resource
- Dreams as pure internal dynamics

## Performance Characteristics

### Resource Usage
- **CPU**: ~10% utilization (mostly idle between cycles)
- **GPU**: ~4% utilization (efficient tensor operations)
- **Memory**: ~8MB for field tensor
- **Cycle Time**: 250ms (biological timescale)

### Scaling Properties
- Fixed tensor size (no memory growth)
- Constant computation per cycle
- Efficient for embedded deployment
- Headroom for larger tensors if needed

## Implementation Status

### Core Features
- 4D simplified tensor architecture
- Unified field dynamics system
- Pattern-based processing
- Predictive action selection
- Reward topology shaping
- Working memory through temporal persistence
- Topology regions for abstraction
- Consolidation and dream states
- Spontaneous dynamics
- Dynamic robot adaptation

### Complete Emergence
- **Evolved Field Dynamics**: Evolution rules are part of the field itself
- **Regional Specialization**: Each area develops unique dynamics
- **Self-Modification**: Strength grows naturally with experience
- **Emergent Sensory Organization**: Patterns find their place through resonance
- **No Fixed Architecture**: Everything emerges from field dynamics

### Known Limitations
- Large persistence files without compression
- MPS limited to 16 dimensions
- Parameter tuning needed for different resolutions

## System Configuration

Configuration through `settings.json`:
- Network parameters (host, port)
- Brain parameters (spatial resolution, learning rates)
- Hardware preferences (device selection)
- Persistence settings (save frequency, location)