# Field-Native Intelligence System

Research implementation of continuous multi-dimensional field dynamics as an alternative to discrete pattern-based AI approaches.

See docs/ folder for technical documentation:
- **ARCHITECTURE.md** - System architecture and field organization
- **TODO.md** - Pending tasks and implementation status
- **CLAUDE.md** - Development instructions

## Quick Start

### Installation & Test
```bash
pip install numpy torch psutil  # Core dependencies
python3 tests/integration/test_installation.py

# Run demos
python3 demo.py  # Interactive launcher
python3 tools/runners/demo_runner.py spatial_learning
python3 tools/runners/demo_runner.py blended_reality
```

**Requirements:** Python 3.8+, 4GB+ RAM, CPU/GPU

## Core Approach

This project implements continuous field dynamics as an alternative to discrete AI. Intelligence emerges from the topology and evolution of a continuous multi-dimensional field.

### Recent Major Improvements
- **Simplified to 4D Tensor**: From complex 11D to simple [32,32,32,64] tensor
- **2.2x Performance Gain**: GPU acceleration now possible (~183ms optimized)
- **Unified Systems**: Energy + confidence + spontaneous dynamics in one system
- **Working Memory**: Temporal persistence through differential decay rates
- **Topology Regions**: Abstraction formation and causal relationship tracking
- **Consolidation System**: Pattern replay and dream states during idle
- **True Emergence**: No semantic structure in tensors - all properties emerge

### 🚀 Latest Achievement: Fully Emergent Brain
The brain now features complete emergence with:
- **Self-Modifying Dynamics**: Evolution rules encoded in the field itself
- **Emergent Sensory Mapping**: Patterns find natural locations through resonance
- **True Autonomy**: System determines how to learn
- **Open-Ended Evolution**: No ceiling on complexity
- **Meta-Learning**: Field improves its own learning
- **No Fixed Architecture**: Everything emerges from field dynamics

### Field Organization (Simplified)
- **4D Unified Field**: Simple tensor where all properties emerge
- **No Semantic Encoding**: Properties like spatial, temporal, oscillatory emerge naturally
- **GPU Optimized**: Works efficiently on CUDA, MPS, and CPU
- **Larger Capacity**: 2M elements for richer dynamics

### Key Features
- **Pattern-Based Processing**: All cognition through pattern recognition, no fixed coordinates
- **Self-Organization**: Constraint discovery and enforcement guide field evolution
- **Spontaneous Dynamics**: Autonomous field activity without sensory input
- **Blended Reality**: Confidence-based mixing of internal simulation with sensory data
- **Cross-Session Learning**: Persistence system maintains brain state across runs

## Architecture Overview

### Dynamic Brain Creation
1. Robot connects with sensor/actuator specifications
2. System calculates optimal field dimensions (logarithmic scaling)
3. Brain created with dimensions matching robot complexity
4. Adapters translate between robot and brain spaces

### Cognitive Systems

#### Evolved Field Dynamics
- **Self-Modifying**: Evolution rules encoded in the field itself
- **Energy & Confidence**: Emerge from field state, not fixed parameters
- **Spontaneous Activity**: Traveling waves with learned parameters
- **Working Memory**: Temporal persistence through differential decay
- **Meta-Learning**: System improves its learning over time

#### Pattern-Based Intelligence
- **Unified Pattern System**: Shared extraction for motor and attention
- **Motor Generation**: Field patterns → motor commands via motor cortex
- **Attention**: Pattern salience (novelty + surprise + importance)
- **No coordinates**: All processing through pattern matching

#### Self-Organizing Dynamics
- **Topology Regions**: Stable patterns form persistent structures
- **Causal Tracking**: Temporal sequences create relationships
- **Abstraction Formation**: Regions compose into concepts
- **Reward Topology**: Goals emerge from field deformations

#### Autonomous Behavior
- **Predictive Actions**: Imagine outcomes before acting
- **Exploration Drive**: Emerges from low energy + high novelty
- **Dream States**: Consolidation and pattern replay during idle
- **No fixed modes**: Continuous energy/confidence dynamics

#### Navigation and Memory
- **Emergent Places**: Stable field configurations, not coordinates
- **Field Tension**: Navigation through field state differences
- **Topology Memory**: Persistent patterns in field structure
- **Experience Integration**: Reward-modulated memory formation

## Implementation Status

### Fully Integrated Features
- Simplified 4D tensor architecture with GPU acceleration
- Evolved field dynamics (self-modifying evolution rules)
- Working memory through temporal persistence
- Topology regions for abstraction and memory
- Consolidation system with dreams and replay
- Predictive action selection (imagine before acting)
- Reward topology shaping (emergent goal-seeking)
- Pattern-based motor and attention (coordinate-free)
- Persistence for cross-session learning

### Recently Removed (Simplifications)
- Enhanced Dynamics → Replaced by unified field dynamics
- Developmental Confidence → Exploration emerges from energy
- Cognitive Autopilot → Continuous dynamics instead of modes
- Coordinate systems → Pure pattern-based processing
- 11D tensor architecture → 4D GPU-optimized tensor

### Known Limitations
- Large persistence files (100+ MB) without compression
- SimplifiedUnifiedBrain not yet integrated everywhere
- Hardware deployment interface requires updating

## Project Structure

```
brain/
├── README.md                  # This file
├── demo.py                    # Interactive demos
├── server/                    # Brain implementation
│   ├── brain.py              # Main entry point
│   ├── src/
│   │   ├── core/             # Dynamic architecture
│   │   ├── brains/field/     # Field intelligence
│   │   └── persistence/      # State management
│   └── settings.json         # Configuration
├── tests/                    # Test suite
├── tools/                    # Development tools
└── docs/                     # Documentation
```

## Communication Protocol

TCP protocol with capability negotiation:
```
# Handshake
Client → Server: [robot_version, sensory_size, action_size, hardware_type]
Server → Client: [brain_version, accepted_sizes, capabilities]

# Runtime
Client → Server: [sensor_data...] 
Server → Client: [action_data...]
```

## Research Foundation

The field approach builds on:
- **Free Energy Principle**: Intelligence as prediction error minimization
- **Embodied Cognition**: Physical constraints enable intelligence
- **Complex Systems**: Simple interactions create complex behaviors
- **Constraint Satisfaction**: Intelligence as optimization under constraints

## Development

See CLAUDE.md for development instructions. Key components:
- `simplified_unified_brain.py` - Primary brain implementation
- `evolved_field_dynamics.py` - Self-modifying field evolution
- `topology_region_system.py` - Abstraction and memory formation
- `consolidation_system.py` - Learning during rest
- `unified_pattern_system.py` - Shared pattern extraction

### Running the System

```bash
# Start brain server
python3 server/brain.py

# Run tests
python3 tools/runners/test_runner.py all

# Run demos
python3 tools/runners/demo_runner.py spatial_learning

# Run validation studies
python3 tools/runners/validation_runner.py biological_embodied_learning
```