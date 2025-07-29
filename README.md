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
- **2.2x Performance Gain**: GPU acceleration now possible (246ms vs 537ms)
- **Removed Complexity**: Enhanced Dynamics, Developmental Confidence, Cognitive Autopilot
- **Unified Systems**: Organic energy, predictive actions, reward topology shaping
- **True Emergence**: No semantic structure in tensors - all properties emerge

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

#### Pattern-Based Intelligence
- **Motor Generation**: Field evolution patterns → motor commands
- **Attention**: Pattern salience (novelty, surprise, importance)
- **Sensory Processing**: Pattern features → field impressions
- No spatial coordinates or gradients used

#### Self-Organizing Dynamics
- **Constraint System**: N-dimensional constraint discovery and enforcement
- **Phase Transitions**: Stable → high energy → chaotic → low energy
- **Attractors**: Stable field configurations as memory anchors
- **Energy Management**: Redistribution for optimal dynamics

#### Autonomous Behavior
- **Spontaneous Activity**: Traveling waves, local recurrence, homeostasis
- **Cognitive Modes**: AUTOPILOT (>90% confidence), FOCUSED (70-90%), DEEP_THINK (<70%)
- **Dream States**: Pure fantasy after extended idle
- **Predictive Processing**: Field evolution as future state anticipation

#### Navigation and Memory
- **Emergent Places**: Stable field configurations, not coordinates
- **Field Tension**: Navigation through field state differences
- **Topology Memory**: Persistent patterns in field structure
- **Experience Integration**: Reward-modulated memory formation

## Implementation Status

### Fully Integrated Features
- Simplified 4D tensor architecture with GPU acceleration
- Organic energy system (emerges from field activity)
- Predictive action selection (imagine before acting)
- Reward topology shaping (emergent goal-seeking)
- Pattern-based motor and attention (coordinate-free)
- Integrated blended reality (confidence-based mixing)
- Spontaneous dynamics (always active)
- Persistence for cross-session learning

### Recently Removed (Simplifications)
- Enhanced Dynamics → Replaced by organic energy + reward topology
- Developmental Confidence → Exploration emerges from energy alone
- Cognitive Autopilot → Behavior emerges from field dynamics
- Complex dimension mapping → Simple 4D tensor
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
- `dynamic_unified_brain_full.py` - Primary brain implementation
- `pattern_based_*.py` - Coordinate-free cognitive systems
- `constraint_field_nd.py` - Self-organization dynamics
- `integrated_persistence.py` - Cross-session learning