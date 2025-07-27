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

This project implements continuous field dynamics as an alternative to discrete AI. Intelligence emerges from the topology and evolution of a continuous multi-dimensional field organized by physics principles rather than sensory categories.

### Field Organization
- **Adaptive Dimensionality**: Field dimensions created dynamically based on robot capabilities
- **Physics-Based Structure**: Dimensions organized by dynamics families:
  - **Spatial**: Position, orientation, scale, time
  - **Oscillatory**: Frequencies, rhythms, periods
  - **Flow**: Gradients, momentum, direction
  - **Topology**: Stable configurations, boundaries
  - **Energy**: Intensity, activation, depletion
  - **Coupling**: Relationships, correlations, binding
  - **Emergence**: Novelty, creativity, phase transitions

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
- Pattern-based motor and attention systems
- Emergent spatial navigation
- Enhanced field dynamics with phase transitions
- Constraint enforcement for self-organization
- Persistence for cross-session learning
- Blended reality processing
- Spontaneous dynamics
- Cognitive autopilot

### Known Limitations
- Large persistence files (100+ MB) without compression
- GPU processing limited to CPU for high-dimensional fields
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