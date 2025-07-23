# Field-Native Intelligence System

Research implementation of continuous multi-dimensional field dynamics as an alternative to discrete pattern-based AI approaches.

See docs/ folder for technical documentation:
- **ARCHITECTURE.md** - System architecture and field organization
- **COMM_PROTOCOL.md** - TCP communication protocol specification
- **PROJECT_HISTORY.md** - Evolution from discrete to field-native intelligence

## Quick Start

### Installation & Test
```bash
pip install numpy torch psutil  # Core dependencies
python3 tests/integration/test_installation.py

# Field brain demos
python3 tools/runners/demo_runner.py spatial_learning
python3 demo.py  # Interactive launcher

# Configure field vs discrete brain in server/settings.json:
{"brain": {"type": "field"}}  # Field-native intelligence
{"brain": {"type": "sparse_goldilocks"}}  # Legacy discrete
```

**Requirements:** Python 3.8+, 4GB+ RAM, GPU recommended

## Core Approach

This project explores continuous field dynamics as an alternative to discrete pattern-based AI. Instead of storing patterns in discrete memory structures, intelligence emerges from the topology and evolution of a continuous multi-dimensional field.

### Field Organization
- **37D Unified Field** organized by physics families rather than sensory modalities
- **Spatial Dimensions** (5D): Position, orientation, scale, time
- **Oscillatory Dimensions** (6D): Frequencies, rhythms, periods
- **Flow Dimensions** (8D): Gradients, momentum, direction
- **Topology Dimensions** (6D): Stable configurations, boundaries
- **Energy Dimensions** (4D): Intensity, activation, depletion
- **Coupling Dimensions** (5D): Relationships, correlations, binding
- **Emergence Dimensions** (3D): Novelty, creativity, phase transitions

### Implementation Status
- Field-native brain implementation with constraint discovery
- TCP server for robot communication
- Hardware adaptation for GPU/MPS/CPU
- State persistence and recovery

## Theoretical Background

The field approach draws from several research areas:

- **Free Energy Principle** (Friston): Intelligence as prediction error minimization under constraints
- **Embodied Cognition** (Varela): Physical constraints enable rather than limit intelligence
- **Sparse Coding** (Olshausen): Resource constraints create efficient representations
- **Complex Systems Theory**: Simple interactions create complex behaviors
- **Constraint Satisfaction**: Intelligence as optimization under competing constraints

## Implementation Components

### Field Dynamics
- Continuous 37D field evolution
- Constraint discovery from field topology
- Temporal dynamics with working memory emergence
- Cross-scale hierarchical processing

### Brain Architecture
- UnifiedFieldBrain as primary implementation
- Hardware adaptation (GPU/MPS/CPU)
- TCP communication protocol
- State persistence and recovery

### Current Capabilities
- Robot sensor input → field coordinates mapping
- Field evolution and constraint satisfaction
- Field gradients → robot action generation
- Cross-session learning and memory

## Architecture

### Field Processing
- Robot sensors mapped to 37D field coordinates
- Field evolution through constraint satisfaction
- Gradient calculation for action generation
- Topology analysis for memory and learning

### Technical Features
- Hardware-adaptive device selection
- Sparse field updates for performance
- Biological optimizations (attention, hierarchy)
- Persistent field state across sessions

## Project Structure

```
brain/
├── README.md, CLAUDE.md       # Documentation
├── demo.py                    # Interactive demos
├── tools/runners/             # Demo/test/validation runners
├── server/                    # Complete brain implementation
│   ├── src/brain.py          # Main coordinator
│   ├── src/brains/field/     # Revolutionary field-native intelligence
│   │   ├── generic_brain.py  # Platform-agnostic 37D field brain
│   │   ├── core_brain.py     # Robot-specific field brain
│   │   └── dynamics/         # Field dynamics families
│   ├── src/brains/sparse_goldilocks/  # Legacy discrete brain (constraint-based)
│   └── settings.json         # Configuration (brain_type: "field" vs "discrete")
├── validation/               # Scientific experiments
├── tests/                    # Test suite
├── client_picarx/           # Robot client
└── logs/                    # Runtime data
```

## Communication Protocol

Simple TCP protocol for robot communication:
```
Client → Server: [sensor_vector_length, sensor_data...]
Server → Client: [action_vector_length, action_data...]
```

## Current Status

This is an active research project exploring field-based intelligence. The implementation includes:

- 37D continuous field dynamics
- Constraint discovery and satisfaction
- Robot communication interface
- Hardware adaptation and optimization

## Research Goals

- Test field dynamics as alternative to discrete AI
- Validate constraint-based intelligence emergence
- Develop robot control through field gradients
- Optimize performance for real-time operation