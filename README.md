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

### Dynamic Field Architecture
- **Adaptive Dimensionality**: Field dimensions are dynamically created based on robot capabilities
- **On-Demand Brain Creation**: Brains are created when robots connect, not at server startup
- **Logarithmic Scaling**: Field complexity scales logarithmically with robot complexity
- **Physics-Based Organization**: Dimensions organized by physics families:
  - **Spatial Dimensions**: Position, orientation, scale, time
  - **Oscillatory Dimensions**: Frequencies, rhythms, periods
  - **Flow Dimensions**: Gradients, momentum, direction
  - **Topology Dimensions**: Stable configurations, boundaries
  - **Energy Dimensions**: Intensity, activation, depletion
  - **Coupling Dimensions**: Relationships, correlations, binding
  - **Emergence Dimensions**: Novelty, creativity, phase transitions

### Implementation Status
- Dynamic brain architecture with on-demand creation
- Adaptive field dimensions based on robot capabilities
- TCP server with capability negotiation protocol
- Hardware adaptation for GPU/MPS/CPU
- State persistence and recovery
- Automatic maintenance and resource management

## Theoretical Background

The field approach draws from several research areas:

- **Free Energy Principle** (Friston): Intelligence as prediction error minimization under constraints
- **Embodied Cognition** (Varela): Physical constraints enable rather than limit intelligence
- **Sparse Coding** (Olshausen): Resource constraints create efficient representations
- **Complex Systems Theory**: Simple interactions create complex behaviors
- **Constraint Satisfaction**: Intelligence as optimization under competing constraints

## Implementation Components

### Field Dynamics
- Dynamic field dimensions (calculated per robot)
- Constraint discovery from field topology
- Temporal dynamics with working memory emergence
- Cross-scale hierarchical processing
- Adaptive complexity based on robot sensors/actuators

### Brain Architecture
- Dynamic brain creation on robot connection
- UnifiedFieldBrain with adaptive dimensions
- Hardware adaptation (GPU/MPS/CPU)
- TCP protocol with capability negotiation
- State persistence and recovery
- Brain pooling for resource efficiency

### Current Capabilities
- Dynamic dimension calculation from robot profile
- Robot sensor input → adaptive field mapping
- Field evolution and constraint satisfaction
- Field gradients → robot action generation
- Cross-session learning and memory
- Multi-robot support with different capabilities

## Architecture

### Field Processing
- Robot sensors mapped to dynamic field coordinates
- Dimension calculation: log₂(sensors) × complexity_factor
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
│   ├── brain.py              # Main entry point
│   ├── src/core/             # Dynamic architecture components
│   │   ├── dynamic_brain_factory.py  # Creates brains on-demand
│   │   ├── brain_service.py         # Session management
│   │   ├── adapters.py              # Robot-brain translation
│   │   └── robot_registry.py        # Robot profiles
│   ├── src/brains/field/     # Field-native intelligence
│   │   ├── unified_field_brain.py   # Adaptive field brain
│   │   └── dynamics/                 # Field dynamics families
│   └── settings.json         # Configuration
├── validation/               # Scientific experiments
├── tests/                    # Test suite
├── client_picarx/           # Robot client
└── logs/                    # Runtime data
```

## Communication Protocol

Dynamic TCP protocol with capability negotiation:
```
# Handshake
Client → Server: [robot_version, sensory_size, action_size, hardware_type, capabilities]
Server → Client: [brain_version, accepted_sensory_size, accepted_action_size, gpu_available, brain_capabilities]

# Runtime
Client → Server: [sensor_data...] (using negotiated dimensions)
Server → Client: [action_data...] (using negotiated dimensions)
```

## Current Status

This is an active research project exploring field-based intelligence. The implementation includes:

- Dynamic field dimensions adapted to each robot
- On-demand brain creation and management
- Constraint discovery and satisfaction
- Robot communication with capability negotiation
- Hardware adaptation and optimization
- Resource pooling and automatic maintenance

## Research Goals

- Test field dynamics as alternative to discrete AI
- Validate constraint-based intelligence emergence
- Develop robot control through field gradients
- Optimize performance for real-time operation