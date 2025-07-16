# Minimal Brain Architecture

## 📁 **Project Structure**

The minimal brain follows clean separation between project coordination and brain implementation:

```
brain/
├── CLAUDE.md                   # Development instructions
├── README.md                   # Project overview
├── demo.py                     # Interactive demo launcher
├── demo_runner.py              # Direct demo execution
├── test_runner.py              # Test orchestration
├── validation_runner.py        # Scientific validation
│
├── demos/                      # Demonstration applications
│   ├── spatial_learning_demo.py    # Basic 2D spatial learning
│   ├── demo_2d.py, demo_3d.py      # Core demos
│   └── picar_x_simulation/          # PiCar-X simulation demos
│       ├── picar_x_brainstem.py         # Local brainstem
│       ├── picar_x_network_brainstem.py # Network brainstem
│       └── visualization/               # 3D rendering
│
├── docs/                       # Project documentation
│   ├── ARCHITECTURE.md         # This file
│   ├── COMM_PROTOCOL.md        # Communication protocol
│   ├── EMBODIED_FREE_ENERGY.md # Embodied system docs
│   └── IMPLEMENTATION.md       # Implementation details
│
├── validation/                 # Scientific validation
│   ├── embodied_learning/      # Embodied learning experiments
│   └── micro_experiments/      # Micro-validation tests
│
├── client_picarx/              # PiCar-X client implementation
│   ├── src/brainstem/          # Hardware integration
│   └── src/hardware/           # Hardware abstraction
│
├── server/                     # Complete brain implementation
│   ├── brain_server.py         # Main server entry point
│   ├── settings.json           # Server configuration
│   │
│   ├── src/                    # Core brain implementation
│   │   ├── brain.py            # Main brain coordinator
│   │   ├── cognitive_constants.py # Core parameters
│   │   │
│   │   ├── experience/         # System 1: Experience Storage
│   │   │   ├── models.py       # Experience data model
│   │   │   └── storage.py      # Experience database
│   │   │
│   │   ├── similarity/         # System 2: Similarity Search
│   │   │   ├── engine.py       # Core similarity search
│   │   │   ├── adaptive_attention.py # Attention mechanisms
│   │   │   └── learnable_similarity.py # Adaptive similarity
│   │   │
│   │   ├── activation/         # System 3: Activation Dynamics
│   │   │   ├── dynamics.py     # Neural activation spreading
│   │   │   └── utility_based_activation.py # Utility-based activation
│   │   │
│   │   ├── prediction/         # System 4: Prediction Engine
│   │   │   ├── engine.py       # Core prediction
│   │   │   └── adaptive_engine.py # Adaptive prediction
│   │   │
│   │   ├── embodiment/         # Embodied Free Energy System
│   │   │   ├── base.py         # Hardware constraints
│   │   │   ├── system.py       # Free Energy minimization
│   │   │   └── brain_adapter.py # Brain integration
│   │   │
│   │   ├── communication/      # Network communication
│   │   │   ├── protocol.py     # Binary message protocol
│   │   │   ├── tcp_server.py   # TCP server implementation
│   │   │   └── client.py       # Client library
│   │   │
│   │   ├── persistence/        # Memory persistence
│   │   │   ├── manager.py      # Checkpoint management
│   │   │   └── serializer.py   # Data serialization
│   │   │
│   │   └── utils/              # Supporting utilities
│   │       ├── memory_manager.py # Memory optimization
│   │       ├── brain_logger.py   # Logging system
│   │       └── hardware_adaptation.py # Hardware adaptation
│   │
│   ├── tests/                  # Complete test suite
│   │   ├── integration/        # Integration tests
│   │   │   ├── test_brain_learning.py # Brain learning tests
│   │   │   └── test_brain_server.py   # Server tests
│   │   │
│   │   ├── test_minimal_brain.py # Core functionality tests
│   │   ├── test_prediction.py    # Prediction engine tests
│   │   └── test_client_server.py # Communication tests
│   │
│   └── tools/                  # Development tools
│       ├── analysis/           # Performance analysis
│       │   ├── performance_analysis.py
│       │   └── archived/       # Historical analysis
│       │
│       └── experiments/        # Quick experiments
│           ├── quick_brain_test.py
│           └── five_minute_test.py
│
├── logs/                       # Runtime logs
└── robot_memory/               # Persistent brain memory
    ├── checkpoints/            # Memory snapshots
    └── metadata/               # Session metadata
```

## 🏗️ **Core Architecture**

### **The 4 Essential Systems**

1. **Experience Storage** (`experience/`): Stores every sensory-motor moment
2. **Similarity Search** (`similarity/`): Finds similar past experiences
3. **Activation Dynamics** (`activation/`): Neural spreading activation
4. **Prediction Engine** (`prediction/`): Generates actions from patterns

### **Embodied Free Energy System**

The `embodiment/` system provides physics-grounded action selection:
- **Hardware constraints**: Battery, temperature, memory limits
- **Free Energy minimization**: Balances energy conservation with learning
- **Emergent behavior**: Natural energy management without programming

### **Server Architecture**

```
brain_server.py (Entry Point)
    ↓
MinimalBrainServer
    ├── MinimalBrain (4 systems + embodiment)
    │   ├── Experience Storage
    │   ├── Similarity Search
    │   ├── Activation Dynamics
    │   ├── Prediction Engine
    │   └── Embodied Free Energy
    │
    └── MinimalTCPServer
        ├── Binary Protocol Handler
        ├── Client Connection Manager
        └── Message Processing
```

### **Communication Flow**

```
Robot Hardware → Brainstem → TCP Client → [Network] → Brain Server → Response
   (Pi Zero)     (client/)   (protocol)              (server/)
```

## 🎯 **Key Design Principles**

### **1. Clean Separation**
- **Root**: Human-approachable project coordination
- **Server**: Complete brain implementation with development tools
- **Client**: Robot-specific hardware integration

### **2. Core Intelligence Simplicity**
- Only 4 cognitive systems + embodied Free Energy
- No complex cognitive modules
- Intelligence emerges from system interactions

### **3. Production Ready**
- Robust TCP communication with binary protocol
- Comprehensive persistence and checkpointing
- GPU acceleration for similarity search
- Multi-robot support

### **4. Scientific Validation**
- Dedicated validation experiments
- Biological timescale testing
- Empirical behavior analysis

## 📋 **Usage Patterns**

### **Running the Brain Server**
```bash
cd server
python3 brain_server.py
```

### **Running Demos**
```bash
# Interactive demo picker
python3 demo.py

# Direct demo execution
python3 demo_runner.py spatial_learning
```

### **Running Tests**
```bash
# All tests
python3 test_runner.py all

# Specific test
python3 test_runner.py brain_learning
```

### **Running Validation**
```bash
# Scientific validation
python3 validation_runner.py embodied_learning.biological_embodied_learning
```

## 🔧 **Extension Points**

### **Adding New Robot Types**
1. Create client in `client_<robot>/`
2. Implement brainstem using `MinimalBrainClient`
3. Add hardware abstraction layer

### **Adding New Capabilities**
- Capabilities emerge from the 4 systems
- Focus on richer sensors or more experiences
- Don't add new cognitive modules

### **Improving Performance**
- Optimize similarity search (GPU acceleration)
- Enhance memory management
- Improve network protocol efficiency

## 📊 **Architecture Benefits**

1. **Conceptually Simple**: 4 systems + embodied Free Energy
2. **Production Ready**: Robust infrastructure for real deployment
3. **Scientifically Valid**: Validates emergence from simple interactions
4. **Extensible**: Easy to add robots, demos, optimizations
5. **Maintainable**: Clear separation between intelligence and infrastructure

The architecture successfully bridges elegant theory with practical deployment capability.