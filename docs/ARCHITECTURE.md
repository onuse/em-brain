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
│   │   ├── brain.py            # Main brain coordinator (vector stream orchestrator)
│   │   ├── cognitive_constants.py # Core parameters
│   │   │
│   │   ├── vector_stream/      # Vector Stream Brain
│   │   │   └── minimal_brain.py # 3-stream processing engine
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
│   │   └── utils/              # Supporting utilities
│   │       ├── memory_manager.py # Memory optimization
│   │       ├── brain_logger.py   # Logging system
│   │       └── hardware_adaptation.py # Hardware adaptation
│   │
│   ├── tests/                  # Complete test suite
│   │   ├── integration/        # Integration tests
│   │   │   ├── test_brain_learning.py # Vector stream learning tests
│   │   │   └── test_brain_server.py   # Server tests
│   │   │
│   │   ├── test_minimal_brain.py # Vector stream functionality tests
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

### **The 3 Vector Streams**

1. **Sensory Stream** (`sensory_stream`): Continuous sensory pattern processing with rolling buffer
2. **Motor Stream** (`motor_stream`): Motor command patterns and cross-stream learning
3. **Temporal Stream** (`temporal_stream`): Biological timing rhythms and temporal context

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
    ├── MinimalBrain (vector stream coordinator)
    │   ├── MinimalVectorStreamBrain
    │   │   ├── Sensory Stream (pattern learning + rolling buffer)
    │   │   ├── Motor Stream (cross-stream prediction)
    │   │   └── Temporal Stream (organic metronome)
    │   └── EmbodiedFreeEnergySystem
    │       ├── Hardware Telemetry
    │       ├── Precision-Weighted Priors
    │       └── Free Energy Minimization
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
- Only 3 vector streams + embodied Free Energy  
- No complex cognitive modules
- Intelligence emerges from stream interactions and physics constraints

### **3. Production Ready**
- Robust TCP communication with binary protocol
- Rolling buffer memory management
- Hardware adaptation and telemetry
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

1. **Conceptually Simple**: 3 vector streams + embodied Free Energy
2. **Biologically Plausible**: Continuous processing matches neural dynamics
3. **Production Ready**: Robust infrastructure for real deployment
4. **Scientifically Testable**: Clear limitations and capabilities to validate
5. **Maintainable**: Clean separation between stream processing and infrastructure

The architecture provides a testable minimal substrate for investigating whether basic intelligent behaviors can emerge from simple continuous processing mechanisms.