# Project Structure

## Root Directory (Clean!)

```
em-brain/
├── README.md              # Project overview
├── CLAUDE.md              # AI assistant instructions
├── demo.py                # Interactive brain demo
├── run_unit_tests.sh      # Test runner script
│
├── server/                # Brain server
│   ├── brain.py           # Main server entry point
│   ├── settings.json      # Configuration
│   ├── requirements.txt
│   └── src/
│       ├── brains/field/  # PureFieldBrain implementation
│       ├── streams/       # Parallel sensor injection
│       ├── core/          # Server infrastructure
│       └── communication/ # TCP/UDP protocols
│
├── client_picarx/         # Robot client
│   ├── picarx_robot.py   # Main robot entry point
│   ├── deploy.sh          # Deployment script
│   ├── requirements.txt
│   ├── config/
│   │   └── robot_config.json  # Robot configuration
│   └── src/
│       ├── brainstem/     # Brain-robot interface
│       ├── hardware/      # Hardware abstraction
│       └── streams/       # UDP vision streaming
│
├── demos/                 # Visualization demos
│   ├── demo_3d.py         # 3D brain visualization
│   └── picar_x_simulation/  # Robot simulation
│
├── docs/                  # Documentation
│   ├── TODO.md            # Current tasks
│   ├── ARCHITECTURE.md   # System architecture
│   ├── COMM_PROTOCOL.md  # Protocol specification
│   └── deployment/       # Deployment guides
│       ├── DEPLOYMENT_READY.md
│       └── DEPLOYMENT_STATUS.md
│
├── tests/                 # Test suites
│   ├── integration/       # Integration tests
│   ├── unit/              # Unit tests
│   └── scratch/           # Temporary test scripts
│
├── tools/                 # Development tools
│   └── analysis/          # Performance analysis
│
└── validation/            # Behavioral validation
    └── embodied_learning/ # Learning experiments
```

## Key Files for Deployment

### Brain Server
- **Entry**: `server/brain.py`
- **Config**: `server/settings.json`
- **Brain**: `server/src/brains/field/pure_field_brain.py`
- **Vision Thread**: `server/src/streams/vision_field_injector.py`

### Robot Client
- **Entry**: `client_picarx/picarx_robot.py`
- **Config**: `client_picarx/config/robot_config.json`
- **Brainstem**: `client_picarx/src/brainstem/brainstem.py`
- **Vision Stream**: `client_picarx/src/streams/vision_stream.py`

## Quick Start

### 1. Start Brain Server
```bash
cd server
python3 brain.py
```

### 2. Deploy to Robot
```bash
cd client_picarx
./deploy.sh
```

### 3. Run Robot
```bash
# On Raspberry Pi
python3 picarx_robot.py --brain-host <SERVER-IP>
```

## Configuration

### Vision Resolution
Set in `client_picarx/config/robot_config.json`:
```json
{
  "vision": {
    "resolution": [640, 480],  // Full resolution!
    "fps": 30
  }
}
```

### Brain Parameters
Set in `server/settings.json`:
```json
{
  "network": {
    "port": 9999,
    "enable_streams": true
  }
}
```

## Recent Cleanup

Moved to `tests/scratch/`:
- All test_*.py files from root
- verify_*.py debugging scripts

Moved to `docs/deployment/`:
- DEPLOYMENT_STATUS.md
- DEPLOYMENT_READY.md
- FINAL_IMPLEMENTATION_PLAN.md
- PARALLEL_INJECTION_SUCCESS.md

## Status

✅ **Ready for Deployment** with:
- 640x480 vision via parallel UDP streaming
- 50Hz+ brain performance
- Dynamic dimension negotiation
- Automatic vision thread spawning