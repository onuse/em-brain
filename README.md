# Minimal Brain

Field-based intelligence with intrinsic motivation. No rewards, just physics and discomfort.

## Quick Start

```bash
# Start the brain server (with smart defaults)
python3 run_server.py

# This automatically:
# - Uses Enhanced Critical Mass Brain (most advanced)
# - Runs with --target speed (optimized for real-time)
# - Auto-saves every 5 minutes
# - Saves when disconnecting
# - Loads previous brain state if it exists

# To start fresh (ignore saved state):
python3 run_server.py --fresh-brain

# In another terminal - monitor the brain
python3 monitor.py

# In another terminal - connect robot  
cd client_picarx && python3 picarx_robot.py --brain-host <IP>
```

## Telemetry Dashboard

Real-time monitoring of brain and brainstem:

```bash
# Terminal dashboard (works over SSH)
python3 monitor.py --terminal

# Graphical dashboard (requires matplotlib)
python3 monitor.py

# Test with mock data
python3 monitor.py --mock
```

Shows:
- Motivation states (BORED, STARVED, ACTIVE, etc.)
- Field energy and comfort levels
- Sensor patterns visualization
- Motor outputs in real-time
- Learning progress

## Core Idea

Intelligence emerges from field discomfort:
- Resting potential creates activity hunger
- Uniformity creates boredom (99.6% of the time!)
- Prediction errors create turbulence
- Turbulence drives exploration

The brain is sophisticated enough to be bored by predictable patterns.

## Architecture

750 lines of core intelligence:
- `truly_minimal_brain.py` - Main brain
- `intrinsic_tensions.py` - Motivation system
- `simple_field_dynamics.py` - Physics
- `simple_prediction.py` - Future prediction
- `simple_learning.py` - Error-driven learning
- `simple_motor.py` - Action generation

Multi-stream sensor architecture:
- TCP 9999: Main control
- TCP 9998: Telemetry broadcast
- UDP 10002: Vision stream
- UDP 10003-10006: Other sensors

## Status

✅ Brain works and gets bored  
✅ Telemetry broadcasting implemented  
✅ Real-time monitoring dashboard  
✅ Multi-stream sensor support  
✅ Persistence (save/load states)  
✅ 97% code reduction achieved  

The brain is ready for deployment!