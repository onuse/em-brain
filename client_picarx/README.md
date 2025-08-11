# PiCar-X Brain Client

## 🧠 Embodied Intelligence for Artificial Life

This is the robotic brainstem that connects the SunFounder PiCar-X robot to an emergent field-based brain. The system enables true artificial life experiments where intelligence emerges from experience, not programming.

## ✨ Key Features

- **No Reward Signals** - Brain discovers good/bad through pure experience
- **Event-Driven Architecture** - Biological-inspired parallel processing
- **Safety Reflexes** - Work even without brain connection
- **16-Channel Sensory Input** - Full environmental awareness
- **5-Channel Motor Control** - Precise movement and camera control
- **Graceful Degradation** - Continues operating with failures
- **Zero Magic Numbers** - Everything configurable

## 🏗️ Architecture

The brainstem uses a nucleus-based architecture inspired by biological nervous systems:

```
Robot Hardware ←→ Brainstem Nuclei ←→ Event Bus ←→ Brain Server
                        ↓
                  Safety Reflexes
                 (Always Active)
```

See [Architecture Documentation](docs/architecture/BRAINSTEM_ARCHITECTURE.md) for details.

## 🚀 Quick Start

### Prerequisites

- Raspberry Pi Zero 2 WH (or better)
- SunFounder PiCar-X robot kit
- Python 3.7+
- Brain server running (see [server README](../server/README.md))

### Installation

```bash
# Clone repository
git clone <repository-url>
cd client_picarx

# Install dependencies
pip install -r requirements.txt

# Configure brain server connection
export BRAIN_HOST=<brain-server-ip>
export BRAIN_PORT=9999
```

### Running

```bash
# Start brainstem
python picarx_robot.py --brain-host $BRAIN_HOST

# Or use a profile (cautious/default/aggressive)
python picarx_robot.py --profile cautious
```

## 📁 Project Structure

```
client_picarx/
├── src/
│   ├── brainstem/
│   │   ├── integrated_brainstem_async.py  # Main conductor
│   │   ├── event_bus.py                   # Async messaging
│   │   ├── nuclei.py                      # Specialized components
│   │   ├── brain_client.py                # TCP protocol
│   │   └── sensor_motor_adapter_fixed.py  # 16→4 mapping
│   ├── config/
│   │   └── brainstem_config.py           # All configuration
│   └── hardware/
│       └── interfaces/                    # Hardware abstraction
├── tests/
│   ├── unit/                              # Component tests
│   ├── behavioral/                        # Emergence tests
│   └── TESTING_STRATEGY.md               # Test philosophy
├── tools/
│   └── profile_brainstem_simple.py       # Performance profiling
└── docs/
    ├── architecture/                      # System design
    └── installation/                      # Setup guides
```

## 🔧 Configuration

All parameters in `src/config/brainstem_config.py`:

```python
# Key settings (no magic numbers!)
min_safe_distance = 0.2      # meters
max_motor_speed = 50.0        # percent
sensor_dimensions = 16        # actual robot sensors
motor_dimensions = 4          # brain output channels
```

Environment variables:
- `BRAIN_HOST` - Brain server IP
- `BRAIN_PORT` - TCP port (default: 9999)
- `ROBOT_PROFILE` - cautious/default/aggressive
- `SAFE_MODE` - Enable safety limits

## 📊 Performance

Measured on development hardware:
- **Brainstem overhead**: < 0.1ms (negligible)
- **Sustainable rate**: 1,456 Hz
- **Production estimate**: ~14,500 Hz (10x faster hardware)

The system is more than fast enough for real-time robotics!

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Behavioral tests (most important!)
python tests/behavioral/field_brain_behavioral_test.py

# Safety-critical tests
python tests/behavioral/safety_critical_tests.py

# Performance profiling
python tools/profile_brainstem_simple.py
```

## 🛡️ Safety Features

Built-in reflexes that work WITHOUT brain:
- Emergency stop (< 5cm obstacles)
- Cliff detection and avoidance
- Battery protection
- Temperature throttling
- Fallback behaviors

## 📚 Documentation

- [Architecture](docs/architecture/BRAINSTEM_ARCHITECTURE.md) - System design
- [Installation Guide](docs/installation/README.md) - Detailed setup
- [Protocol Mapping](PICARX_PROTOCOL_MAPPING.md) - Sensor/motor details
- [Testing Strategy](tests/TESTING_STRATEGY.md) - Test philosophy

## 🚦 Status

**Production Ready!** ✅

Recent achievements:
- Removed reward signals (pure emergence)
- Fixed 16-channel sensor mapping
- Event-driven architecture complete
- Sub-millisecond overhead achieved
- Comprehensive safety system

## 🔮 Future Enhancements

- [ ] Visual cortex nucleus (camera processing)
- [ ] Vocal expression system (emotional sounds)
- [ ] Multi-robot coordination
- [ ] Environmental mapping

## 📝 Philosophy

> "The brainstem doesn't teach the brain what's good or bad - it lets the brain discover meaning through embodied experience."

This system enables true artificial life where:
- Intelligence emerges from simple rules
- Learning happens through experience
- Behaviors surprise us
- The robot develops its own goals

## 🤝 Contributing

We welcome contributions! Key areas:
- New sensory modalities
- Additional safety features
- Performance optimizations
- Behavioral experiments

## 📄 License

[Your License Here]

---

**Version**: 2.0  
**Last Updated**: 2025-01-11  
**Status**: Ready for artificial life! 🚀🧠🤖