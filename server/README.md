# Brain Server

A PureFieldBrain intelligence system - the ultimate synthesis for real intelligence research.
Standardized on pure field dynamics without architectural complexity.

## Quick Start

```bash
# Start the brain server
python3 brain.py

# In another terminal, run the behavioral test
python3 tools/testing/behavioral_test_dynamic.py
```

## Directory Structure

```
server/
├── brain.py                   # Main entry point
├── settings.json              # Configuration file
├── src/                       # Source code
│   ├── core/                  # Core components (new architecture)
│   │   ├── interfaces.py      # Clean interfaces
│   │   ├── robot_registry.py  # Robot profile management
│   │   ├── brain_pool.py      # Brain instance pooling
│   │   ├── brain_service.py   # Session management
│   │   ├── adapters.py        # Robot-brain translation
│   │   ├── connection_handler.py    # Connection orchestration
│   │   ├── dynamic_brain_factory.py # Dynamic brain creation
│   │   ├── monitoring_server.py     # Real-time monitoring
│   │   ├── maintenance_scheduler.py # Automatic maintenance
│   │   ├── logging_service.py       # Centralized logging
│   │   └── error_codes.py           # Error handling
│   ├── brains/                # Brain implementations
│   │   └── field/             # Field-based brains
│   ├── communication/         # Network layer
│   │   └── clean_tcp_server.py     # TCP server implementation
│   ├── persistence/           # Storage layer
│   │   └── persistence_manager.py  # Brain state persistence
│   └── utils/                 # Utilities
│       ├── async_logger.py    # Async logging system
│       └── brain_logger.py    # Brain-specific logging
├── tests/                     # Test suite
│   ├── unit/                  # Unit tests
│   └── integration/           # Integration tests
├── tools/                     # Development tools
│   ├── testing/               # Test frameworks
│   ├── analysis/              # Analysis scripts
│   └── runners/               # Test/demo runners
├── docs/                      # Documentation
├── logs/                      # Log files (generated)
├── robot_memory/              # Persistent brain states
└── archive/                   # Old/deprecated code
```

## Architecture Overview

The PureFieldBrain system creates real intelligence:

1. **Robot connects** with capabilities (sensors, motors)
2. **PureFieldBrain** is instantiated with optimal parameters
3. **Pure field dynamics** process all sensory input through emergent computation
4. **Behavior emerges** from field gradients and energy landscapes
5. **No architectural complexity** - just pure field intelligence

## Key Features

- **Pure Field Intelligence**: Single tensor field with learnable evolution rules
- **GPU-Optimized**: Native CUDA/Metal support for real-time field computation
- **Emergent Behavior**: Intelligence emerges from field dynamics, not programming
- **Biologically-Inspired**: Aggressive parameters and metabolic state modeling
- **Real Intelligence Focus**: No more options or complexity - just the path to AGI
- **Professional Monitoring**: Connect to port 9998 for real-time field metrics
- **Robust Persistence**: PureFieldBrain state management with recovery

## Configuration

Edit `settings.json` to configure:

```json
{
  "brain": {
    "type": "field",
    "spatial_resolution": null,  // Auto-determined based on hardware
    "features": {
      "enhanced_dynamics": true,
      "attention_guidance": true,
      "hierarchical_processing": true
    }
    // Standardized on PureFieldBrain - no brain type options
  },
  "network": {
    "host": "0.0.0.0",
    "port": 9999,
    "monitoring_port": 9998
  },
  "memory": {
    "persistent_memory_path": "./robot_memory",
    "enable_persistence": true
  },
  "logging": {
    "log_directory": "./logs",
    "enable_async_logging": true
  }
}
```

## Monitoring

Connect to the monitoring server:

```bash
telnet localhost 9998
```

Available commands:
- `brain_stats` - Overall brain statistics
- `session_info` - Active session details
- `connection_stats` - Connection information
- `active_brains` - List of active brains
- `performance_metrics` - Performance data

## Development

### Running Tests

```bash
# Run all tests
python3 tools/runners/test_runner.py all

# Run specific test
python3 tests/integration/test_monitoring_server.py
```

### Adding a New Robot Type

1. Create a robot profile JSON in the client directory
2. The system will automatically adapt brain dimensions
3. No code changes needed!

## Migration from Old Architecture

If you're upgrading from the old static brain architecture:

```bash
# Run the migration script
./tools/migrate_to_dynamic.sh
```

This will archive all deprecated files.

## Troubleshooting

### Server won't start
- Check if port 9999 is already in use
- Verify PyTorch is installed: `pip install torch`
- Check logs in `logs/` directory

### Brain creation fails
- Check available memory
- Verify robot capabilities are valid
- Check error stats on monitoring port

### Performance issues
- Monitor cycle times via port 9998
- Check maintenance scheduler status
- Verify GPU acceleration is working

## License

See LICENSE file in project root.