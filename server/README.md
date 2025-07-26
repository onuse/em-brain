# Brain Server

A field-native intelligence system that creates adaptive brains on-demand based on robot capabilities.

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

The Dynamic Brain Architecture creates brains on-demand:

1. **Robot connects** with capabilities (sensors, motors)
2. **System calculates** optimal brain dimensions
3. **Brain is created** with field dimensions matching robot complexity
4. **Session manages** the robot-brain interaction
5. **Adapters translate** between robot and brain spaces

## Key Features

- **Dynamic Dimensioning**: Brains adapt to robot capabilities
- **Resource Efficient**: Only creates brains when needed
- **Real-time Monitoring**: Connect to port 9998 for stats
- **Automatic Maintenance**: Keeps brains healthy over time
- **Robust Persistence**: Incremental saves with recovery
- **Professional Logging**: Async logging with session tracking
- **Error Handling**: Standardized error codes and tracking

## Configuration

Edit `settings.json` to configure:

```json
{
  "brain": {
    "type": "field",
    "spatial_resolution": null,  // Auto-determined
    "features": {
      "enhanced_dynamics": true,
      "attention_guidance": true,
      "hierarchical_processing": true
    }
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