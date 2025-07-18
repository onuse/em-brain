# Brain Project Organization Plan

## Current Problems

### 1. **Log File Chaos**
- `logs/` directory: 200+ brain session files
- `server/logs/`: More brain session files
- `server/tests/logs/`: Even more brain session files  
- `validation/logs/`: Additional brain errors
- No log rotation or cleanup

### 2. **Result File Scatter**
- `validation/scaling_results_*/`: 15+ directories
- `validation/embodied_learning/reports/`: Multiple subdirectories
- `validation/camera_prediction_results/`: Mixed JSON/MD files
- `validation/emergence_results/`: More JSON files
- No unified result structure

### 3. **Root Folder Clutter**
- Multiple isolated experiment files
- Screenshot images at root level
- Cleanup summary files
- Mixed test/demo/validation files

### 4. **Memory Checkpoints Duplication**  
- `robot_memory/checkpoints/`: 10 files
- `server/robot_memory/checkpoints/`: 10 more files
- Unclear which is canonical

## Proposed Organization

### **Root Directory Structure**
```
brain/
├── README.md                    # Main project documentation
├── CLAUDE.md                    # Development instructions
├── requirements.txt             # Top-level dependencies
├── .gitignore                   # Ignore patterns
│
├── src/                         # Main brain implementation
│   ├── brain.py                 # Core brain coordinator
│   ├── vector_stream/           # Vector stream architecture  
│   ├── utils/                   # Shared utilities
│   └── communication/           # Client-server communication
│
├── tests/                       # All tests (unit + integration)
│   ├── unit/                    # Unit tests
│   ├── integration/             # Integration tests
│   └── fixtures/                # Test data/fixtures
│
├── validation/                  # Scientific validation studies
│   ├── experiments/             # Experiment scripts
│   ├── environments/            # Test environments
│   └── results/                 # All validation results
│       ├── embodied_learning/   # Embodied learning results
│       ├── scaling/             # Performance scaling results
│       ├── emergence/           # Emergence validation results
│       └── camera/              # Vision system results
│
├── demos/                       # Demonstration scripts
│   ├── spatial_learning/        # Spatial demos
│   ├── picar_x/                 # Robot-specific demos
│   └── visualization/           # Brain visualization demos
│
├── docs/                        # Documentation
│   ├── architecture/            # Technical architecture docs
│   ├── protocols/               # Communication protocols
│   └── guides/                  # User guides
│
├── server/                      # Brain server (for robot clients)
│   ├── brain_server.py          # TCP server entry point
│   ├── settings.json            # Server configuration
│   └── logs/                    # Server-specific logs only
│
├── client_picarx/               # PiCar-X robot client
│   └── (existing structure)
│
├── archive/                     # Historical/deprecated code
│   └── (existing structure)
│
├── logs/                        # Centralized logging
│   ├── brain_sessions/          # Brain session logs
│   ├── experiments/             # Experiment logs
│   ├── errors/                  # Error logs
│   └── archive/                 # Old logs (auto-archived)
│
└── tools/                       # Development/analysis tools
    ├── cleanup/                 # Project maintenance scripts
    ├── analysis/                # Performance analysis tools
    └── runners/                 # Test/demo/validation runners
```

### **Logging Consolidation**
- **Single `logs/` directory** at root
- **Automatic log rotation** (keep last 50 sessions)
- **Categorized subdirectories** by log type
- **Archive old logs** automatically

### **Results Unification**
- **Single `validation/results/` tree** for all validation outputs
- **Consistent naming** with timestamps and experiment types
- **Automatic cleanup** of old results (keep last 20 per experiment type)
- **Summary generation** for result directories

### **Root Folder Cleanup**
- Move experiment files to appropriate directories
- Remove screenshot clutter (move to `validation/results/`)
- Consolidate test runners into `tools/runners/`
- Keep only essential files at root

## Implementation Steps

### Phase 1: Create New Structure
1. Create new directory structure
2. Set up logging configuration
3. Create result organization system

### Phase 2: Move Files
1. Migrate source code to `src/`
2. Consolidate all logs to `logs/`
3. Organize all results under `validation/results/`
4. Move tools to `tools/`

### Phase 3: Update References
1. Update import paths
2. Fix logging destinations
3. Update runner scripts
4. Update documentation

### Phase 4: Cleanup
1. Remove duplicated files
2. Archive old logs/results
3. Clean root directory
4. Update .gitignore

## Benefits

### **Developer Experience**
- Clear separation of concerns
- Predictable file locations
- Easier navigation
- Reduced cognitive overhead

### **Maintenance**
- Automatic log rotation
- Result archiving
- Duplicate detection
- Space management

### **Collaboration**
- Consistent organization
- Clear documentation structure
- Standardized result formats
- Easy onboarding

## Next Steps

1. **Get approval** for organization plan
2. **Implement Phase 1** (create structure)
3. **Migrate files** systematically  
4. **Update tooling** to use new structure
5. **Document** new organization

This will transform the project from chaotic file scatter to a professional, maintainable structure.