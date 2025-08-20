# Brain Persistence

The Enhanced Critical Mass Brain automatically persists its complete state, allowing learned knowledge to accumulate over time. The brain remembers everything it has learned between sessions.

## Default Behavior

**By default, the brain:**
- Auto-saves every 5 minutes
- Saves when disconnecting
- Automatically loads previous state on startup
- Uses the Enhanced brain with speed optimization

Just run:
```bash
python3 run_server.py
```

The brain will automatically continue from where it left off!

## Features

### Starting Fresh
To start with a completely new brain (ignoring saved state):

```bash
# Start fresh
python3 run_server.py --fresh-brain
```

### Custom Save Intervals
Change the auto-save frequency:

```bash
# Save every 10 minutes instead of 5
python3 run_server.py --save-interval 600

# Disable auto-save
python3 run_server.py --save-interval 0
```

### Disable Save on Exit
If you don't want to save when disconnecting:

```bash
# Don't save on exit
python3 run_server.py --no-save-on-exit
```

### Load Specific State
Load a particular brain state file:

```bash
# Load a specific saved state
python3 run_server.py --load-state brain_states/my_experiment.brain

# Load from backup
python3 run_server.py --load-state brain_states/enhanced_20241220_143022_exit.brain
```

## What Gets Saved

The complete brain state is preserved:

- **Field Dynamics**: The 4D tensor field and momentum
- **Learned Concepts**: All resonance patterns and their couplings
- **Causal Knowledge**: Temporal chains between patterns
- **Semantic Meanings**: Pattern-to-outcome mappings
- **Success Rates**: Which behaviors work
- **Preferences**: Learned preferences and goals
- **Memory**: Holographic memory storage
- **Metrics**: All learning metrics and progress

## File Management

Brain states are saved as compressed files in the `brain_states/` directory:

- `enhanced_autosave.brain` - Default auto-save location
- `enhanced_autosave_exit.brain` - Saved on disconnect
- `enhanced_YYYYMMDD_HHMMSS_exit.brain` - Timestamped exit saves

Files are compressed with gzip and typically range from 50-200 MB depending on field size.

## Usage Examples

### Default Usage (Recommended)
```bash
# Just run it - everything is handled automatically
python3 run_server.py

# The brain will:
# - Load previous state if it exists
# - Auto-save every 5 minutes
# - Save when you disconnect
# - Use optimal settings (enhanced brain, speed target)
```

### Long Training Session
```bash
# More frequent saves for important training
python3 run_server.py --save-interval 60

# Later, it automatically continues from where you left off
python3 run_server.py
```

### Experiment Branches
```bash
# Save a baseline after initial training
python3 brain_manager.py save --name baseline

# Try different approaches
python3 run_server.py --brain enhanced --load-state brain_states/baseline.brain --save-on-exit

# Compare results
python3 brain_manager.py list
```

### Recovery from Crashes
If the server crashes, you can recover from the most recent auto-save:

```bash
# List available saves
python3 brain_manager.py list

# Load the most recent
python3 run_server.py --brain enhanced --load-state brain_states/enhanced_autosave.brain
```

## Performance Impact

- **Auto-save**: Takes 1-3 seconds depending on brain size
- **Loading**: Takes 2-5 seconds on startup
- **Storage**: Each save is 50-200 MB compressed

Auto-save runs in the main thread but is fast enough not to disrupt operation. For real-time applications, use longer save intervals (600+ seconds).

## Brain Manager Utility

Use the brain manager for manual saves:

```python
# Save current brain
python3 brain_manager.py save --name my_experiment

# Load a saved brain
python3 brain_manager.py load --file brain_states/my_experiment.brain

# List all saved brains
python3 brain_manager.py list
```

## Tips

1. **Regular Saves**: Use `--save-interval 300` for automatic 5-minute saves
2. **Exit Saves**: Always use `--save-on-exit` to prevent losing progress
3. **Experiments**: Save named checkpoints before trying new approaches
4. **Recovery**: Keep multiple saves for rollback options
5. **Storage**: Old saves can be deleted to free space - they're just files

## Architecture Notes

The persistence system uses:
- Python pickle for serialization
- gzip compression for storage efficiency
- Atomic writes to prevent corruption
- Versioning support for future compatibility

The save/load methods are implemented in:
- `enhanced_critical_mass_brain.py`: Core save_state() and load_state() methods
- `simple_brain_service.py`: Auto-save integration
- `brain_manager.py`: Manual save/load utility