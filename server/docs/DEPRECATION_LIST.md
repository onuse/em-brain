# Files to Deprecate - Dynamic Brain Architecture Migration

## Overview
These files are part of the old static brain architecture and can be safely removed after verifying all functionality has been migrated to the new dynamic architecture.

## Files to Remove

### 1. Main Server Files
- **`brain_server.py`** - Old static brain server (replaced by `dynamic_brain_server.py`)
- **`src/brain_factory.py`** - Old static brain factory (replaced by `src/core/dynamic_brain_factory.py`)

### 2. Communication Files  
- **`src/communication/tcp_server.py`** - Old TCP server (replaced by `src/communication/clean_tcp_server.py`)
- **`src/communication/monitoring_server.py`** - Old monitoring server (replaced by `src/core/monitoring_server.py`)

### 3. Test Files (Old Architecture)
- **`tests/integration/test_brain_server.py`** - Tests for old server
- **`tools/testing/behavioral_test.py`** - Old behavioral test (replaced by `behavioral_test_dynamic.py`)
- **`tools/testing/behavioral_test_fast.py`** - Uses old BrainFactory
- **`tools/testing/behavioral_test_framework.py`** - Uses old BrainFactory
- **`tools/testing/behavioral_test_single_cycle.py`** - Uses old BrainFactory
- **`tools/testing/capacity_test.py`** - If it uses old BrainFactory
- **`tools/testing/network_test.py`** - If it uses old server
- **All files in `tools/analysis/`** - Most use old BrainFactory and need updating

### 4. Old Brain Implementations
- Any brain files that don't support dynamic dimensions
- Old adapter implementations that don't implement the new interfaces

## Verification Checklist

Before removing these files, ensure:

- [ ] All functionality from `brain_server.py` is in `dynamic_brain_server.py`
- [ ] All functionality from `brain_factory.py` is in `dynamic_brain_factory.py`
- [ ] All tests have been updated to use new architecture
- [ ] No remaining imports of these files exist
- [ ] Client code has been updated to work with new server
- [ ] Monitoring functionality is fully migrated
- [ ] Error handling is improved in new implementation

## Migration Status

### Completed Migrations ✅
1. Dynamic brain creation based on robot capabilities
2. Persistence system with incremental saves
3. Experience storage and tracking
4. Monitoring server with JSON endpoints
5. Logging infrastructure with async support
6. Maintenance scheduler for field health
7. Enhanced error handling with error codes
8. System information display on startup

### Not Yet Migrated ❌
None - all core functionality has been migrated!

## Recommended Approach

1. First, search for any remaining imports of deprecated files:
   ```bash
   grep -r "from src.brain_factory import" --include="*.py"
   grep -r "import brain_server" --include="*.py"
   grep -r "from src.communication.tcp_server import" --include="*.py"
   ```

2. Move deprecated files to archive:
   ```bash
   mkdir -p archive/old_architecture
   mv brain_server.py archive/old_architecture/
   mv src/brain_factory.py archive/old_architecture/
   mv src/communication/tcp_server.py archive/old_architecture/
   ```

3. Run all tests to ensure nothing breaks:
   ```bash
   python3 tools/runners/test_runner.py all
   python3 tools/testing/behavioral_test_dynamic.py
   ```

4. If all tests pass, the files can be permanently removed.

## Notes

- The new dynamic architecture is superior in every way:
  - Brains adapt to robot capabilities
  - Better resource utilization
  - Cleaner separation of concerns
  - Enhanced error handling
  - Better monitoring and logging
  
- Keep `src/communication/clean_tcp_server.py` as it's the new implementation
- Keep all files in `src/core/` as they're part of the new architecture