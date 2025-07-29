# Migration Roadmap: Dynamic Brain Architecture

## Overview
This roadmap outlines the steps to fully migrate from the old static brain architecture to the new dynamic architecture, enabling us to deprecate old files.

## Phase 1: Critical Fixes (Priority: HIGH)
### 1.1 Fix Adapter Error ✓
- **Issue**: `'int' object is not iterable` error in adapters
- **Location**: `src/core/adapters.py` or motor output processing
- **Impact**: Errors during every cycle, though processing continues

### 1.2 Fix Dimension Reporting
- **Issue**: Brain reports "NoneD → NoneD" in configuration display
- **Location**: `AdaptiveConfiguration` or brain initialization
- **Impact**: Confusing output, potential configuration issues

## Phase 2: Core Functionality Migration (Priority: HIGH)
### 2.1 Persistence System
- **Current**: `PersistenceManager` in old `BrainFactory`
- **Target**: Add to `BrainService` or `BrainPool`
- **Components**:
  - Brain state serialization
  - Incremental saves
  - Consolidation engine
  - Recovery manager

### 2.2 Experience Storage
- **Current**: `store_experience()` in `BrainFactory`
- **Target**: Add to `BrainSession` or new `ExperienceManager`
- **Components**:
  - Experience tracking per session
  - Learning from outcomes
  - Experience replay capability

## Phase 3: Operational Features (Priority: MEDIUM)
### 3.1 Monitoring Server
- **Current**: `BrainMonitoringServer` on port 9998
- **Target**: Add to `DynamicBrainServer`
- **Components**:
  - Read-only statistics endpoint
  - WebSocket support for real-time data
  - Multi-session monitoring

### 3.2 Logging Infrastructure
- **Current**: `BrainLogger` in old system
- **Target**: Add to new architecture
- **Components**:
  - Session-based logging
  - Performance metrics
  - Debug information

### 3.3 Maintenance Operations
- **Current**: Background maintenance in brain
- **Target**: Add scheduled maintenance
- **Components**:
  - Field maintenance scheduler
  - Memory cleanup
  - Performance optimization

## Phase 4: Server Enhancement (Priority: MEDIUM)
### 4.1 System Information
- **Current**: Hardware info on startup
- **Target**: Add to `DynamicBrainServer`
- **Components**:
  - GPU/CPU detection
  - Memory information
  - Performance recommendations

### 4.2 Error Handling
- **Current**: Comprehensive error classification
- **Target**: Enhance `ConnectionHandler`
- **Components**:
  - Error code system
  - Client-specific error tracking
  - Recovery mechanisms

## Phase 5: Testing & Validation (Priority: HIGH)
### 5.1 Update All Tests
- Convert remaining tests to use new architecture
- Ensure backward compatibility during transition
- Performance benchmarking

### 5.2 Integration Testing
- Multi-robot scenarios
- Persistence/recovery testing
- Long-running stability tests

## Phase 6: Deprecation (Priority: LOW)
### 6.1 File Removal
**After all features migrated:**
- Remove `brain_server.py`
- Remove `src/brain_factory.py`
- Remove `src/communication/tcp_server.py`
- Archive old test files

### 6.2 Documentation Update
- Update all references to old architecture
- Update examples and tutorials
- Update client code samples

## Implementation Order
1. **Week 1**: Phase 1 (Critical Fixes)
2. **Week 2**: Phase 2.1 (Persistence)
3. **Week 3**: Phase 2.2 (Experience) + Phase 3.1 (Monitoring)
4. **Week 4**: Phase 3.2-3.3 (Logging/Maintenance)
5. **Week 5**: Phase 4 (Server Enhancement)
6. **Week 6**: Phase 5 (Testing)
7. **Week 7**: Phase 6 (Deprecation)

## Success Criteria
- [ ] All tests pass with new architecture
- [ ] No functionality regression
- [ ] Performance equal or better
- [ ] Clean deprecation of old files
- [ ] Updated documentation

## Current Status
- ✅ Dynamic brain architecture implemented
- ✅ Basic behavioral test working
- ❌ Critical errors need fixing
- ❌ Core features need migration
- ❌ Old files still in use