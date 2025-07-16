# Experience-Based Architecture Archive

This directory contains the archived implementation of the experience-based brain architecture that was replaced by vector streams in 2024.

## Why This Code Was Archived

Based on scientific comparison testing, the vector stream architecture showed superior performance:
- **2.9x better dead reckoning** (59% vs 20% confidence)
- **Superior timing adaptation** across variable timing scenarios  
- **5/6 scenario wins** in head-to-head comparison
- **More biologically realistic** continuous processing

See `docs/architecture_evolution.md` for complete analysis.

## Archived Components

### Core Experience Infrastructure
- `experience/storage.py` - Experience storage with discrete objects
- `experience/working_memory.py` - Working memory buffer for experiences
- `experience/memory_consolidation.py` - Asynchronous experience consolidation
- `experience/experience.py` - Experience data model

### Prediction Systems
- `adaptive_engine.py` - Experience-based prediction engine

### Test Files
- `test_*dead_reckoning.py` - Experience-based dead reckoning tests
- `test_*predictive_streaming.py` - Experience streaming tests  
- `test_*dual_memory_brain.py` - Dual memory architecture tests

## What Replaced This Code

The experience-based architecture was replaced by:
- **Vector Stream Brain** (`server/src/vector_stream/minimal_brain.py`)
- **Continuous vector processing** instead of discrete experience packages
- **Time-as-data-stream** approach with organic metronome
- **Modular streams** (sensory, motor, temporal) with cross-stream learning

## If You Need To Use This Code

This code is kept for:
1. **Historical reference** - understanding the evolution of our architecture
2. **Emergency rollback** - if vector streams encounter critical issues
3. **Scientific comparison** - validating architectural decisions
4. **Learning purposes** - seeing what we tried before vector streams

To use this code:
1. Copy components back to their original locations
2. Update imports throughout the codebase
3. Ensure test compatibility with current infrastructure

## Key Lessons Learned

From the experience-based approach:
1. **Working memory concept** - recent experiences must participate in reasoning
2. **Dual memory search** - both immediate and long-term memory matter
3. **Asynchronous consolidation** - don't block actions on memory operations
4. **Prediction streaming** - predictions should become part of working memory

These concepts were successfully carried forward to the vector stream architecture.

## Scientific Evidence

The comparison testing that led to this archival:

| Scenario | Experience Confidence | Vector Confidence | Improvement |
|----------|----------------------|-------------------|-------------|
| Dead Reckoning | 0.20 | 0.59 | **2.9x** |
| Variable Timing | 0.20 | 0.32 | 1.6x |
| Training | 0.20 | 0.30 | 1.5x |
| Slow Rhythm | 0.20 | 0.26 | 1.3x |
| Latency Spike | 0.20 | 0.22 | 1.1x |

Overall: Vector streams won 5/6 scenarios with dramatically superior dead reckoning.

---

*Archived: 2024*  
*Replaced by: Vector Stream Architecture*  
*Reason: Scientific evidence of superior biological realism and performance*