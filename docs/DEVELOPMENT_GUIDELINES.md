# Development Guidelines

## Demo Management

### ⚠️ CRITICAL RULE: Never Create New Demo Files for Fixes

**ALWAYS update the existing canonical demo files instead of creating new ones.**

### Wrong Approach ❌
```bash
# DON'T DO THIS
demo_ultimate_2d_brain.py          # Original
demo_ultimate_2d_brain_fixed.py    # Bug fix
demo_ultimate_2d_brain_v2.py       # Feature addition
demo_ultimate_longer_life.py       # Parameter adjustment
```

### Correct Approach ✅
```bash
# DO THIS
demo_ultimate_2d_brain.py          # Always update this same file
```

### Why This Matters

**Creating new demo files for every change:**
- ❌ Fragments the user experience
- ❌ Creates confusion about which is "current"
- ❌ Defeats the purpose of consolidation
- ❌ Makes documentation outdated immediately
- ❌ Multiplies maintenance burden

**Updating the canonical demo:**
- ✅ Single source of truth
- ✅ Users always get latest improvements
- ✅ Documentation stays accurate
- ✅ Easier maintenance and testing
- ✅ Clear progression of capabilities

## Demo File Responsibilities

### `demo_ultimate_2d_brain.py` - THE Canonical 2D Demo
**This file should ALWAYS represent the current state-of-the-art 2D brain system.**

**When to update this file:**
- Bug fixes (survival rates, display issues, crashes)
- Feature additions (new drives, improved algorithms)
- Performance improvements (optimization, better parameters)
- User experience improvements (better visualization, controls)
- Integration of new brain capabilities

**What this file should always include:**
- All latest brain features integrated
- Balanced parameters for good demonstration
- Optimal window sizing for most screens
- Clear console output explaining what's happening
- Proper error handling and graceful degradation

### Other Demo Files
- `demo_world_agnostic_brain.py` - Update when universality features change
- `demo_distributed_brain.py` - Update when distributed architecture changes
- `demo_phase1.py` - Update when core architecture changes

## Version Control Best Practices

### Commit Messages for Demo Updates
```bash
# Good commit messages
git commit -m "Fix survival rates in ultimate 2D demo for better learning"
git commit -m "Add goal generation integration to ultimate 2D demo"
git commit -m "Improve window sizing in ultimate 2D demo"

# Bad commit messages  
git commit -m "Create new longer life demo"
git commit -m "Add demo_ultimate_2d_brain_v2.py"
```

### Development Workflow
1. **Identify issue** in canonical demo
2. **Fix directly** in canonical demo file
3. **Test thoroughly** to ensure no regressions
4. **Update documentation** if behavior changes significantly
5. **Commit with clear message** describing the improvement

## Parameter Tuning Guidelines

### Current Balanced Parameters (as of latest update)
```python
# Survival rates (in grid_world.py)
RED_SQUARE_DAMAGE = 0.02    # 2% per step (was 10%)
ENERGY_DECAY = 0.0002       # 5000 steps to starve (was 1000)
STARVATION_DAMAGE = 0.001   # When energy = 0 (was 0.005)

# Display settings (in demo files)
WORLD_SIZE = 12x12          # Fits most screens (was 15x15)
CELL_SIZE = 25              # Good balance (was 30)
STEP_DELAY = 0.3            # Observable learning (was 0.25)
```

### When to Adjust Parameters
- **Robot dies too quickly**: Reduce damage rates
- **Robot lives too long**: Increase damage rates  
- **Learning too slow**: Increase exploration incentives
- **Behavior too chaotic**: Increase survival weight
- **Display issues**: Adjust world size or cell size

### How to Test Parameter Changes
1. Run demo for 5+ minutes to observe full learning cycle
2. Check robot survives long enough to show intelligent behavior
3. Verify brain panel is visible and functional
4. Ensure console output is helpful and informative

## Feature Integration

### When Adding New Brain Features
1. **Integrate into ultimate demo first** - never create separate demos
2. **Maintain backward compatibility** - old saved sessions should still work
3. **Add console output** explaining new capabilities
4. **Update DEMOS.md** to mention new features
5. **Test extensively** before committing

### Example: Adding Goal Generation
```python
# Don't create demo_goal_generation.py
# Instead, update demo_ultimate_2d_brain.py to include goal generation
```

## Documentation Maintenance

### When Demo Behavior Changes
- Update DEMOS.md with new capability descriptions
- Update troubleshooting sections if new issues arise
- Update performance expectations if learning speeds change
- Keep README accurate with current demo capabilities

### User Experience Consistency
- Console output should always be informative and encouraging
- Error messages should be helpful, not technical
- Window sizing should work on most common screen resolutions
- Controls should be intuitive and well-documented

## Emergency Procedures

### If Ultimate Demo Breaks
1. **Fix immediately** - this is the primary user entry point
2. **Don't create workaround demos** - fix the root cause
3. **Test thoroughly** before pushing fixes
4. **Document what was broken** for future prevention

### If Major Refactoring Needed
1. **Create feature branch** for development
2. **Update ultimate demo** in feature branch
3. **Test extensively** before merging to main
4. **Update all documentation** in same commit
5. **Notify users** of any breaking changes

---

**Remember: The ultimate demo is the face of the project. It should always represent our best, most impressive capabilities working together seamlessly.**