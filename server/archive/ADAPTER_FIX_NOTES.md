# Adapter Dimension Fix for SimplifiedUnifiedBrain

## Issue
The biological_embodied_learning experiment was failing with:
```
❌ Session session_70e76939 processing error: Expected 5D field state, got 3D
```

## Root Cause
- SimplifiedUnifiedBrain outputs 3 motor commands directly
- The regular MotorAdapter expects field_dimensions (5D) not motor commands (3D)
- Wrong adapter factory was being used

## Fix Applied

1. **Created SimplifiedAdapterFactory** in `src/core/simplified_adapters.py`
   - SimplifiedMotorAdapter handles tensor/list inputs
   - Automatically adds confidence value if missing (3D → 4D)

2. **Updated brain.py** to use SimplifiedAdapterFactory when SimplifiedBrainFactory is used:
   ```python
   if isinstance(self.brain_factory, SimplifiedBrainFactory):
       self.adapter_factory = SimplifiedAdapterFactory()
   ```

3. **SimplifiedMotorAdapter** now:
   - Accepts both tensor and list inputs
   - Adds default confidence (0.5) when brain outputs 3 motors but robot expects 4
   - Validates all values are in [-1, 1] range

## Result
✅ The 8-hour biological_embodied_learning test can now run successfully
✅ Brain outputs 3 motors, adapter converts to 4 for robot compatibility
✅ No dimension mismatch errors

## Testing
Run: `python3 tools/analysis/test_adapter_fix.py` to verify the fix