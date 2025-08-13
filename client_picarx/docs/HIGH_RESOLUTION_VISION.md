# High Resolution Vision Implementation

## ✅ 640×480 Minimum Resolution

Based on your insight, we're now using **640×480 (307,200 pixels)** as the bare minimum for real vision. This is configured in `config/robot_config.json`.

## How The Brain Handles This

### Automatic Adaptation
The PureFieldBrain automatically handles ANY input dimension:

```python
# In pure_field_brain.py:
if actual_input_dim != self.input_dim:
    # Creates new projection layer on the fly!
    self._dynamic_input_projection = nn.Linear(
        actual_input_dim,  # 307,200 for 640×480
        first_channels,     # Brain's internal channels
        device='cuda'
    )
```

### What Happens:
1. **Brainstem sends 307,200 values** (640×480 grayscale)
2. **Brain detects new dimension** via handshake
3. **Creates GPU-accelerated projection** (307,200 → 64 channels)
4. **Projects into 3D neural field** for spatial processing

### Performance Impact:
```
307,200 inputs × 64 channels = 19.7M parameters just for input!
```
This FORCES the brain to:
- Develop compression strategies
- Learn spatial attention
- Extract relevant features
- Create hierarchical representations

## Configuration Options

In `config/robot_config.json`:

```json
{
  "vision": {
    "enabled": true,
    "resolution": [640, 480],  // Change this!
    "fps": 30,
    "format": "grayscale"     // or "rgb" for 3× data
  }
}
```

### Resolution Presets:
- `[640, 480]` - **307,200 pixels** (recommended minimum)
- `[800, 600]` - **480,000 pixels**
- `[1024, 768]` - **786,432 pixels**  
- `[1280, 720]` - **921,600 pixels** (HD)
- `[1920, 1080]` - **2,073,600 pixels** (Full HD!)

### Format Options:
- `"grayscale"` - 1 value per pixel
- `"rgb"` - 3 values per pixel (3× bandwidth!)

## Network Bandwidth

At 640×480 grayscale, 20Hz:
```
307,200 pixels × 4 bytes × 20 Hz = 24.6 MB/s
```

At 1920×1080 RGB, 20Hz:
```
2,073,600 × 3 × 4 × 20 = 497.7 MB/s (!!)
```

Local gigabit network can handle this easily.

## Why This Matters

### Biological Inspiration:
- Human retina: 126 million photoreceptors
- Mouse retina: 4-5 million photoreceptors
- Even insects: 10,000+ ommatidia

We were giving the brain **9 pixels** (3×3 grid). That's not vision, that's a light sensor!

### Computational Pressure:
With 307,200 inputs, the brain CANNOT:
- Memorize all patterns
- Use simple lookup tables
- Brute force solutions

It MUST develop:
- Feature extraction
- Spatial pooling
- Attention mechanisms
- Hierarchical processing

## Implementation Files

### Core Vision Module:
`src/hardware/configurable_vision.py`
- Reads resolution from config
- Handles any resolution
- Efficient frame capture

### Configuration:
`config/robot_config.json`
- Set resolution here
- No code changes needed

### Integration:
```python
# In bare_metal_hal.py
from hardware.configurable_vision import ConfigurableVision

def _init_vision_audio(self):
    self.vision = ConfigurableVision()
    # Automatically uses config resolution
    
def read_raw_sensors(self):
    # Get full resolution frame
    vision_data = self.vision.get_flattened_frame()
    # This is 307,200 values at 640×480!
```

## Testing Different Resolutions

1. **Start "small"** (but not too small):
   ```json
   "resolution": [640, 480]
   ```

2. **Monitor brain performance**:
   ```python
   # Brain will report processing time
   # If < 50ms, can go higher resolution
   ```

3. **Gradually increase**:
   ```json
   "resolution": [1280, 720]  // HD
   ```

4. **Ultimate test**:
   ```json
   "resolution": [1920, 1080]  // Full HD
   ```

## The Bottom Line

You were absolutely right:
- 64×64 was too conservative
- 640×480 should be the MINIMUM
- The brain can and should handle real vision data
- High bandwidth creates pressure for intelligence

The implementation now supports this fully, with configuration-driven resolution that the brain automatically adapts to!