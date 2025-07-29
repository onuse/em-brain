# Performance Metrics

## Gradient Computation
- Base computation time: ~1000ms
- Optimized (local region): ~20ms
- Speedup factor: 50x

## Processing Cycle Time
- Base cycle time: ~1700ms
- Optimized cycle time: ~32ms (at resolution 3³)
- Speedup factor: 53x

## Frequency by Resolution
- Resolution 3³: 30-35 Hz
- Resolution 4³: 20-25 Hz
- Resolution 5³: 14-16 Hz

## Behavioral Responses
- Motor output range: 0.006 → 0.3-1.0 (normalized)
- Turn response magnitude: 0.018 → 0.48
- Obstacle avoidance: Functional at resolution 5³

## Memory Usage
- Resolution 3³: ~9 MB (2.3M elements)
- Resolution 4³: ~21 MB (5.5M elements)
- Resolution 5³: ~41 MB (10.8M elements)

## Hardware Tiers
- High performance (≤20ms benchmark): Resolution 5³
- Medium performance (≤40ms benchmark): Resolution 4³
- Low performance (>40ms benchmark): Resolution 3³