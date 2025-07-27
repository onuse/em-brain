# TODO: Field-Native Intelligence System

## Current Focus: Real-World Testing

The brain now has blended reality - seamlessly mixing spontaneous dynamics (fantasy) with sensory input (reality) based on prediction confidence! Ready for hardware testing.

### Immediate Next Steps
- [ ] **CRITICAL**: Restore persistence system for cross-session learning
- [ ] **CRITICAL**: Update picarx_brainstem.py for hardware deployment  
- [ ] Test blended reality with actual robot behavior patterns
- [ ] Evaluate attention system for enhanced sensor processing
- [ ] Clean up 44 orphaned files (55% of codebase)
- [ ] Deploy brain server with actual PiCar-X hardware
- [ ] Test and calibrate sensor normalization ranges
- [ ] Validate motor command scaling on real motors
- [ ] Test network latency between brain server and robot

### High Priority
- [ ] Verify ultrasonic sensor range mapping (currently assumes 0-4m)
- [ ] Test safety reflexes with real sensor data
- [ ] Implement connection retry logic for network failures
- [ ] Add performance monitoring for Raspberry Pi Zero 2

### Medium Priority
- [ ] Tune reward signal parameters based on robot behavior
- [ ] Optimize motor smoothing factor
- [ ] Implement secure API authentication
- [ ] Add real-time field visualization dashboard

### Low Priority
- [ ] Document sensor-to-brain mapping algorithm
- [ ] Create performance benchmarks for different surfaces
- [ ] Add support for multiple robot connections
- [ ] Implement predictive maintenance alerts

### Completed Recently
- [x] **Code inventory analysis** - identified critical gaps!
  - [x] Analyzed all 80 files in server/src
  - [x] Found only 17.5% actively used (14 files)
  - [x] Discovered 55% orphaned (44 files)
  - [x] Identified lost persistence system (no learning survives restart!)
  - [x] Found missing hardware integration (can't deploy to robot!)
  - [x] Created detailed inventory and recovery plan
- [x] **Implemented Blended Reality System** - fantasy and reality seamlessly mix!
  - [x] Confidence-based blending: high confidence → more fantasy
  - [x] Variable sensory imprint strength (weak when confident)
  - [x] Weighted spontaneous activity (not just gated)
  - [x] Smooth confidence transitions (temporal smoothing)
  - [x] Dream mode after 100 cycles idle (95% fantasy)
  - [x] Fixed inverted confidence (novel=low, known=high)
  - [x] Fixed dream detection (variance from neutral, not magnitude)
- [x] **Implemented spontaneous field dynamics** - the brain thinks without input!
  - [x] Traveling waves create coherent internal patterns
  - [x] Local recurrence maintains activity
  - [x] Homeostatic drive balances field energy
  - [x] Critical dynamics at edge of chaos
  - [x] Generates motor commands autonomously
  - [x] Develops preferences through experience
- [x] **Confidence-based sensory processing** - brain decides when to check sensors!
  - [x] Integrated CognitiveAutopilot with spontaneous dynamics
  - [x] Stochastic sensor checking based on prediction confidence
  - [x] AUTOPILOT mode: 20% sensor attention
  - [x] FOCUSED mode: 50% sensor attention  
  - [x] DEEP_THINK mode: 90% sensor attention
  - [x] Sensors suppressed, never ignored (max 80% suppression)
- [x] Fixed motor generation (gradient key mismatch bug)
- [x] Fixed field activation strength (was too weak)
- [x] Fixed sensory-to-field coordinate mapping
- [x] Implemented complete brainstem layer with:
  - [x] 16→24 sensor channel expansion
  - [x] 4→5 motor channel mapping
  - [x] Reward signal generation
  - [x] Safety reflexes and fallback behaviors
- [x] Validated 40ms cycle time (suitable for 25Hz control)
- [x] Created integrated brainstem with HTTP communication
- [x] Added deployment checklist documentation

## Key Architecture Decisions

1. **Dynamic Brain Creation**: Brains created on-demand with logarithmic dimension scaling
2. **Robot Profiles**: JSON-based profiles define robot capabilities
3. **Brain Pooling**: Reuse brain instances for efficiency
4. **Adapter Pattern**: Clean translation between robot and brain spaces
5. **Maintenance Automation**: Background scheduler keeps brains healthy

## Next Major Milestone

Deploy and test the brain server with actual PiCar-X hardware to validate real-world performance and latency characteristics.