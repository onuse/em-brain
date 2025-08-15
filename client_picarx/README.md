# PiCar-X Robot Client

Bare-metal brainstem implementation for the PiCar-X robot that connects to the unified field brain server.

## Key Features

- **Direct Hardware Control**: No SunFounder libraries - pure I2C/GPIO via smbus
- **High-Resolution Vision**: 640×480 minimum (307,200 pixels) - real vision
- **Full Sensor Suite**: Camera, microphone, ultrasonic, grayscale, battery
- **Safety Reflexes**: Hardware reflexes work even without brain
- **Dynamic Adaptation**: Brain auto-adapts to any resolution via handshake

## Quick Start

```bash
# Deploy to Raspberry Pi
export PI_HOST=pi@192.168.1.231
./deploy.sh

# On the Pi
cd ~/picarx_robot
sudo python3 picarx_robot.py --brain-host <SERVER_IP>
```

## Architecture

```
PiCar-X Hardware
    ↓
Bare Metal HAL (I2C @ 0x40, GPIO pins)
    ↓
Brainstem (safety reflexes + sensor fusion)
    ↓
Brain Client (TCP binary protocol)
    ↓
Brain Server (307,200 inputs → 64 channels)
```

## Configuration

Edit `config/robot_config.json`:
```json
{
  "vision": {
    "resolution": [640, 480],  // Or [1280,720], [1920,1080]
    "format": "grayscale"      // Or "rgb" for 3× bandwidth
  },
  "brain": {
    "host": "192.168.1.100",   // Your brain server IP
    "port": 9999
  }
}
```

## Hardware Mapping

- **PCA9685 @ 0x40**: All servos and motor PWM
  - Channel 0: Camera pan (±90°)
  - Channel 1: Camera tilt (-35° to +65°)
  - Channel 2: Steering (±30°)
  - Channel 4: Left motor PWM
  - Channel 5: Right motor PWM
- **GPIO 23/24**: Motor direction pins
- **I2C ADC**: Grayscale sensors, battery voltage
- **Camera**: PiCamera2 (640×480+ resolution)
- **Audio**: I2S microphone/speaker

## Vision Philosophy

We send **full resolution** (640×480 minimum) to force intelligence:
- 307,200 pixels → Brain compresses to 64 channels (4,800:1 ratio)
- Brain learns feature extraction, not pixel memorization
- Computational pressure drives emergence of visual intelligence
- See [`docs/BRAIN_HIGH_BANDWIDTH_ANALYSIS.md`](docs/BRAIN_HIGH_BANDWIDTH_ANALYSIS.md)

## Safety Features

Brainstem reflexes (work without brain):
- Collision stop: Ultrasonic < 10cm
- Cliff detection: Grayscale threshold
- Battery protection: Warning 6.5V, critical 6.0V
- Emergency stop: On disconnect
- Safe mode: Limited speed for testing

## Network Protocol

Binary TCP on port 9999:
- **Handshake**: Dynamic dimension negotiation
- **Input**: 307,200+ float32 values (vision + sensors)
- **Output**: 6 float32 motor commands
- **Bandwidth**: ~25 MB/s at 640×480, 20Hz

## Documentation

- [`docs/BRAIN_HIGH_BANDWIDTH_ANALYSIS.md`](docs/BRAIN_HIGH_BANDWIDTH_ANALYSIS.md) - Brain's vision handling
- [`docs/HIGH_RESOLUTION_VISION.md`](docs/HIGH_RESOLUTION_VISION.md) - Vision configuration
- [`docs/BARE_METAL_COMPARISON.md`](docs/BARE_METAL_COMPARISON.md) - Hardware details
- [`docs/QUICK_REFERENCE.md`](docs/QUICK_REFERENCE.md) - Command reference

## Testing

```bash
# Test without brain (reflexes only)
sudo python3 picarx_robot.py --test-mode

# Test vision module
python3 -m src.hardware.configurable_vision --test

# Full integration test
sudo python3 picarx_robot.py --brain-host 192.168.1.100
```

---

**"Test the infrastructure, let the intelligence emerge."**