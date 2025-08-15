# PiCar-X Installation Guide

## ✅ NO SunFounder Software Required!

This implementation controls the PiCar-X hardware directly through standard Raspberry Pi interfaces. We do **NOT** need:
- ❌ SunFounder robot-hat library
- ❌ SunFounder picarx library  
- ❌ Adafruit servo libraries
- ❌ Any third-party motor control libraries

## What We Use Instead

### Standard Raspberry Pi Libraries (Pre-installed)
- **RPi.GPIO** - Direct GPIO pin control (comes with Raspbian)
- **smbus** - I2C communication for PCA9685 and ADC (comes with Raspbian)
- **Python 3** - Standard library only

### Our Approach
We control the PCA9685 PWM chip directly via I2C registers:
- Write directly to PCA9685 registers for servo/motor control
- Read directly from ADC via I2C for sensor data
- Use GPIO pins directly for motor direction and ultrasonic

## Prerequisites on Raspberry Pi

### 1. Enable I2C
```bash
sudo raspi-config
# Navigate to: Interface Options → I2C → Enable
# Reboot after enabling
```

### 2. Verify I2C Devices
```bash
sudo i2cdetect -y 1
# Should show:
#   0x14 - ADC (analog sensors)
#   0x40 - PCA9685 (servo/motor PWM)
```

### 3. Install Dependencies

#### Minimal (motors and basic sensors only):
```bash
sudo apt-get update
sudo apt-get install python3-numpy
```

#### Full Installation (with camera and audio):
```bash
sudo apt-get update
sudo apt-get install python3-numpy python3-opencv python3-pyaudio
sudo apt-get install libatlas-base-dev  # NumPy optimization

# For Raspberry Pi Camera Module:
sudo apt-get install python3-picamera2

# Enable camera in raspi-config
sudo raspi-config
# Interface Options → Camera → Enable
```

#### Verify Camera (if installed):
```bash
vcgencmd get_camera
# Should show: supported=1 detected=1

# Test camera
libcamera-hello
```

#### Verify Audio (if using):
```bash
# List audio devices
aplay -l   # Playback devices
arecord -l # Recording devices

# Test speaker
speaker-test -t sine -f 1000 -l 1
```

## Deployment

```bash
# From your development machine
export PI_HOST=pi@192.168.1.231
bash deploy.sh

# On the Raspberry Pi
cd ~/picarx_robot
./test.sh  # Test hardware directly
```

## Why This Approach?

### Advantages
- **No dependency hell** - No version conflicts with SunFounder libraries
- **Direct control** - We know exactly what's happening at hardware level
- **Portable** - Code is self-contained, no external dependencies
- **Maintainable** - Can see and modify every hardware interaction
- **Educational** - Shows how servo control actually works

### How It Works
Instead of:
```python
# SunFounder way (abstracted)
px = Picarx()
px.set_dir_servo_angle(30)
```

We do:
```python
# Our way (direct)
# Calculate PWM counts for 30° servo position
pulse_us = 1500 + (30 * 500/90)  # Convert angle to microseconds
counts = int(pulse_us * 4096 / 20000)  # Convert to PCA9685 counts
# Write directly to PCA9685 register
i2c_bus.write_byte_data(0x40, register, counts)
```

## Troubleshooting

### "GPIO not available" on development machine
This is normal! The mock hardware allows testing logic without a Pi.

### "I2C device not found"
```bash
# Check I2C is enabled
ls /dev/i2c*  # Should show /dev/i2c-1

# Check devices are connected
sudo i2cdetect -y 1
```

### Motors don't move
```bash
# Test PCA9685 directly
sudo i2cset -y 1 0x40 0x00 0x00  # Reset
sudo i2cset -y 1 0x40 0xFE 0x79  # Set 50Hz
# If no error, PCA9685 is working
```

---

**Bottom line**: This is a truly standalone implementation that needs NO external robot libraries!