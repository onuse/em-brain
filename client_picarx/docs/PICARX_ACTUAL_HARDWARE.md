# PiCar-X Actual Hardware Inventory

## Standard PiCar-X Kit Components

### ✅ What We're Using
1. **PCA9685 PWM Controller** (I2C address 0x40)
   - 16 channels of 12-bit PWM
   - Controls servos and motor speed

2. **ADC Module** (I2C address 0x14)  
   - 8-channel 12-bit ADC
   - Reads analog sensors

3. **3x Grayscale Sensors**
   - IR LED + phototransistor pairs
   - Connected to ADC channels 0,1,2

4. **1x Ultrasonic Sensor**
   - HC-SR04 compatible
   - GPIO trigger/echo

5. **2x DC Motors**
   - With gear reduction
   - Controlled via PCA9685 + GPIO

6. **3x Servos**
   - Steering (1x)
   - Camera pan/tilt (2x)

7. **Battery Pack**
   - 2x 18650 batteries
   - Voltage monitoring via ADC

### ❓ Optional/Variable Components

These depend on which PiCar-X version you have:

1. **Raspberry Pi Camera**
   - Some kits include it, some don't
   - Connects via CSI ribbon cable
   - Would need `picamera2` library

2. **Speaker**
   - Some versions have a small speaker
   - Connected to PWM or audio jack
   - Could do simple beeps/tones

3. **Microphone**
   - NOT standard in most kits
   - Would need USB microphone

4. **Line Following Module**
   - Optional 5-sensor array
   - More grayscale sensors

5. **RGB LEDs**
   - Some versions have programmable LEDs
   - WS2812 or simple RGB

## What Your Robot Probably Has

Based on standard PiCar-X kit:
- ✅ All sensors/actuators we're currently using
- ❓ Camera (check if ribbon cable connected)
- ❓ Speaker (check for small speaker on board)
- ❌ Microphone (unlikely unless added)

## How to Check What You Have

```bash
# Check I2C devices
sudo i2cdetect -y 1
# 0x14 = ADC
# 0x40 = PCA9685

# Check for camera
vcgencmd get_camera
# supported=1 detected=1 means camera present

# Check for audio devices  
aplay -l  # List playback devices
arecord -l  # List recording devices

# Check GPIO in use
gpio readall
```

## Adding Missing Components

### Camera (If Not Included)
- Raspberry Pi Camera Module v2: ~$25
- Connect to CSI port
- Enable in raspi-config

### Audio Output
- Use 3.5mm audio jack (built into Pi)
- Or USB speaker
- Or PWM buzzer on GPIO

### Microphone
- USB microphone: ~$10
- Or I2S MEMS microphone

## Our Current Implementation

We're using only the **guaranteed standard components**:
- Sensors: grayscale, ultrasonic, battery
- Actuators: motors, servos
- No camera/audio (yet)

This ensures the code works on ANY PiCar-X without modification.

## Next Steps If You Want More

1. **Check if you have a camera**:
   ```bash
   vcgencmd get_camera
   ```
   If yes, we could add basic vision in ~100 lines

2. **Check audio capability**:
   ```bash
   # Test speaker (if present)
   speaker-test -t sine -f 1000
   ```

3. **Decide if worth adding**:
   - Camera: High value for emergent behaviors
   - Speaker: Medium value for feedback
   - Microphone: Low value (complex, CPU intensive)

---

**The safe approach**: Stick with current sensors until basic behaviors emerge, then add camera if available.