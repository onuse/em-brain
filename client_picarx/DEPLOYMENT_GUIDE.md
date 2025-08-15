# 🚀 PiCar-X Brainstem Deployment Guide

Last Updated: 2025-08-14

## Prerequisites

### On Your Development Machine
- Code pushed to GitHub
- SSH access to Raspberry Pi
- Brain server running (or know its IP)

### On Raspberry Pi
- Raspbian OS installed
- Python 3.7+ 
- Git installed
- GPIO and I2C enabled
- Network connection to brain server

## Quick Git Deployment

### 1. Push Latest Changes (on dev machine)
```bash
# Commit your changes
cd em-brain
git add .
git commit -m "Updated brainstem for deployment"
git push origin master
```

### 2. Deploy to Raspberry Pi via Git
```bash
# SSH to Pi
ssh pi@192.168.1.xxx

# First time setup - clone repo
cd ~
git clone https://github.com/yourusername/em-brain.git
cd em-brain/client_picarx

# OR if already cloned - just pull latest
cd ~/em-brain
git pull origin master
cd client_picarx

# Install dependencies
sudo pip3 install -r requirements.txt
```

### 3. Configure for Your Network
```bash
# Edit config for your brain server IP
nano config/robot_config.json

# Change brain host to your server:
{
  "brain": {
    "host": "192.168.1.231",  # Your brain server IP
    "port": 9999
  }
}
```

That's it! Much cleaner than copying files.

## Audio Setup (Optional but Recommended)

### Install Audio Support
```bash
# Install PyAudio for microphone support
sudo apt-get update
sudo apt-get install python3-pyaudio portaudio19-dev

# Test audio devices
python3 -c "
import pyaudio
p = pyaudio.PyAudio()
print(f'Found {p.get_device_count()} audio devices:')
for i in range(p.get_device_count()):
    info = p.get_device_info_by_index(i)
    print(f'  {i}: {info[\"name\"]} ({info[\"maxInputChannels\"]} in, {info[\"maxOutputChannels\"]} out)')
"

# If you have a USB microphone, it should appear in the list
```

### What the Brain Receives from Audio
The microphone provides 7 real-time features to the brain:
1. **Volume** - Overall sound level
2. **Bass** (20-250 Hz) - Rumbles, motors
3. **Mid-Low** (250-1000 Hz) - Voice fundamentals  
4. **Mid-High** (1-4 kHz) - Voice harmonics
5. **Treble** (4-8 kHz) - Sharp sounds
6. **Pitch** - Dominant frequency
7. **Onset** - Sudden changes (claps, impacts)

The brain learns what these patterns mean through experience!

## Testing Deployment

### 1. Run Deployment Test
```bash
cd ~/picarx_robot
python3 test_brainstem_deployment.py
```

Expected output:
```
🧪 Testing: Import brainstem module...
   ✅ PASS
🧪 Testing: Import brain client...
   ✅ PASS
[... more tests ...]
🎉 ALL TESTS PASSED - READY FOR DEPLOYMENT!
```

### 2. Test Hardware (No Brain Needed)
```bash
# Test with mock HAL first
sudo python3 -c "
from src.brainstem.brainstem import Brainstem
b = Brainstem(enable_monitor=False)
print('Brainstem created successfully')
"
```

### 3. Test Brain Connection
```bash
# Make sure brain server is running first!
python3 -c "
from src.brainstem.brain_client import BrainClient, BrainServerConfig
config = BrainServerConfig(host='192.168.1.100', port=9999)
client = BrainClient(config)
if client.connect():
    print('✅ Connected to brain!')
else:
    print('❌ Could not connect')
"
```

### 4. Monitor Telemetry
```bash
# In another terminal
nc localhost 9997  # Check brainstem monitor
nc localhost 9998  # Check brain monitor (if accessible)
```

## Running the Robot

### Manual Start
```bash
cd ~/picarx_robot

# With default config
sudo python3 picarx_robot.py

# With specific brain host
sudo python3 picarx_robot.py --brain-host 192.168.1.100

# With monitoring
sudo python3 picarx_robot.py --brain-host 192.168.1.100 2>&1 | tee robot.log
```

### Create Convenience Scripts
```bash
# Create run.sh
cat > ~/picarx_robot/run.sh << 'EOF'
#!/bin/bash
BRAIN_HOST=${BRAIN_HOST:-192.168.1.100}
echo "Starting PiCar-X Robot (Brain: $BRAIN_HOST)"
sudo python3 picarx_robot.py --brain-host $BRAIN_HOST
EOF
chmod +x ~/picarx_robot/run.sh

# Create monitor.sh
cat > ~/picarx_robot/monitor.sh << 'EOF'
#!/bin/bash
echo "Brainstem Telemetry (refresh every 2s)"
while true; do
    clear
    echo "" | nc localhost 9997 | python3 -m json.tool | head -40
    sleep 2
done
EOF
chmod +x ~/picarx_robot/monitor.sh
```

### Systemd Service (Auto-start)
```bash
sudo tee /etc/systemd/system/picarx-robot.service << EOF
[Unit]
Description=PiCar-X Robot Brainstem
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=root
WorkingDirectory=/home/pi/picarx_robot
Environment="PYTHONUNBUFFERED=1"
ExecStart=/usr/bin/python3 /home/pi/picarx_robot/picarx_robot.py --brain-host 192.168.1.100
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

# Enable and start
sudo systemctl daemon-reload
sudo systemctl enable picarx-robot
sudo systemctl start picarx-robot

# Check status
sudo systemctl status picarx-robot
sudo journalctl -u picarx-robot -f  # Follow logs
```

## Troubleshooting

### Common Issues

**1. ImportError: No module named 'RPi'**
```bash
sudo pip3 install RPi.GPIO
```

**2. Permission denied on GPIO/I2C**
```bash
# Must run with sudo
sudo python3 picarx_robot.py

# Or add user to gpio group
sudo usermod -a -G gpio,i2c $USER
# Then logout and login
```

**3. Cannot connect to brain**
```bash
# Check network
ping 192.168.1.100

# Check brain server is running
nc -zv 192.168.1.100 9999

# Check firewall on brain server
```

**4. Sensors reading wrong values**
```bash
# Edit config/robot_config.json
# Adjust conversion factors:
"us_per_cm": 58.0,  # Try 56-60
"adc_to_voltage_multiplier": 10.0  # Depends on circuit
```

**5. Motors not moving**
```bash
# Check I2C devices detected
sudo i2cdetect -y 1
# Should show devices at 0x14 (ADC) and 0x40 (PCA9685)

# Test PWM directly
sudo python3 -c "
from src.hardware.bare_metal_hal import create_hal, RawMotorCommand
hal = create_hal()
hal.initialize()
# Very slow test
cmd = RawMotorCommand(0.1, True, 0.1, True)
hal.execute_motor_command(cmd)
import time; time.sleep(1)
hal.emergency_stop()
"
```

### Monitoring During Testing

**Terminal 1 - Run Robot:**
```bash
cd ~/picarx_robot
sudo ./run.sh
```

**Terminal 2 - Watch Telemetry:**
```bash
./monitor.sh
```

**Terminal 3 - Watch Logs:**
```bash
tail -f robot.log
```

**Terminal 4 - Emergency Stop:**
```bash
# If robot goes crazy
sudo pkill -9 python3
```

## Safety Checklist

Before first run:
- [ ] Robot on blocks (wheels off ground)
- [ ] Power supply stable (7.4V LiPo)
- [ ] Emergency stop ready (Ctrl+C or power switch)
- [ ] Config has safe values (3cm collision, low speeds)
- [ ] Monitor shows telemetry (port 9997)

During testing:
- [ ] Start with reduced motor speeds
- [ ] Test reflexes work (block ultrasonic)
- [ ] Verify battery monitoring
- [ ] Check cycle times (<50ms)

## Performance Tuning

If cycle times are too slow:

1. **Reduce vision resolution** in config:
```json
"vision": {
  "resolution": [320, 240]  // Instead of 640x480
}
```

2. **Increase brain timeout**:
```json
"brain": {
  "timeout": 0.1  // 100ms instead of 50ms
}
```

3. **Check CPU temperature**:
```bash
vcgencmd measure_temp
# If >70°C, add cooling
```

## Success Criteria

You know deployment is successful when:
- ✅ `test_brainstem_deployment.py` passes all tests
- ✅ Telemetry accessible on port 9997
- ✅ Brain connection established (if server running)
- ✅ Safety reflexes trigger (test with hand in front)
- ✅ Motors respond to brain commands
- ✅ No errors in logs for 1+ minute

## Next Steps

1. **Tune safety thresholds** based on actual sensor readings
2. **Calibrate motor curves** for your specific motors
3. **Add vision/audio modules** if hardware available
4. **Create dashboard** to visualize telemetry
5. **Train the brain** with real sensor data!

---

Happy deploying! The robot awaits its brain! 🤖🧠