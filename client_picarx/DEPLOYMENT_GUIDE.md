# 🚀 Raspberry Pi Deployment Guide

## Complete Instructions for Deploying to PiCar-X Robot

This guide walks you through deploying the brainstem software to your Raspberry Pi-powered PiCar-X robot.

---

## 📋 Prerequisites

### Hardware
- **Raspberry Pi Zero 2 W** (minimum) or Raspberry Pi 4 (recommended)
- **SunFounder PiCar-X** robot kit (assembled)
- **MicroSD card** (16GB minimum, 32GB recommended)
- **Power supply** for initial setup
- **WiFi network** for brain server connection

### Your Development Machine
- SSH client (built into Linux/Mac, use PuTTY on Windows)
- SD card writer
- Git (to get the code)

---

## 🖥️ Step 1: Prepare Raspberry Pi OS

### Recommended OS
**Raspberry Pi OS Lite (64-bit)** - No desktop needed!
- Lighter weight = more resources for brain
- Faster boot time
- Less SD card wear

### Flash the OS

1. Download Raspberry Pi Imager: https://www.raspberrypi.com/software/
2. Insert your SD card
3. In Imager, choose:
   - **OS**: Raspberry Pi OS Lite (64-bit)
   - **Storage**: Your SD card

4. **IMPORTANT**: Click the gear icon for advanced settings:
   ```
   ☑ Enable SSH
   ☑ Set username and password
     Username: jonas
     Password: [choose secure password]
   ☑ Configure WiFi
     SSID: [your network]
     Password: [your wifi password]
   ☑ Set locale settings
     Time zone: [your timezone]
   ```

5. Write the image and wait for completion

---

## 🔌 Step 2: Initial Pi Setup

1. **Insert SD card** into Pi and power on
2. **Wait 2-3 minutes** for first boot
3. **Find your Pi's IP address**:
   ```bash
   # From your computer, scan network:
   # On Linux/Mac:
   arp -a | grep brain
   # Or use your router's admin panel
   ```

4. **SSH into your Pi**:
   ```bash
   ssh pi@<raspberry-pi-ip>
   # Enter the password you set
   ```

5. **Update the system**:
   ```bash
   sudo apt update && sudo apt upgrade -y
   sudo apt install -y git python3-pip python3-venv python3-setuptools python3-smbus
   ```

6. **Enable required interfaces**:
   ```bash
   sudo raspi-config
   # Navigate to: Interface Options
   # Enable: I2C, SPI, Camera (if using)
   # Finish and reboot
   ```

---

## 🤖 Step 3: Install SunFounder Libraries

The PiCar-X needs specific drivers for its hardware:

```bash
# Install SunFounder robot-hat library
cd ~
git clone -b v2.0 https://github.com/sunfounder/robot-hat.git
cd robot-hat
sudo python3 setup.py install
(The "setup.py" installation method is planned to be abandoned.
Please execute "install.py" to install.))

# Install the vilib camera module.
cd ~/
git clone -b picamera2 https://github.com/sunfounder/vilib.git
cd vilib
sudo python3 install.py

# Install PiCar-X library
cd ~
git clone -b v2.0 https://github.com/sunfounder/picar-x.git --depth 1
cd picar-x
sudo python3 setup.py install

# Finally, you need to run the script i2samp.sh to install the components required by the i2s amplifier, otherwise the picar-x will have no sound.
cd ~/picar-x
sudo bash i2samp.sh

# Run calibration (do this with robot on blocks!)
cd ~/picar-x/example/calibration
sudo python3 calibration.py
```

---

## 🧠 Step 4: Deploy Brain Client

### Option A: Direct Transfer (Recommended)

From your development machine:

```bash
# Package the client
cd /path/to/em-brain
tar -czf picarx-client.tar.gz client_picarx/

# Transfer to Pi
scp picarx-client.tar.gz pi@<raspberry-pi-ip>:~/

# On the Pi, extract
ssh pi@<raspberry-pi-ip>
tar -xzf picarx-client.tar.gz
cd client_picarx
```

### Option B: Git Clone

```bash
# On the Pi
cd ~
git clone <your-repo-url> em-brain
cd em-brain/client_picarx
```

---

## 📦 Step 5: Install Dependencies

```bash
cd ~/client_picarx

# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate

# Install requirements
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu  # CPU-only for Pi
pip install numpy psutil

# If you have a requirements.txt:
pip install -r requirements.txt
```

---

## ⚙️ Step 6: Configure Brain Connection

```bash
# Edit configuration
nano ~/client_picarx/config/client_settings.json
```

Set your brain server details:
```json
{
  "brain_host": "192.168.1.100",  // Your brain server IP
  "brain_port": 9999,
  "robot_profile": "default",
  "safe_mode": true
}
```

Or use environment variables:
```bash
# Add to ~/.bashrc
export BRAIN_HOST=192.168.1.100
export BRAIN_PORT=9999
export SAFE_MODE=true
```

---

## 🏃 Step 7: Test Run

```bash
cd ~/client_picarx

# Activate virtual environment
source venv/bin/activate

# Test with mock brain first
python picarx_robot.py --mock-brain

# If that works, test with real brain
python picarx_robot.py --brain-host $BRAIN_HOST
```

Watch for:
- "✅ Connected to brain server" message
- Motor test sequence
- Sensor readings

---

## 🚦 Step 8: Auto-Start on Boot

Create a systemd service:

```bash
sudo nano /etc/systemd/system/picarx-brain.service
```

Add:
```ini
[Unit]
Description=PiCar-X Brain Client
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/client_picarx
Environment="BRAIN_HOST=192.168.1.100"
Environment="BRAIN_PORT=9999"
ExecStart=/home/pi/client_picarx/venv/bin/python /home/pi/client_picarx/picarx_robot.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl daemon-reload
sudo systemctl enable picarx-brain.service
sudo systemctl start picarx-brain.service

# Check status
sudo systemctl status picarx-brain.service

# View logs
sudo journalctl -u picarx-brain.service -f
```

---

## 🔍 Step 9: Verify Deployment

### Check Everything is Running:

```bash
# 1. Check service
sudo systemctl status picarx-brain

# 2. Check connection to brain
netstat -an | grep 9999

# 3. Check CPU/Memory usage
htop

# 4. Monitor logs
tail -f /var/log/syslog | grep picarx
```

### Test Robot Functions:

```python
# Quick hardware test
cd ~/client_picarx
python3 -c "
from picarx import Picarx
px = Picarx()
px.forward(30)
import time; time.sleep(1)
px.stop()
print('✅ Motors work!')
"
```

---

## 🛠️ Troubleshooting

### Can't Connect to Brain Server
```bash
# Test network
ping <brain-server-ip>

# Check firewall on brain server
# Port 9999 must be open

# Try with explicit IP
python picarx_robot.py --brain-host 192.168.1.100 --debug
```

### GPIO Permission Errors
```bash
# Add user to gpio group
sudo usermod -a -G gpio pi
# Logout and back in
```

### Performance Issues on Pi Zero
```bash
# Reduce sensor polling rate
# In brainstem_config.py:
sensor_poll_rate = 0.02  # 50Hz instead of 100Hz

# Use smaller brain configuration
export BRAIN_SIZE=tiny
```

---

## 📁 What Gets Deployed

Only these directories are needed on the Pi:

```
client_picarx/
├── picarx_robot.py          # Main entry point
├── src/
│   ├── brainstem/          # Core logic
│   ├── config/             # Configuration
│   └── hardware/           # Hardware interfaces
├── config/
│   └── client_settings.json
└── requirements.txt
```

**NOT needed on Pi:**
- `/docs` - Documentation
- `/tests` - Test files
- `/tools` - Development tools
- Research files (`research_*.py`)

---

## 🎯 Quick Deployment Script

Save this as `deploy.sh` on your development machine:

```bash
#!/bin/bash
PI_HOST="pi@192.168.1.xxx"
PI_DIR="/home/pi/client_picarx"

echo "🚀 Deploying to $PI_HOST..."

# Package only needed files
tar -czf deploy.tar.gz \
  --exclude='*.pyc' \
  --exclude='__pycache__' \
  --exclude='docs' \
  --exclude='tests' \
  --exclude='tools' \
  --exclude='.git' \
  client_picarx/

# Transfer
scp deploy.tar.gz $PI_HOST:~/

# Extract and restart
ssh $PI_HOST << 'EOF'
  tar -xzf deploy.tar.gz
  sudo systemctl restart picarx-brain
  echo "✅ Deployment complete!"
EOF

rm deploy.tar.gz
```

---

## ✅ Success Checklist

- [ ] Raspberry Pi OS Lite installed and updated
- [ ] SSH access working
- [ ] I2C and SPI enabled
- [ ] SunFounder libraries installed
- [ ] Brain client code deployed
- [ ] Python dependencies installed
- [ ] Brain server connection configured
- [ ] Test run successful
- [ ] Auto-start service enabled
- [ ] Robot responds to brain commands

---

## 🎉 You're Ready!

Your PiCar-X is now a body for emergent intelligence! The brain will discover how to use its sensors and motors through pure experience.

**Next steps:**
1. Start the brain server (on your powerful machine)
2. Power on the robot
3. Watch behaviors emerge!
4. Document what surprises you

---

*Remember: The robot knows nothing at first. Every behavior it develops is discovered, not programmed!*