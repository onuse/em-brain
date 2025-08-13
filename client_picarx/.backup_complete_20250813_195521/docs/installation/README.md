# PiCar-X Brain Client Installation Guide

Complete deployment guide for installing the PiCar-X brain client on Raspberry Pi Zero 2 WH hardware.

## 🎯 Overview

This installation creates a complete robotic brain client that:
- Controls PiCar-X hardware (motors, servos, camera, sensors)
- Communicates with the centralized brain server
- Provides vocal expression capabilities
- Runs autonomously with auto-start on boot
- Supports remote updates and monitoring

## 📋 Prerequisites

### Hardware Requirements
- **Raspberry Pi Zero 2 WH** (recommended)
- **PiCar-X Robot Kit** from SunFounder
- **MicroSD card** (32GB+ recommended)
- **Stable internet connection** (WiFi)

### Software Requirements
- **Raspberry Pi OS** (latest version)
- **SSH access** enabled
- **Git** installed
- **Python 3.7+** (included in Raspberry Pi OS)

## 🚀 Quick Installation

### 1. Prepare Your Pi
```bash
# On your Pi Zero (via SSH)
sudo apt update && sudo apt upgrade -y
sudo apt install git -y
```

### 2. Clone and Install
```bash
# Clone the brain client repository
git clone https://github.com/your-username/picarx-brain-client.git
cd picarx-brain-client

# Run complete installation
chmod +x docs/installation/install.sh
./docs/installation/install.sh
```

### 3. Configure Brain Server Connection
```bash
# Edit configuration file
nano config/client_settings.json

# Set your brain server IP address:
{
  "brain_server": {
    "host": "192.168.1.100",  # <- Change this to your brain server IP
    "port": 8080
  }
}
```

### 4. Test Installation
```bash
# Start the brain client service
sudo systemctl start picarx-brain.service

# Check service status
sudo systemctl status picarx-brain.service

# View live logs
sudo journalctl -u picarx-brain.service -f
```

### 5. Enable Auto-Start
```bash
# Enable automatic startup on boot
sudo systemctl enable picarx-brain.service

# Reboot to test
sudo reboot
```

## 📁 Installation Components

The installation script sets up several components:

### SunFounder Hardware Drivers
- **robot-hat v2.0** - GPIO and I2C hardware control
- **vilib (picamera2)** - Computer vision and camera support
- **picar-x v2.0** - PiCar-X specific motor and sensor drivers
- **I2S audio driver** - Audio amplifier support for vocal expressions

### Brain Client System
- **Python dependencies** - All required packages
- **Configuration files** - Robot settings and brain server connection
- **Logging system** - Structured logging to files and systemd journal
- **Auto-start service** - Systemd service for boot-time startup

### Directory Structure
```
picarx-brain-client/
├── main.py                    # Main entry point
├── config/
│   └── client_settings.json   # Robot configuration
├── src/                       # Brain client source code
├── docs/
│   └── installation/          # Installation scripts
├── logs/                      # Log files (created at runtime)
└── requirements.txt           # Python dependencies
```

## ⚙️ Configuration

### Brain Server Connection
Edit `config/client_settings.json`:
```json
{
  "brain_server": {
    "host": "192.168.1.100",     # IP of your brain server
    "port": 8080,                # Brain server port
    "connection_timeout": 10,     # Connection timeout (seconds)
    "retry_interval": 5          # Retry interval on connection loss
  }
}
```

### Hardware Constraints
Safety limits are configured in `client_settings.json`:
```json
{
  "hardware": {
    "max_speed": 50,              # Maximum motor speed (0-100)
    "max_steering_angle": 30,     # Maximum steering angle (degrees)
    "ultrasonic_safety_distance": 10,  # Emergency stop distance (cm)
    "camera_enabled": true,       # Enable camera system
    "vocal_enabled": true         # Enable vocal expression system
  }
}
```

### Robot Identity
Each robot can have unique identification:
```json
{
  "robot_identity": {
    "robot_id": "picarx_001",     # Unique robot identifier
    "robot_type": "picar-x",      # Robot hardware type
    "location": "lab"             # Deployment location
  }
}
```

## 🔧 Management Commands

### Service Control
```bash
# Start brain client
sudo systemctl start picarx-brain.service

# Stop brain client
sudo systemctl stop picarx-brain.service

# Restart brain client
sudo systemctl restart picarx-brain.service

# Check service status
sudo systemctl status picarx-brain.service

# Enable auto-start on boot
sudo systemctl enable picarx-brain.service

# Disable auto-start
sudo systemctl disable picarx-brain.service
```

### Logging and Debugging
```bash
# View live logs
sudo journalctl -u picarx-brain.service -f

# View recent logs
sudo journalctl -u picarx-brain.service --since "1 hour ago"

# View all logs for today
sudo journalctl -u picarx-brain.service --since today

# Check log files
tail -f /home/pi/picarx_logs/client.log
```

### Updates and Maintenance
```bash
# Update brain client code
cd ~/picarx-brain-client
git pull origin main
sudo systemctl restart picarx-brain.service

# Check Python dependencies
pip3 list | grep -E "(robot|picar|vilib)"

# Test hardware connections
python3 -c "
from picarx import Picarx
import robot_hat
car = Picarx()
print('✅ Hardware initialization successful')
"
```

## 🛠️ Troubleshooting

### Common Issues

#### 1. Service Won't Start
```bash
# Check service logs
sudo journalctl -u picarx-brain.service --no-pager

# Check file permissions
ls -la /etc/systemd/system/picarx-brain.service

# Reload systemd
sudo systemctl daemon-reload
```

#### 2. Hardware Not Detected
```bash
# Check I2C is enabled
sudo raspi-config
# Interface Options > I2C > Enable

# Test I2C devices
sudo i2cdetect -y 1

# Reinstall SunFounder drivers
cd ~/picarx-brain-client
./docs/installation/sunfounder_setup.sh
```

#### 3. Brain Server Connection Failed
```bash
# Test network connectivity
ping 192.168.1.100  # Replace with your brain server IP

# Check configuration
cat config/client_settings.json

# Test manual connection
python3 -c "
import requests
response = requests.get('http://192.168.1.100:8080/health')
print(f'Brain server response: {response.status_code}')
"
```

#### 4. Audio System Issues
```bash
# Check audio devices
aplay -l

# Test I2S audio
speaker-test -t wav -c 2

# Reinstall I2S driver
cd ~/picar-x
sudo bash i2samp.sh
```

### Getting Help

1. **Check logs first**: `sudo journalctl -u picarx-brain.service -f`
2. **Verify hardware**: Test individual components (motors, camera, sensors)
3. **Test network**: Ensure brain server is reachable
4. **Check configuration**: Validate JSON syntax in config files

## 🔄 Uninstallation

To completely remove the brain client:

```bash
# Stop and disable service
sudo systemctl stop picarx-brain.service
sudo systemctl disable picarx-brain.service

# Remove service file
sudo rm /etc/systemd/system/picarx-brain.service
sudo systemctl daemon-reload

# Remove brain client directory
rm -rf ~/picarx-brain-client

# Optionally remove SunFounder libraries
rm -rf ~/robot-hat ~/vilib ~/picar-x

# Remove logs
rm -rf /home/pi/picarx_logs
```

## 📚 Next Steps

After successful installation:

1. **Test basic functionality** - Verify motors, sensors, and camera work
2. **Connect to brain server** - Ensure bidirectional communication  
3. **Test vocal expressions** - Verify audio output works
4. **Run autonomous navigation** - Test complete brain-robot integration
5. **Monitor performance** - Check logs and system resources

## 🔗 Related Documentation

- [Brain Server Setup Guide](../../server/docs/setup.md)
- [Hardware Troubleshooting](./hardware_troubleshooting.md)
- [Network Configuration](./network_setup.md)
- [Development Workflow](./development.md)

---

**Note**: This installation guide assumes a production deployment. For development and testing, you may want to use the mock drivers and simplified setup procedures.