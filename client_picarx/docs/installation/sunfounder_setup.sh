#!/bin/bash
"""
SunFounder PiCar-X Hardware Setup Script

This script installs all SunFounder dependencies required for PiCar-X hardware.
Based on official SunFounder installation documentation.

Run this script on your Raspberry Pi Zero 2 WH before installing the brain client.
"""

set -e  # Exit on any error

echo "🔧 SunFounder PiCar-X Hardware Setup"
echo "======================================"
echo "Installing official SunFounder drivers and dependencies..."
echo ""

# Check if running on Raspberry Pi
if ! grep -q "Raspberry Pi" /proc/cpuinfo 2>/dev/null; then
    echo "⚠️  Warning: This script is designed for Raspberry Pi hardware"
    echo "   Continuing anyway for development purposes..."
fi

# 1. System Dependencies
echo "📦 Step 1: Installing system packages..."
sudo apt update
sudo apt upgrade
sudo apt install -y git python3-pip python3-setuptools python3-smbus

echo "✅ System packages installed"
echo ""

# 2. Robot HAT Library (GPIO/I2C control)
echo "🤖 Step 2: Installing robot-hat library..."
cd ~/
if [ -d "robot-hat" ]; then
    echo "   robot-hat directory exists, removing old version..."
    rm -rf robot-hat
fi

git clone -b v2.0 https://github.com/sunfounder/robot-hat.git
cd robot-hat
sudo python3 setup.py install

echo "✅ robot-hat library installed"
echo ""

# 3. Vilib Vision Library (Camera support)
echo "📷 Step 3: Installing vilib vision library..."
cd ~/
if [ -d "vilib" ]; then
    echo "   vilib directory exists, removing old version..."
    rm -rf vilib
fi

git clone -b picamera2 https://github.com/sunfounder/vilib.git
cd vilib
sudo python3 install.py

echo "✅ vilib vision library installed"
echo ""

# 4. PiCar-X Specific Drivers
echo "🚗 Step 4: Installing PiCar-X drivers..."
cd ~/
if [ -d "picar-x" ]; then
    echo "   picar-x directory exists, removing old version..."
    rm -rf picar-x
fi

git clone -b v2.0 https://github.com/sunfounder/picar-x.git --depth 1
cd picar-x
sudo python3 setup.py install

echo "✅ PiCar-X drivers installed"
echo ""

# 5. I2S Audio Amplifier Driver
echo "🔊 Step 5: Installing I2S audio amplifier driver..."
cd ~/picar-x

# Check if i2samp.sh exists
if [ -f "i2samp.sh" ]; then
    sudo bash i2samp.sh
    echo "✅ I2S audio driver installed"
else
    echo "⚠️  i2samp.sh not found, downloading from Adafruit..."
    
    # Fallback: Download from Adafruit directly
    wget -O i2samp.py https://raw.githubusercontent.com/adafruit/Raspberry-Pi-Installer-Scripts/main/i2samp.py
    sudo python3 i2samp.py
    echo "✅ I2S audio driver installed (via Adafruit script)"
fi

echo ""

# 6. Verification
echo "🔍 Step 6: Verifying installation..."

# Test Python imports
python3 -c "
try:
    import robot_hat
    print('✅ robot_hat import: OK')
except ImportError as e:
    print(f'❌ robot_hat import failed: {e}')

try:
    import vilib
    print('✅ vilib import: OK')
except ImportError as e:
    print(f'❌ vilib import failed: {e}')

try:
    from picarx import Picarx
    print('✅ picarx import: OK')
except ImportError as e:
    print(f'❌ picarx import failed: {e}')
"

echo ""
echo "🎉 SunFounder Hardware Setup Complete!"
echo "======================================"
echo ""
echo "📋 What was installed:"
echo "   • robot-hat v2.0      - GPIO and I2C hardware control"
echo "   • vilib (picamera2)   - Computer vision and camera support"  
echo "   • picar-x v2.0        - PiCar-X specific motor and sensor drivers"
echo "   • I2S audio driver    - Audio amplifier support for vocal expressions"
echo ""
echo "🔄 Reboot recommended before proceeding with brain client installation"
echo ""
echo "📁 Installed directories:"
echo "   ~/robot-hat/          - Robot HAT library source"
echo "   ~/vilib/              - Vision library source"
echo "   ~/picar-x/            - PiCar-X library source"
echo ""
echo "🔗 Next step: Install the PiCar-X brain client with install.sh"