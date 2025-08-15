# Installation Folder File Explanation

## Essential Files

### ✅ Keep These:

- **install_minimal.sh** - Clean installation script for Raspberry Pi
- **requirements.txt** - Python dependencies
- **README.md** - Installation overview

## Research/Development Files

### 🔬 These are NOT needed for deployment:

- **hardware_discovery.py** - Development tool for exploring GPIO/hardware
- **research_motor_hal.py** - Research into motor control approaches
- **hybrid_control_design.md** - Design notes (not implementation)
- **sunfounder_api_reference.md** - API documentation (for reference only)
- **sunfounder_setup.sh** - Old setup script (replaced by install_minimal.sh)
- **install.sh** - Old complex installer (replaced)

## What Actually Gets Deployed

When deploying to Raspberry Pi, you only need:

```
client_picarx/
├── picarx_robot.py          # Entry point
├── src/                     # Core code
│   ├── brainstem/
│   ├── config/
│   └── hardware/
├── config/                  # Settings
│   └── client_settings.json
└── install_minimal.sh       # Setup script
```

The `/docs`, `/tests`, and `/tools` folders are for development only and should NOT be deployed to the Pi to save space.

## Deployment Size

- **Full repository**: ~50MB
- **Deployed code only**: ~5MB
- **With Python deps**: ~100MB (numpy, psutil)
- **SD card needed**: 16GB minimum (for OS + code + logs)