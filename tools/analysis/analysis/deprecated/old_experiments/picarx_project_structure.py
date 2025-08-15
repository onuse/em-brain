#!/usr/bin/env python3
"""
PiCar-X Source Project Structure Generator

Creates the recommended structure for the separate picarx_src project
that will contain the robot client code (brainstem, HAL, drivers).

This separates brain server code from robot client code for clean architecture.
"""

import os
from pathlib import Path


def generate_picarx_project_structure():
    """Generate recommended picarx_src project structure."""
    
    print("🏗️ PICARX_SRC PROJECT STRUCTURE DESIGN")
    print("=" * 60)
    
    structure = {
        'picarx_src/': {
            'description': 'Root directory for PiCar-X robot client code',
            'contents': {
                'README.md': 'Project overview and setup instructions',
                'requirements.txt': 'Python dependencies for robot client',
                'setup.py': 'Package installation configuration',
                '.gitignore': 'Git ignore patterns for robot code',
                'LICENSE': 'License for robot client code',
                
                'src/': {
                    'description': 'Main source code directory',
                    'contents': {
                        'picarx_brainstem/': {
                            'description': 'Core brainstem implementation',
                            'contents': {
                                '__init__.py': 'Package initialization',
                                'brainstem.py': 'Main brainstem controller',
                                'brain_client.py': 'Brain server API client',
                                'control_loop.py': 'Main robot control cycle'
                            }
                        },
                        
                        'hardware/': {
                            'description': 'Hardware Abstraction Layer',
                            'contents': {
                                '__init__.py': 'HAL package initialization',
                                'interfaces/': {
                                    'description': 'Hardware interface definitions',
                                    'contents': {
                                        'motor_interface.py': 'Motor control interface',
                                        'sensor_interface.py': 'Sensor reading interface',
                                        'camera_interface.py': 'Camera capture interface',
                                        'vocal_interface.py': 'Digital vocal cords interface'
                                    }
                                },
                                'drivers/': {
                                    'description': 'Real hardware drivers',
                                    'contents': {
                                        'picarx_motors.py': 'PiCar-X motor control',
                                        'picarx_sensors.py': 'PiCar-X sensor drivers',
                                        'picarx_camera.py': 'PiCar-X camera driver',
                                        'picarx_audio.py': 'PiCar-X audio/vocal driver'
                                    }
                                },
                                'mock/': {
                                    'description': 'Mock drivers for testing',
                                    'contents': {
                                        'mock_motors.py': 'Mock motor implementation',
                                        'mock_sensors.py': 'Mock sensor implementation',
                                        'mock_camera.py': 'Mock camera implementation',
                                        'mock_vocal.py': 'Mock vocal implementation'
                                    }
                                }
                            }
                        },
                        
                        'vocal/': {
                            'description': 'Digital vocal cord system',
                            'contents': {
                                '__init__.py': 'Vocal package initialization',
                                'digital_vocal_cords.py': 'Core vocal synthesis engine',
                                'emotional_mapper.py': 'Brain state to vocal mapping',
                                'audio_output.py': 'Audio system integration',
                                'vocal_safety.py': 'Safety constraint enforcement'
                            }
                        },
                        
                        'deployment/': {
                            'description': 'Deployment and update system',
                            'contents': {
                                '__init__.py': 'Deployment package initialization',
                                'update_client.py': 'Brain server update client',
                                'package_manager.py': 'Local package management',
                                'health_monitor.py': 'System health monitoring',
                                'rollback_manager.py': 'Update rollback capability'
                            }
                        },
                        
                        'utils/': {
                            'description': 'Utility modules',
                            'contents': {
                                '__init__.py': 'Utils package initialization',
                                'config.py': 'Configuration management',
                                'logging.py': 'Logging setup',
                                'safety.py': 'Safety utilities',
                                'performance.py': 'Performance monitoring'
                            }
                        }
                    }
                },
                
                'tests/': {
                    'description': 'Test suite',
                    'contents': {
                        'test_brainstem.py': 'Brainstem functionality tests',
                        'test_hardware.py': 'Hardware interface tests',
                        'test_vocal.py': 'Vocal system tests',
                        'test_integration.py': 'Integration tests',
                        'test_safety.py': 'Safety constraint tests'
                    }
                },
                
                'config/': {
                    'description': 'Configuration files',
                    'contents': {
                        'robot_config.yaml': 'Robot hardware configuration',
                        'vocal_config.yaml': 'Vocal system configuration',
                        'safety_config.yaml': 'Safety constraint configuration',
                        'brain_client_config.yaml': 'Brain server connection config'
                    }
                },
                
                'docs/': {
                    'description': 'Documentation',
                    'contents': {
                        'API.md': 'API documentation',
                        'DEPLOYMENT.md': 'Deployment instructions',
                        'HARDWARE.md': 'Hardware setup guide',
                        'SAFETY.md': 'Safety considerations'
                    }
                },
                
                'scripts/': {
                    'description': 'Utility scripts',
                    'contents': {
                        'start_robot.py': 'Robot startup script',
                        'stop_robot.py': 'Robot shutdown script',
                        'update_robot.py': 'Manual update script',
                        'test_hardware.py': 'Hardware test script'
                    }
                }
            }
        }
    }
    
    print("📁 Recommended Project Structure:")
    print()
    
    def print_structure(items, prefix=""):
        for name, details in items.items():
            if isinstance(details, dict):
                if 'description' in details:
                    print(f"{prefix}{name}")
                    print(f"{prefix}  └─ {details['description']}")
                    if 'contents' in details:
                        print_structure(details['contents'], prefix + "    ")
                else:
                    print(f"{prefix}{name}")
                    print_structure(details, prefix + "  ")
            else:
                print(f"{prefix}{name} - {details}")
    
    print_structure(structure)
    
    return structure


def generate_brain_picarx_api_contract():
    """Define the API contract between brain server and PiCar-X client."""
    
    print(f"\n🔗 BRAIN ↔ PICARX API CONTRACT")
    print("=" * 60)
    
    api_contract = {
        'brain_to_picarx': {
            'description': 'Commands sent from brain server to robot',
            'endpoints': {
                '/robot/control': {
                    'method': 'POST',
                    'description': 'Send motor control commands',
                    'payload': {
                        'motor_speed': 'float (-100 to 100)',
                        'steering_angle': 'float (-30 to 30 degrees)',
                        'camera_pan': 'float (-90 to 90 degrees)',
                        'camera_tilt': 'float (-30 to 30 degrees)'
                    }
                },
                '/robot/vocal': {
                    'method': 'POST', 
                    'description': 'Trigger vocal expression',
                    'payload': {
                        'emotional_state': 'string (curiosity, confidence, etc.)',
                        'intensity': 'float (0.0 to 1.0)',
                        'duration': 'float (seconds)'
                    }
                },
                '/robot/update': {
                    'method': 'POST',
                    'description': 'Deploy software update',
                    'payload': {
                        'update_url': 'string (signed package URL)',
                        'version': 'string (semantic version)',
                        'checksum': 'string (SHA256 hash)'
                    }
                },
                '/robot/config': {
                    'method': 'PUT',
                    'description': 'Update robot configuration',
                    'payload': {
                        'config_section': 'string',
                        'config_data': 'object (configuration values)'
                    }
                }
            }
        },
        
        'picarx_to_brain': {
            'description': 'Data sent from robot to brain server',
            'endpoints': {
                '/brain/sensors': {
                    'method': 'POST',
                    'description': 'Send sensor data to brain',
                    'payload': {
                        'timestamp': 'float (unix timestamp)',
                        'ultrasonic_distance': 'float (cm)',
                        'camera_data': 'bytes (compressed image)',
                        'line_sensors': 'list[float] (sensor values)',
                        'battery_voltage': 'float (volts)',
                        'motor_feedback': 'object (motor status)'
                    }
                },
                '/brain/status': {
                    'method': 'POST',
                    'description': 'Send robot status updates',
                    'payload': {
                        'timestamp': 'float (unix timestamp)',
                        'operational_status': 'string (OK, WARNING, ERROR)',
                        'error_messages': 'list[string]',
                        'performance_metrics': 'object (CPU, memory, etc.)',
                        'update_status': 'string (IDLE, UPDATING, COMPLETED, FAILED)'
                    }
                }
            }
        },
        
        'authentication': {
            'method': 'API key or certificate-based',
            'description': 'Secure authentication between brain and robot',
            'implementation': 'Bearer token in Authorization header'
        },
        
        'error_handling': {
            'connection_loss': 'Robot continues with last known safe commands',
            'invalid_commands': 'Robot rejects and logs invalid commands',
            'update_failure': 'Robot automatically rolls back to previous version'
        }
    }
    
    print("📡 Brain → PiCar-X Communication:")
    for endpoint_name, details in api_contract['brain_to_picarx']['endpoints'].items():
        print(f"   {details['method']} {endpoint_name}")
        print(f"      {details['description']}")
    
    print(f"\n📡 PiCar-X → Brain Communication:")
    for endpoint_name, details in api_contract['picarx_to_brain']['endpoints'].items():
        print(f"   {details['method']} {endpoint_name}")
        print(f"      {details['description']}")
    
    return api_contract


def generate_creation_commands():
    """Generate shell commands to create the project structure."""
    
    print(f"\n🛠️ PROJECT CREATION COMMANDS")
    print("=" * 60)
    
    commands = [
        "# Navigate to robot project root",
        "cd /Users/jkarlsson/Documents/Projects/robot-project/",
        "",
        "# Create picarx_src project structure", 
        "mkdir -p picarx_src/{src,tests,docs,config,scripts}",
        "mkdir -p picarx_src/src/{picarx_brainstem,hardware,vocal,deployment,utils}",
        "mkdir -p picarx_src/src/hardware/{interfaces,drivers,mock}",
        "",
        "# Create initial files",
        "touch picarx_src/README.md",
        "touch picarx_src/requirements.txt", 
        "touch picarx_src/setup.py",
        "",
        "# Create Python package files",
        "touch picarx_src/src/__init__.py",
        "touch picarx_src/src/picarx_brainstem/__init__.py",
        "touch picarx_src/src/hardware/__init__.py",
        "touch picarx_src/src/hardware/interfaces/__init__.py",
        "touch picarx_src/src/hardware/drivers/__init__.py",
        "touch picarx_src/src/hardware/mock/__init__.py",
        "touch picarx_src/src/vocal/__init__.py",
        "touch picarx_src/src/deployment/__init__.py",
        "touch picarx_src/src/utils/__init__.py",
        "",
        "# Initialize git repository",
        "cd picarx_src",
        "git init",
        "git add .",
        "git commit -m 'Initial picarx_src project structure'",
        "",
        "# Optional: Set up remote repository",
        "# git remote add origin <your-repo-url>",
        "# git push -u origin main"
    ]
    
    print("📋 Run these commands to create the project:")
    print()
    for command in commands:
        print(command)
    
    return commands


def main():
    """Generate complete picarx_src project plan."""
    
    print("🤖 PiCar-X Source Project Generator")
    print("=" * 80)
    print("Creating separate robot client project structure")
    print("Clean separation from brain server code")
    print()
    
    # Generate project structure
    structure = generate_picarx_project_structure()
    
    # Generate API contract
    api_contract = generate_brain_picarx_api_contract()
    
    # Generate creation commands
    commands = generate_creation_commands()
    
    print(f"\n🎯 SUMMARY")
    print("=" * 40)
    print("✅ Separate picarx_src project structure designed")
    print("✅ Brain ↔ PiCar-X API contract defined")  
    print("✅ Clean client/server separation achieved")
    print("✅ Vocal cords integrate as HAL component")
    print("✅ Ready for independent development and deployment")
    
    return {
        'project_structure': structure,
        'api_contract': api_contract,
        'creation_commands': commands
    }


if __name__ == "__main__":
    main()