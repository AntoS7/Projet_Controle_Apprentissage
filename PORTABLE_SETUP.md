# 🚀 Portable Setup Guide

## Quick Start (Any Operating System)

### Option 1: One-Command Setup
```bash
# Download and run the portable launcher
python portable_launcher.py setup
```

### Option 2: Manual Setup
```bash
# 1. Setup SUMO environment
python setup_environment.py

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Run a quick demo
python portable_launcher.py demo
```

## Platform-Specific Instructions

### 🪟 Windows

#### Method 1: Using the Batch File
1. Double-click `launcher.bat`
2. Choose `setup` from the menu
3. Wait for automatic installation
4. Run `demo` to test

#### Method 2: Command Line
```cmd
# Open Command Prompt or PowerShell
python setup_environment.py
python portable_launcher.py demo
```

#### Manual SUMO Installation (if automatic fails)
1. Download SUMO from: https://sumo.dlr.de/docs/Downloads.php
2. Install to `C:\sumo` or default location
3. Add `C:\sumo\bin` to your PATH environment variable
4. Run: `python portable_launcher.py check`

### 🍎 macOS

#### Automatic Installation
```bash
# The script will install SUMO via Homebrew automatically
python setup_environment.py
```

#### Manual Installation
```bash
# Install Homebrew (if not installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install SUMO
brew install sumo

# Verify installation
python portable_launcher.py check
```

### 🐧 Linux

#### Ubuntu/Debian
```bash
# Automatic installation
python setup_environment.py

# Or manual installation
sudo apt update
sudo apt install sumo sumo-tools sumo-doc
```

#### Fedora/CentOS/RHEL
```bash
# Automatic installation
python setup_environment.py

# Or manual installation
sudo dnf install sumo sumo-tools
```

#### Arch Linux
```bash
# Automatic installation
python setup_environment.py

# Or manual installation
sudo pacman -S sumo
```

## Usage Examples

### Training Commands
```bash
# Quick demo (2 episodes with GUI)
python portable_launcher.py demo

# Fast training (20 episodes, no GUI)
python portable_launcher.py train-fast

# GUI training (50 episodes with visualization)
python portable_launcher.py train-gui

# Long training (300 episodes for serious learning)
python portable_launcher.py train-long
```

### Testing Commands
```bash
# Check if everything is working
python portable_launcher.py check

# Test SUMO installation
python portable_launcher.py test

# Test enhanced reward function
python portable_launcher.py test-reward
```

### Utility Commands
```bash
# Show project status
python portable_launcher.py status

# Clean generated files
python portable_launcher.py clean

# Get help
python portable_launcher.py help
```

## 🛠️ Troubleshooting

### SUMO Not Found
1. **Windows**: Make sure SUMO is in your PATH or installed in `C:\sumo`
2. **macOS**: Run `brew install sumo`
3. **Linux**: Run `sudo apt install sumo sumo-tools` (Ubuntu/Debian)

### Python Dependencies Missing
```bash
# Install missing dependencies
python portable_launcher.py setup-deps
```

### Permission Issues (Linux/macOS)
```bash
# Make scripts executable
chmod +x portable_launcher.py
chmod +x setup_environment.py
```

### Port Conflicts
The system automatically finds free ports. If issues persist:
```bash
# Kill any existing SUMO processes
pkill sumo
# Or on Windows:
taskkill /f /im sumo.exe
```

## 🔧 Advanced Configuration

### Environment Variables
The setup automatically creates a `.env` file with:
```
SUMO_HOME=/path/to/sumo
```

### Custom SUMO Installation
If SUMO is installed in a non-standard location:
```bash
export SUMO_HOME="/your/custom/sumo/path"
python portable_launcher.py check
```

### Python Virtual Environment (Recommended)
```bash
# Create virtual environment
python -m venv sumo_env

# Activate it
# Windows:
sumo_env\Scripts\activate
# macOS/Linux:
source sumo_env/bin/activate

# Install dependencies
python portable_launcher.py setup-deps
```

## 📊 Project Structure

```
Project/
├── portable_launcher.py    # Main portable launcher
├── setup_environment.py    # Cross-platform SUMO setup
├── launcher.bat            # Windows batch file
├── complete_sumo_training.py
├── requirements.txt
├── src/
│   ├── environment/
│   │   ├── traffic_env.py  # Enhanced with cross-platform support
│   │   └── ...
│   └── utils/
│       ├── sumo_utils.py   # Enhanced SUMO detection
│       └── ...
├── config/                 # Generated SUMO files
├── models/                 # Saved models
├── plots/                  # Training plots
└── docs/                   # Documentation
```

## 🎯 Quick Verification

After setup, verify everything works:

```bash
# 1. Check installation
python portable_launcher.py check

# 2. Run quick test
python portable_launcher.py test

# 3. Try demo
python portable_launcher.py demo
```

You should see:
- ✅ SUMO: Installed
- ✅ Python dependencies: Available
- ✅ SUMO execution: Working
- ✅ TraCI: Importable

## 🆘 Getting Help

If you encounter issues:

1. **Check system requirements**: Python 3.8+, SUMO 1.8+
2. **Run diagnostics**: `python portable_launcher.py check`
3. **Check logs**: Look for error messages in console output
4. **Manual installation**: Follow platform-specific manual steps above
5. **Clean start**: `python portable_launcher.py clean` then setup again

## 🚀 Ready to Go!

Once setup is complete, you can:
- Run training with: `python portable_launcher.py train-gui`
- Test performance with: `python portable_launcher.py test-reward`  
- View status with: `python portable_launcher.py status`

The system is now portable and will work on any computer with Python installed!
