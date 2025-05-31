#!/usr/bin/env python3
"""
Portable SUMO Project Launcher

This script provides a cross-platform way to launch the SUMO Traffic Light Control project.
It automatically detects and sets up SUMO environment before running any commands.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from setup_environment import SUMOEnvironmentSetup
except ImportError:
    print("❌ Cannot import setup_environment. Make sure setup_environment.py exists.")
    sys.exit(1)


class PortableLauncher:
    """Portable launcher for SUMO Traffic Light Control project."""
    
    def __init__(self):
        self.project_root = PROJECT_ROOT
        self.python_exe = sys.executable
        self.setup = SUMOEnvironmentSetup()
        
        # Available commands
        self.commands = {
            # Environment commands
            'setup': {
                'description': 'Setup complete environment (SUMO + dependencies)',
                'action': self.setup_environment
            },
            'setup-sumo': {
                'description': 'Setup SUMO only',
                'action': self.setup_sumo_only
            },
            'setup-deps': {
                'description': 'Install Python dependencies only',
                'action': self.setup_dependencies_only
            },
            'check': {
                'description': 'Check SUMO installation and environment',
                'action': self.check_environment
            },
            
            # Training commands
            'train': {
                'description': 'Run headless training (50 episodes)',
                'action': lambda: self.run_training(episodes=50)
            },
            'train-gui': {
                'description': 'Run GUI training (50 episodes)',
                'action': lambda: self.run_training(episodes=50, gui=True)
            },
            'train-fast': {
                'description': 'Run fast training (20 episodes)',
                'action': lambda: self.run_training(episodes=20)
            },
            'train-long': {
                'description': 'Run long training (300 episodes)',
                'action': lambda: self.run_training(episodes=300)
            },
            'demo': {
                'description': 'Run quick demo (2 episodes with GUI)',
                'action': lambda: self.run_training(episodes=2, gui=True)
            },
            
            # Testing commands
            'test': {
                'description': 'Test SUMO installation',
                'action': self.test_sumo
            },
            'test-reward': {
                'description': 'Test enhanced reward function',
                'action': self.test_reward_function
            },
            
            # Utility commands
            'clean': {
                'description': 'Clean generated files',
                'action': self.clean_files
            },
            'status': {
                'description': 'Show project status',
                'action': self.show_status
            }
        }
    
    def ensure_sumo_environment(self):
        """Ensure SUMO environment is properly set up."""
        sumo_home = self.setup.detect_sumo_installation()
        if not sumo_home:
            print("⚠️ SUMO not detected. Setting up environment...")
            if not self.setup.setup_complete_environment():
                print("❌ Failed to setup SUMO environment")
                print("💡 Try running: python portable_launcher.py setup")
                return False
        else:
            # Just set environment variables
            self.setup._set_environment_variables(sumo_home)
        
        return True
    
    def setup_environment(self):
        """Setup complete environment."""
        print("🚀 Setting up complete environment...")
        return self.setup.setup_complete_environment()
    
    def setup_sumo_only(self):
        """Setup SUMO only."""
        print("🚀 Setting up SUMO...")
        sumo_home = self.setup.detect_sumo_installation()
        if not sumo_home:
            return self.setup.install_sumo()
        else:
            print(f"✅ SUMO already installed: {sumo_home}")
            return True
    
    def setup_dependencies_only(self):
        """Install Python dependencies only."""
        print("🐍 Installing Python dependencies...")
        return self.setup.install_python_dependencies()
    
    def check_environment(self):
        """Check SUMO installation and environment."""
        print("🔍 Checking environment...")
        
        # Check SUMO
        sumo_home = self.setup.detect_sumo_installation()
        if sumo_home:
            self.setup._set_environment_variables(sumo_home)
        
        # Verify installation
        return self.setup.verify_installation()
    
    def run_training(self, episodes=50, gui=False, max_steps=600):
        """Run training with automatic environment setup."""
        if not self.ensure_sumo_environment():
            return False
        
        print(f"🚂 Starting training: {episodes} episodes, GUI={'Yes' if gui else 'No'}")
        
        # Prepare command
        cmd = [
            self.python_exe,
            str(self.project_root / 'complete_sumo_training.py'),
            '--episodes', str(episodes),
            '--max-steps', str(max_steps)
        ]
        
        if gui:
            cmd.append('--gui')
        
        try:
            result = subprocess.run(cmd, cwd=self.project_root)
            return result.returncode == 0
        except KeyboardInterrupt:
            print("\n⏹️ Training interrupted by user")
            return True
        except Exception as e:
            print(f"❌ Training failed: {e}")
            return False
    
    def test_sumo(self):
        """Test SUMO installation."""
        if not self.ensure_sumo_environment():
            return False
        
        print("🧪 Testing SUMO installation...")
        
        # Test basic SUMO command
        try:
            import platform
            system = platform.system().lower()
            binary_name = 'sumo.exe' if system == 'windows' else 'sumo'
            
            result = subprocess.run([binary_name, '--help'], 
                                 capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                print("✅ SUMO test passed")
                return True
            else:
                print("❌ SUMO test failed")
                return False
        except Exception as e:
            print(f"❌ SUMO test error: {e}")
            return False
    
    def test_reward_function(self):
        """Test enhanced reward function."""
        if not self.ensure_sumo_environment():
            return False
        
        print("🧪 Testing enhanced reward function...")
        
        try:
            cmd = [self.python_exe, str(self.project_root / 'test_enhanced_reward.py')]
            result = subprocess.run(cmd, cwd=self.project_root)
            return result.returncode == 0
        except Exception as e:
            print(f"❌ Reward function test failed: {e}")
            return False
    
    def clean_files(self):
        """Clean generated files."""
        print("🧹 Cleaning generated files...")
        
        patterns = [
            'config/*.xml',
            'config/*.net.xml',
            'config/*.rou.xml',
            'config/*.sumocfg',
            '*.log',
            '__pycache__',
            '*.pyc'
        ]
        
        import glob
        import shutil
        
        cleaned = 0
        for pattern in patterns:
            for file_path in glob.glob(str(self.project_root / pattern), recursive=True):
                try:
                    if os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                    else:
                        os.remove(file_path)
                    cleaned += 1
                except Exception:
                    pass
        
        print(f"✅ Cleaned {cleaned} files/directories")
        return True
    
    def show_status(self):
        """Show project status."""
        print("📊 Project Status")
        print("-" * 40)
        
        # Check SUMO
        sumo_home = self.setup.detect_sumo_installation()
        if sumo_home:
            print(f"✅ SUMO: Installed ({sumo_home})")
        else:
            print("❌ SUMO: Not found")
        
        # Check Python dependencies
        try:
            import traci
            import numpy
            import matplotlib
            import gymnasium
            print("✅ Python dependencies: Available")
        except ImportError as e:
            print(f"❌ Python dependencies: Missing ({e})")
        
        # Check project structure
        directories = ['src', 'config', 'models', 'plots']
        for directory in directories:
            if (self.project_root / directory).exists():
                print(f"✅ Directory {directory}: Exists")
            else:
                print(f"❌ Directory {directory}: Missing")
        
        # Check main files
        files = ['complete_sumo_training.py', 'setup_environment.py', 'requirements.txt']
        for file in files:
            if (self.project_root / file).exists():
                print(f"✅ File {file}: Exists")
            else:
                print(f"❌ File {file}: Missing")
        
        return True
    
    def print_help(self):
        """Print help message."""
        print("🚀 Portable SUMO Traffic Light Control Launcher")
        print("=" * 50)
        print()
        print("This launcher automatically detects and sets up SUMO environment")
        print("before running any commands. Works on Windows, macOS, and Linux.")
        print()
        print("Usage: python portable_launcher.py [command]")
        print()
        
        # Group commands
        groups = {
            'Environment Setup': ['setup', 'setup-sumo', 'setup-deps', 'check'],
            'Training': ['train', 'train-gui', 'train-fast', 'train-long', 'demo'],
            'Testing': ['test', 'test-reward'],
            'Utilities': ['clean', 'status']
        }
        
        for group, commands in groups.items():
            print(f"{group}:")
            for cmd in commands:
                if cmd in self.commands:
                    print(f"  {cmd:<15} - {self.commands[cmd]['description']}")
            print()
        
        print("Examples:")
        print("  python portable_launcher.py setup       # First-time setup")
        print("  python portable_launcher.py demo        # Quick demo")
        print("  python portable_launcher.py train-gui   # Training with GUI")
        print("  python portable_launcher.py check       # Check installation")
    
    def run(self):
        """Main entry point."""
        parser = argparse.ArgumentParser(description='Portable SUMO Project Launcher')
        parser.add_argument('command', nargs='?', default='help',
                          help='Command to run (see help for available commands)')
        parser.add_argument('--force-setup', action='store_true',
                          help='Force environment setup even if SUMO is detected')
        
        args = parser.parse_args()
        
        if args.force_setup and args.command != 'help':
            print("🔄 Force setup requested...")
            self.setup_environment()
        
        if args.command == 'help' or args.command not in self.commands:
            self.print_help()
            return True
        
        # Run the command
        try:
            success = self.commands[args.command]['action']()
            if success:
                print(f"✅ Command '{args.command}' completed successfully")
            else:
                print(f"❌ Command '{args.command}' failed")
            return success
        except KeyboardInterrupt:
            print(f"\n⏹️ Command '{args.command}' interrupted by user")
            return True
        except Exception as e:
            print(f"❌ Command '{args.command}' error: {e}")
            return False


def main():
    """Main function."""
    launcher = PortableLauncher()
    success = launcher.run()
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
