#!/usr/bin/env python3
"""
Cross-Platform SUMO Environment Setup Script

This script automatically detects, installs, and configures SUMO on Windows, macOS, and Linux
for the SUMO Traffic Light Control project. It ensures portability across different systems.
"""

import os
import sys
import platform
import subprocess
import shutil
import urllib.request
import zipfile
import tarfile
from pathlib import Path
import json


class SUMOEnvironmentSetup:
    """Cross-platform SUMO environment setup and detection."""
    
    def __init__(self):
        self.system = platform.system().lower()
        self.architecture = platform.machine().lower()
        self.python_executable = sys.executable
        self.project_root = Path(__file__).parent.absolute()
        
        # SUMO version to install (can be updated)
        self.sumo_version = "1.19.0"
        
        # Platform-specific configurations
        self.configs = {
            'windows': {
                'binary_name': 'sumo.exe',
                'gui_binary_name': 'sumo-gui.exe',
                'tools_binary': 'netconvert.exe',
                'install_paths': [
                    'C:/Program Files (x86)/Eclipse/Sumo',
                    'C:/Program Files/Eclipse/Sumo',
                    'C:/sumo',
                    str(Path.home() / 'sumo')
                ],
                'download_url': f'https://sourceforge.net/projects/sumo/files/sumo/version-{self.sumo_version}/sumo-win64-{self.sumo_version}.zip/download',
                'extract_folder': f'sumo-{self.sumo_version}'
            },
            'darwin': {  # macOS
                'binary_name': 'sumo',
                'gui_binary_name': 'sumo-gui',
                'tools_binary': 'netconvert',
                'install_paths': [
                    '/opt/homebrew/bin',
                    '/opt/homebrew/opt/sumo/bin',
                    '/opt/homebrew/share/sumo',
                    '/usr/local/bin',
                    '/usr/local/opt/sumo/bin',
                    '/usr/local/share/sumo'
                ],
                'homebrew_formula': 'sumo'
            },
            'linux': {
                'binary_name': 'sumo',
                'gui_binary_name': 'sumo-gui',
                'tools_binary': 'netconvert',
                'install_paths': [
                    '/usr/bin',
                    '/usr/local/bin',
                    '/usr/share/sumo',
                    '/usr/local/share/sumo',
                    '/opt/sumo/bin',
                    str(Path.home() / '.local/bin')
                ],
                'package_managers': {
                    'apt': 'sumo sumo-tools sumo-doc',
                    'yum': 'sumo sumo-tools',
                    'dnf': 'sumo sumo-tools',
                    'pacman': 'sumo',
                    'snap': 'sumo'
                }
            }
        }
    
    def detect_sumo_installation(self):
        """Detect existing SUMO installation."""
        print("🔍 Detecting SUMO installation...")
        
        # Check SUMO_HOME environment variable
        sumo_home = os.environ.get('SUMO_HOME')
        if sumo_home and self._validate_sumo_home(sumo_home):
            print(f"✅ SUMO_HOME found: {sumo_home}")
            return sumo_home
        
        # Check system PATH
        binary_name = self.configs[self.system]['binary_name']
        sumo_binary = shutil.which(binary_name)
        if sumo_binary:
            sumo_home = self._extract_sumo_home_from_binary(sumo_binary)
            if sumo_home:
                print(f"✅ SUMO detected in PATH: {sumo_home}")
                return sumo_home
        
        # Check common installation paths
        for path in self.configs[self.system]['install_paths']:
            if self._validate_sumo_home(path):
                print(f"✅ SUMO found at: {path}")
                return path
            
            # Check for binary in bin subdirectory
            bin_path = Path(path) / 'bin'
            if (bin_path / binary_name).exists():
                parent_path = str(Path(path))
                if self._validate_sumo_home(parent_path):
                    print(f"✅ SUMO found at: {parent_path}")
                    return parent_path
        
        print("❌ SUMO installation not detected")
        return None
    
    def _validate_sumo_home(self, path):
        """Validate if a path is a valid SUMO_HOME."""
        if not path or not os.path.exists(path):
            return False
        
        path = Path(path)
        
        # Check for essential SUMO components
        required_dirs = ['tools', 'data']
        required_files = []
        
        # Platform-specific binary checks
        binary_name = self.configs[self.system]['binary_name']
        tools_binary = self.configs[self.system]['tools_binary']
        
        # Check for binaries in bin subdirectory or current directory
        bin_dir = path / 'bin'
        if bin_dir.exists():
            required_files.extend([
                bin_dir / binary_name,
                bin_dir / tools_binary
            ])
        else:
            required_files.extend([
                path / binary_name,
                path / tools_binary
            ])
        
        # Check tools directory
        tools_dir = path / 'tools'
        if tools_dir.exists():
            required_files.append(tools_dir / 'traci')
        
        # Validate existence of required components
        for dir_name in required_dirs:
            if not (path / dir_name).exists():
                return False
        
        # At least one binary should exist
        binary_found = any(f.exists() for f in required_files if 'bin' in str(f) or f.name.endswith(('.exe', binary_name)))
        
        return binary_found
    
    def _extract_sumo_home_from_binary(self, binary_path):
        """Extract SUMO_HOME from binary path."""
        binary_path = Path(binary_path)
        
        # Go up the directory tree to find SUMO_HOME
        for parent in binary_path.parents:
            if self._validate_sumo_home(str(parent)):
                return str(parent)
            
            # Check if parent/share/sumo exists (common on Linux/macOS)
            share_sumo = parent / 'share' / 'sumo'
            if self._validate_sumo_home(str(share_sumo)):
                return str(share_sumo)
        
        return None
    
    def install_sumo(self):
        """Install SUMO based on the current platform."""
        print(f"🚀 Installing SUMO for {self.system}...")
        
        if self.system == 'darwin':
            return self._install_sumo_macos()
        elif self.system == 'linux':
            return self._install_sumo_linux()
        elif self.system == 'windows':
            return self._install_sumo_windows()
        else:
            print(f"❌ Unsupported platform: {self.system}")
            return False
    
    def _install_sumo_macos(self):
        """Install SUMO on macOS using Homebrew."""
        print("📦 Installing SUMO on macOS...")
        
        # Check if Homebrew is installed
        if not shutil.which('brew'):
            print("❌ Homebrew not found. Please install Homebrew first:")
            print("   /bin/bash -c \"$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\"")
            return False
        
        try:
            # Update Homebrew
            print("🔄 Updating Homebrew...")
            subprocess.run(['brew', 'update'], check=True, capture_output=True)
            
            # Install SUMO
            print("📥 Installing SUMO via Homebrew...")
            result = subprocess.run(['brew', 'install', 'sumo'], check=True, capture_output=True, text=True)
            print("✅ SUMO installed successfully via Homebrew")
            
            # Detect installation
            sumo_home = self.detect_sumo_installation()
            if sumo_home:
                self._set_environment_variables(sumo_home)
                return True
            else:
                print("⚠️ SUMO installed but not detected. Manual configuration may be needed.")
                return False
                
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install SUMO via Homebrew: {e}")
            return False
    
    def _install_sumo_linux(self):
        """Install SUMO on Linux using package managers."""
        print("📦 Installing SUMO on Linux...")
        
        # Detect package manager
        package_managers = self.configs['linux']['package_managers']
        
        for pm, package in package_managers.items():
            if shutil.which(pm):
                print(f"📥 Installing SUMO using {pm}...")
                try:
                    if pm == 'apt':
                        # Update package list first
                        subprocess.run(['sudo', 'apt', 'update'], check=True)
                        subprocess.run(['sudo', 'apt', 'install', '-y'] + package.split(), check=True)
                    elif pm in ['yum', 'dnf']:
                        subprocess.run(['sudo', pm, 'install', '-y'] + package.split(), check=True)
                    elif pm == 'pacman':
                        subprocess.run(['sudo', 'pacman', '-S', '--noconfirm'] + package.split(), check=True)
                    elif pm == 'snap':
                        subprocess.run(['sudo', 'snap', 'install', package], check=True)
                    
                    print(f"✅ SUMO installed successfully via {pm}")
                    
                    # Detect installation
                    sumo_home = self.detect_sumo_installation()
                    if sumo_home:
                        self._set_environment_variables(sumo_home)
                        return True
                    
                except subprocess.CalledProcessError as e:
                    print(f"❌ Failed to install SUMO via {pm}: {e}")
                    continue
        
        print("❌ No supported package manager found for automatic installation")
        print("💡 Please install SUMO manually from: https://sumo.dlr.de/docs/Installing/index.html")
        return False
    
    def _install_sumo_windows(self):
        """Install SUMO on Windows by downloading from official source."""
        print("📦 Installing SUMO on Windows...")
        
        try:
            download_url = self.configs['windows']['download_url']
            extract_folder = self.configs['windows']['extract_folder']
            
            # Create temporary download directory
            temp_dir = self.project_root / 'temp_sumo_install'
            temp_dir.mkdir(exist_ok=True)
            
            zip_file = temp_dir / 'sumo.zip'
            
            print("📥 Downloading SUMO...")
            urllib.request.urlretrieve(download_url, zip_file)
            
            print("📦 Extracting SUMO...")
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            
            # Move to final location
            install_dir = Path.home() / 'sumo'
            if install_dir.exists():
                shutil.rmtree(install_dir)
            
            extracted_dir = temp_dir / extract_folder
            shutil.move(str(extracted_dir), str(install_dir))
            
            # Clean up
            shutil.rmtree(temp_dir)
            
            print(f"✅ SUMO installed to: {install_dir}")
            
            # Set environment variables
            self._set_environment_variables(str(install_dir))
            return True
            
        except Exception as e:
            print(f"❌ Failed to install SUMO on Windows: {e}")
            return False
    
    def _set_environment_variables(self, sumo_home):
        """Set SUMO environment variables."""
        print("🔧 Setting environment variables...")
        
        # Set SUMO_HOME
        os.environ['SUMO_HOME'] = sumo_home
        
        # Add SUMO binaries to PATH
        bin_dir = Path(sumo_home) / 'bin'
        if bin_dir.exists():
            current_path = os.environ.get('PATH', '')
            if str(bin_dir) not in current_path:
                os.environ['PATH'] = f"{bin_dir}{os.pathsep}{current_path}"
        
        # Create environment file for future sessions
        self._create_environment_file(sumo_home)
        
        print(f"✅ Environment variables set (SUMO_HOME={sumo_home})")
    
    def _create_environment_file(self, sumo_home):
        """Create environment configuration file."""
        env_file = self.project_root / '.env'
        
        env_content = f"""# SUMO Environment Configuration
# Auto-generated by setup_environment.py

SUMO_HOME={sumo_home}
"""
        
        with open(env_file, 'w') as f:
            f.write(env_content)
        
        print(f"📝 Environment file created: {env_file}")
    
    def verify_installation(self):
        """Verify SUMO installation and functionality."""
        print("🧪 Verifying SUMO installation...")
        
        tests = [
            ('SUMO binary', self.configs[self.system]['binary_name']),
            ('SUMO GUI binary', self.configs[self.system]['gui_binary_name']),
            ('NetConvert tool', self.configs[self.system]['tools_binary']),
        ]
        
        all_passed = True
        
        for test_name, binary in tests:
            if shutil.which(binary):
                print(f"✅ {test_name}: Found")
            else:
                print(f"❌ {test_name}: Not found")
                all_passed = False
        
        # Test SUMO_HOME
        sumo_home = os.environ.get('SUMO_HOME')
        if sumo_home and os.path.exists(sumo_home):
            print(f"✅ SUMO_HOME: {sumo_home}")
        else:
            print("❌ SUMO_HOME: Not set or invalid")
            all_passed = False
        
        # Test TraCI import
        try:
            import traci
            print("✅ TraCI: Importable")
        except ImportError:
            print("❌ TraCI: Import failed")
            all_passed = False
        
        # Test basic SUMO command
        try:
            binary_name = self.configs[self.system]['binary_name']
            result = subprocess.run([binary_name, '--help'], 
                                 capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                print("✅ SUMO execution: Working")
            else:
                print("❌ SUMO execution: Failed")
                all_passed = False
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.CalledProcessError):
            print("❌ SUMO execution: Failed")
            all_passed = False
        
        return all_passed
    
    def install_python_dependencies(self):
        """Install Python dependencies."""
        print("🐍 Installing Python dependencies...")
        
        requirements_file = self.project_root / 'requirements.txt'
        if not requirements_file.exists():
            print("❌ requirements.txt not found")
            return False
        
        try:
            subprocess.run([
                self.python_executable, '-m', 'pip', 'install', '-r', str(requirements_file)
            ], check=True)
            print("✅ Python dependencies installed")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install Python dependencies: {e}")
            return False
    
    def create_project_structure(self):
        """Create necessary project directories."""
        print("📁 Creating project structure...")
        
        directories = ['config', 'models', 'plots', 'logs']
        
        for directory in directories:
            dir_path = self.project_root / directory
            dir_path.mkdir(exist_ok=True)
            print(f"✅ Created directory: {directory}")
    
    def setup_complete_environment(self):
        """Complete environment setup process."""
        print("🚀 Starting complete SUMO environment setup...")
        print(f"🖥️  Platform: {self.system} ({self.architecture})")
        print(f"📁 Project root: {self.project_root}")
        print("-" * 60)
        
        # Step 1: Detect existing installation
        sumo_home = self.detect_sumo_installation()
        
        # Step 2: Install if not found
        if not sumo_home:
            print("⚠️ SUMO not detected. Attempting installation...")
            if not self.install_sumo():
                print("❌ SUMO installation failed")
                return False
            sumo_home = self.detect_sumo_installation()
        
        # Step 3: Set environment variables
        if sumo_home:
            self._set_environment_variables(sumo_home)
        
        # Step 4: Install Python dependencies
        if not self.install_python_dependencies():
            print("⚠️ Python dependencies installation failed")
        
        # Step 5: Create project structure
        self.create_project_structure()
        
        # Step 6: Verify installation
        print("-" * 60)
        if self.verify_installation():
            print("🎉 SUMO environment setup completed successfully!")
            print("\n💡 Next steps:")
            print("   1. Run: ./launch.sh train-demo")
            print("   2. Or: make demo")
            print("   3. Check: ./launch.sh status")
            return True
        else:
            print("❌ SUMO environment setup completed with issues")
            print("💡 Please check the errors above and install SUMO manually if needed")
            return False


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='SUMO Environment Setup')
    parser.add_argument('--detect-only', action='store_true',
                       help='Only detect existing installation')
    parser.add_argument('--install-only', action='store_true',
                       help='Only install SUMO')
    parser.add_argument('--verify-only', action='store_true',
                       help='Only verify installation')
    parser.add_argument('--python-deps-only', action='store_true',
                       help='Only install Python dependencies')
    
    args = parser.parse_args()
    
    setup = SUMOEnvironmentSetup()
    
    if args.detect_only:
        sumo_home = setup.detect_sumo_installation()
        if sumo_home:
            setup._set_environment_variables(sumo_home)
    elif args.install_only:
        setup.install_sumo()
    elif args.verify_only:
        setup.verify_installation()
    elif args.python_deps_only:
        setup.install_python_dependencies()
    else:
        setup.setup_complete_environment()


if __name__ == '__main__':
    main()
