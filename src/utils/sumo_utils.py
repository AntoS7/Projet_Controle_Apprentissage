"""
SUMO utility functions for traffic simulation.

This module provides utility functions for working with SUMO simulation,
including connection management and data extraction.
"""

import traci
import subprocess
import time
import os
import platform
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import xml.etree.ElementTree as ET


def setup_sumo():
    """Setup SUMO environment variables with enhanced cross-platform detection."""
    import platform
    import shutil
    from pathlib import Path
    
    # Check if already set
    if 'SUMO_HOME' in os.environ:
        sumo_home = os.environ['SUMO_HOME']
        if os.path.exists(sumo_home):
            print(f"SUMO_HOME already set to: {sumo_home}")
            return sumo_home
        else:
            print(f"Warning: SUMO_HOME set to non-existent path: {sumo_home}")
    
    # Load from .env file if available
    env_file = Path(__file__).parent.parent.parent / '.env'
    if env_file.exists():
        try:
            with open(env_file, 'r') as f:
                for line in f:
                    if line.startswith('SUMO_HOME='):
                        sumo_home = line.split('=', 1)[1].strip()
                        if os.path.exists(sumo_home):
                            os.environ['SUMO_HOME'] = sumo_home
                            print(f"SUMO_HOME loaded from .env: {sumo_home}")
                            return sumo_home
        except Exception:
            pass
    
    # Platform-specific detection
    system = platform.system().lower()
    
    if system == 'windows':
        sumo_paths = [
            'C:/Program Files (x86)/Eclipse/Sumo',
            'C:/Program Files/Eclipse/Sumo',
            'C:/sumo',
            str(Path.home() / 'sumo')
        ]
        binary_name = 'sumo.exe'
    elif system == 'darwin':  # macOS
        sumo_paths = [
            '/opt/homebrew/share/sumo',
            '/opt/homebrew/opt/sumo/share/sumo',
            '/usr/local/share/sumo',
            '/usr/local/opt/sumo/share/sumo',
            '/usr/share/sumo'
        ]
        binary_name = 'sumo'
    else:  # Linux
        sumo_paths = [
            '/usr/share/sumo',
            '/usr/local/share/sumo',
            '/opt/sumo',
            str(Path.home() / '.local/share/sumo')
        ]
        binary_name = 'sumo'
    
    # Check common installation paths
    for path in sumo_paths:
        if os.path.exists(path) and _validate_sumo_path(path):
            os.environ['SUMO_HOME'] = path
            print(f"SUMO_HOME detected and set to: {path}")
            return path
    
    # Try to detect from binary location
    sumo_binary = shutil.which(binary_name)
    if sumo_binary:
        sumo_home = _extract_sumo_home_from_binary(sumo_binary)
        if sumo_home:
            os.environ['SUMO_HOME'] = sumo_home
            print(f"SUMO_HOME detected from binary path: {sumo_home}")
            return sumo_home
    
    # Final attempt: use which/where command
    try:
        if system == 'windows':
            result = subprocess.run(['where', binary_name], capture_output=True, text=True)
        else:
            result = subprocess.run(['which', binary_name], capture_output=True, text=True)
        
        if result.returncode == 0:
            binary_path = result.stdout.strip().split('\n')[0]
            sumo_home = _extract_sumo_home_from_binary(binary_path)
            if sumo_home:
                os.environ['SUMO_HOME'] = sumo_home
                print(f"SUMO_HOME detected from system path: {sumo_home}")
                return sumo_home
    except Exception:
        pass
    
    print("Warning: SUMO_HOME not found. Run 'python setup_environment.py' to install SUMO.")
    return None


def _validate_sumo_path(path):
    """Validate if a path contains a valid SUMO installation."""
    if not os.path.exists(path):
        return False
    
    path = Path(path)
    required_items = ['tools']  # At minimum, tools directory should exist
    
    for item in required_items:
        if not (path / item).exists():
            return False
    
    return True


def _extract_sumo_home_from_binary(binary_path):
    """Extract SUMO_HOME from binary path."""
    binary_path = Path(binary_path)
    
    # Common patterns for SUMO installation
    for parent in binary_path.parents:
        # Check if this directory contains SUMO structure
        if _validate_sumo_path(str(parent)):
            return str(parent)
        
        # Check share/sumo subdirectory (common on Linux/macOS)
        share_sumo = parent / 'share' / 'sumo'
        if _validate_sumo_path(str(share_sumo)):
            return str(share_sumo)
    
    return None


class SumoConnection:
    """Manages connection to SUMO simulation."""
    
    def __init__(self, sumo_cfg: str, use_gui: bool = False, step_length: float = 1.0):
        """
        Initialize SUMO connection.
        
        Args:
            sumo_cfg: Path to SUMO configuration file
            use_gui: Whether to use SUMO GUI
            step_length: Simulation step length in seconds
        """
        self.sumo_cfg = sumo_cfg
        self.use_gui = use_gui
        self.step_length = step_length
        self.is_connected = False
        self.sumo_process = None

    def start(self, port: int = 8813):
        """Start SUMO simulation."""
        if self.is_connected:
            self.close()
        
        try:
            # Use sumolib to find the correct SUMO binary
            import sumolib
            sumo_binary = sumolib.checkBinary('sumo-gui' if self.use_gui else 'sumo')
            
            sumo_cmd = [
                sumo_binary,
                "-c", self.sumo_cfg,
                "--remote-port", str(port),
                "--step-length", str(self.step_length),
                "--waiting-time-memory", "1000",
                "--time-to-teleport", "-1",
                "--no-step-log", "true",
                "--no-warnings", "true"
            ]
            
            # Start SUMO process
            self.sumo_process = subprocess.Popen(sumo_cmd, 
                                               stdout=subprocess.DEVNULL,
                                               stderr=subprocess.DEVNULL)
            time.sleep(2)  # Give SUMO time to start
            
            # Connect via TraCI
            traci.init(port)
            self.is_connected = True
            
        except Exception as e:
            raise RuntimeError(f"Failed to start SUMO: {e}")
    
    def close(self):
        """Close SUMO simulation."""
        if self.is_connected:
            try:
                traci.close()
            except:
                pass
            self.is_connected = False
        
        if self.sumo_process:
            try:
                self.sumo_process.terminate()
                self.sumo_process.wait(timeout=5)
            except:
                try:
                    self.sumo_process.kill()
                except:
                    pass
            self.sumo_process = None
    
    def step(self):
        """Advance simulation by one step."""
        if self.is_connected:
            traci.simulationStep()
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


def get_traffic_light_state(tl_id: str) -> Dict[str, str]:
    """
    Get current state of a traffic light.
    
    Args:
        tl_id: Traffic light ID
        
    Returns:
        Dictionary with traffic light information
    """
    try:
        return {
            'current_phase': traci.trafficlight.getPhase(tl_id),
            'current_program': traci.trafficlight.getProgram(tl_id),
            'next_switch': traci.trafficlight.getNextSwitch(tl_id),
            'controlled_lanes': traci.trafficlight.getControlledLanes(tl_id),
            'controlled_links': traci.trafficlight.getControlledLinks(tl_id)
        }
    except:
        return {}


def get_lane_queue_length(lane_id: str) -> int:
    """
    Get number of vehicles waiting in a lane.
    
    Args:
        lane_id: Lane identifier
        
    Returns:
        Number of halting vehicles
    """
    try:
        return traci.lane.getLastStepHaltingNumber(lane_id)
    except:
        return 0


def get_lane_vehicle_count(lane_id: str) -> int:
    """
    Get total number of vehicles in a lane.
    
    Args:
        lane_id: Lane identifier
        
    Returns:
        Number of vehicles
    """
    try:
        return traci.lane.getLastStepVehicleNumber(lane_id)
    except:
        return 0


def get_lane_waiting_time(lane_id: str) -> float:
    """
    Get total waiting time of vehicles in a lane.
    
    Args:
        lane_id: Lane identifier
        
    Returns:
        Total waiting time in seconds
    """
    try:
        return traci.lane.getWaitingTime(lane_id)
    except:
        return 0.0


def get_intersection_metrics(tl_id: str) -> Dict[str, float]:
    """
    Get comprehensive metrics for an intersection.
    
    Args:
        tl_id: Traffic light ID
        
    Returns:
        Dictionary with intersection metrics
    """
    metrics = {
        'total_queue': 0,
        'total_vehicles': 0,
        'total_waiting_time': 0.0,
        'average_speed': 0.0,
        'lane_metrics': {}
    }
    
    try:
        controlled_lanes = traci.trafficlight.getControlledLanes(tl_id)
        unique_lanes = list(set(controlled_lanes))  # Remove duplicates
        
        total_speed = 0.0
        lane_count = 0
        
        for lane in unique_lanes:
            queue = get_lane_queue_length(lane)
            vehicles = get_lane_vehicle_count(lane)
            waiting_time = get_lane_waiting_time(lane)
            
            try:
                mean_speed = traci.lane.getLastStepMeanSpeed(lane)
            except:
                mean_speed = 0.0
            
            metrics['lane_metrics'][lane] = {
                'queue': queue,
                'vehicles': vehicles,
                'waiting_time': waiting_time,
                'mean_speed': mean_speed
            }
            
            metrics['total_queue'] += queue
            metrics['total_vehicles'] += vehicles
            metrics['total_waiting_time'] += waiting_time
            
            if vehicles > 0:
                total_speed += mean_speed * vehicles
                lane_count += vehicles
        
        # Calculate average speed
        if lane_count > 0:
            metrics['average_speed'] = total_speed / lane_count
            
    except Exception as e:
        print(f"Error getting metrics for {tl_id}: {e}")
    
    return metrics


def set_traffic_light_phase(tl_id: str, phase_index: int):
    """
    Set traffic light to specific phase.
    
    Args:
        tl_id: Traffic light ID
        phase_index: Phase index to set
    """
    try:
        traci.trafficlight.setPhase(tl_id, phase_index)
    except Exception as e:
        print(f"Error setting phase for {tl_id}: {e}")


def get_simulation_time() -> float:
    """Get current simulation time."""
    try:
        return traci.simulation.getTime()
    except:
        return 0.0


def get_departed_vehicles() -> int:
    """Get number of vehicles that departed in last step."""
    try:
        return traci.simulation.getDepartedNumber()
    except:
        return 0


def get_arrived_vehicles() -> int:
    """Get number of vehicles that arrived in last step."""
    try:
        return traci.simulation.getArrivedNumber()
    except:
        return 0


def get_network_statistics() -> Dict[str, float]:
    """
    Get network-wide statistics.
    
    Returns:
        Dictionary with network statistics
    """
    try:
        return {
            'time': get_simulation_time(),
            'departed': get_departed_vehicles(),
            'arrived': get_arrived_vehicles(),
            'running': traci.simulation.getMinExpectedNumber(),
            'waiting': sum(get_lane_queue_length(lane) 
                          for lane in traci.lane.getIDList()),
            'total_vehicles': sum(get_lane_vehicle_count(lane) 
                                for lane in traci.lane.getIDList())
        }
    except:
        return {}


def create_sumo_config(network_file: str, 
                      routes_file: str,
                      output_file: str = "config/simulation.sumocfg",
                      begin: int = 0,
                      end: int = 3600) -> str:
    """
    Create SUMO configuration file.
    
    Args:
        network_file: Path to network file
        routes_file: Path to routes file
        output_file: Path for output config file
        begin: Simulation start time
        end: Simulation end time
        
    Returns:
        Path to created config file
    """
    # Create root element
    root = ET.Element("configuration")
    root.set("xmlns:xsi", "http://www.w3.org/2001/XMLSchema-instance")
    root.set("xsi:noNamespaceSchemaLocation", "http://sumo.dlr.de/xsd/sumoConfiguration.xsd")
    
    # Input section
    input_elem = ET.SubElement(root, "input")
    net_elem = ET.SubElement(input_elem, "net-file")
    net_elem.set("value", network_file)
    route_elem = ET.SubElement(input_elem, "route-files")
    route_elem.set("value", routes_file)
    
    # Time section
    time_elem = ET.SubElement(root, "time")
    begin_elem = ET.SubElement(time_elem, "begin")
    begin_elem.set("value", str(begin))
    end_elem = ET.SubElement(time_elem, "end")
    end_elem.set("value", str(end))
    
    # Processing section
    processing_elem = ET.SubElement(root, "processing")
    time_to_teleport = ET.SubElement(processing_elem, "time-to-teleport")
    time_to_teleport.set("value", "-1")
    
    # Write to file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    tree = ET.ElementTree(root)
    tree.write(output_file, encoding='utf-8', xml_declaration=True)
    
    return output_file


if __name__ == "__main__":
    # Example usage
    print("SUMO utilities loaded successfully")
    print("Available functions:")
    print("- SumoConnection: Manage SUMO simulation")
    print("- get_traffic_light_state: Get TL state")
    print("- get_intersection_metrics: Get intersection data")
    print("- create_sumo_config: Create config file")
