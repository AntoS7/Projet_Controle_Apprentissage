#!/usr/bin/env python3
"""
Test SUMO daemon connection before training.
"""

import traci
import time
import subprocess
import os
import signal
import sys

def test_sumo_daemon():
    """Test SUMO daemon connection."""
    print('🔄 Testing SUMO TraCI connection...')
    
    # Start SUMO in daemon mode
    sumo_cmd = [
        'sumo', 
        '--remote-port', '8813', 
        '-c', 'config/simulation.sumocfg', 
        '--step-length', '1',
        '--no-step-log',
        '--no-warnings',
        '--quit-on-end'
    ]
    
    sumo_process = None
    
    try:
        print('   Starting SUMO server...')
        sumo_process = subprocess.Popen(
            sumo_cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            text=True
        )
        
        print('   Waiting for SUMO to initialize...')
        time.sleep(2)
        
        # Check if process is still running
        if sumo_process.poll() is not None:
            stdout, stderr = sumo_process.communicate()
            print(f'❌ SUMO process exited early: {stderr}')
            return False
        
        print('   Connecting to SUMO via TraCI...')
        traci.init(8813)
        print('✅ Connected to SUMO successfully!')
        
        print('   Running quick simulation test...')
        for step in range(5):
            traci.simulationStep()
            print(f'     Step {step} completed')
        
        print('✅ SUMO daemon is working correctly!')
        traci.close()
        return True
        
    except Exception as e:
        print(f'❌ SUMO connection failed: {e}')
        if sumo_process and sumo_process.poll() is None:
            try:
                stdout, stderr = sumo_process.communicate(timeout=1)
                if stderr:
                    print(f'   SUMO stderr: {stderr}')
            except subprocess.TimeoutExpired:
                pass
        return False
        
    finally:
        if sumo_process and sumo_process.poll() is None:
            print('   Cleaning up SUMO process...')
            sumo_process.terminate()
            try:
                sumo_process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                sumo_process.kill()
        print('   SUMO process cleaned up')

if __name__ == "__main__":
    success = test_sumo_daemon()
    sys.exit(0 if success else 1)
