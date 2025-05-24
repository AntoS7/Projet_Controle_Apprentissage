# Traffic Light Control with SUMO

This project implements a reinforcement learning environment for traffic light control using SUMO (Simulation of Urban MObility). The system models three intersections connected to each other, where each intersection can be controlled independently to minimize global queue lengths.

## Features

- **SARSA Agent**: Reinforcement learning agent for traffic light control
- **Network Model**: Three interconnected intersections with realistic traffic flow
- **State Representation**: Queue lengths on different lanes and time since last phase change
- **Action Space**: Different traffic light phases for each intersection
- **Reward Function**: Minimizes global queues with penalties for frequent phase changes
- **SUMO Integration**: Full integration with SUMO traffic simulation
- **GUI Support**: Visual simulation and training capabilities
- **Fallback System**: Mock environment when SUMO is unavailable

## Structure

```
├── src/
│   ├── environment/
│   │   ├── __init__.py
│   │   ├── traffic_env.py          # Main RL environment
│   │   └── intersection.py         # Intersection model
│   ├── network/
│   │   ├── __init__.py
│   │   ├── network_generator.py    # Generate SUMO network files
│   │   └── traffic_generator.py    # Generate traffic flows
│   └── utils/
│       ├── __init__.py
│       └── sumo_utils.py          # SUMO utility functions
├── config/
│   ├── network.net.xml            # SUMO network file
│   ├── routes.rou.xml             # Traffic routes
│   └── simulation.sumocfg         # SUMO configuration
├── examples/
│   ├── basic_simulation.py        # Basic simulation example
│   └── training_example.py        # RL training example
├── requirements.txt
└── README.md
```

## Installation

### 1. Install SUMO

Install SUMO from https://sumo.dlr.de/docs/Installing/index.html

**⚠️ Important**: The SUMO installation path should point to the `sumo/bin` folder containing the SUMO executables (`sumo`, `sumo-gui`, `netconvert`, etc.).

**macOS (Homebrew)**:
```bash
brew install sumo
# This installs SUMO to /opt/homebrew/opt/sumo/bin/
```

**Ubuntu/Debian**:
```bash
sudo apt-get install sumo sumo-tools sumo-doc
# This installs SUMO to /usr/bin/
```

**Windows**:
- Download and install from the SUMO website
- Add the SUMO `bin` directory to your system PATH
- Typical path: `C:\Program Files (x86)\Eclipse\Sumo\bin`

**Verify Installation**:
```bash
# Check if SUMO binaries are accessible
sumo --version
netconvert --help
```

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

## Usage

## Quick Start

### 1. Basic Training (Recommended)
```bash
# Complete setup and quick training
make setup
make train-quick
```

### 2. Visual Training with GUI
```bash
# Training with SUMO GUI for visual feedback
make train-gui

# Quick GUI demonstration
make demo-gui

# Interactive single episode test
make demo-quick
```

### 3. Manual Commands
```bash
# Headless training
python complete_sumo_training.py

# GUI training with custom episodes
python complete_sumo_training.py --gui --episodes 10

# Quick training with GUI
python quick_sumo_training.py --gui --episodes 5
```

## GUI Features

The SUMO GUI provides visual feedback during training:

- **Real-time Visualization**: Watch vehicles move through intersections
- **Traffic Light States**: See current phase (red/green) for each intersection
- **Queue Monitoring**: Visual indication of vehicle accumulations
- **Learning Progress**: Observe how the agent's decisions improve over time
- **Continuous Viewing**: SUMO GUI stays open between episodes for uninterrupted observation

### GUI Controls

- **Pause/Resume**: Space bar to pause/resume simulation
- **Speed Control**: Use +/- to adjust simulation speed
- **Zoom**: Mouse wheel to zoom in/out
- **Information**: Click on vehicles/intersections for details

### GUI Improvements

**🆕 Continuous Episode Viewing**: The GUI now keeps SUMO running between episodes, eliminating the disruptive close/restart cycle. This provides:
- Smoother visual training experience
- Uninterrupted observation of agent learning
- Faster episode transitions in GUI mode
- Better user experience for presentations and debugging

**🆕 Auto-Start Simulation**: When you press Enter between episodes, the simulation automatically starts running:
- No need to manually start the simulation in SUMO GUI
- Seamless transition between episodes
- Clear status messages showing simulation state
- Enhanced user experience for continuous training observation

## Training Scripts

| Script | Purpose | GUI Support | Episodes |
|--------|---------|-------------|----------|
| `complete_sumo_training.py` | Full setup and training | ✅ | 30 (default) |
| `quick_sumo_training.py` | Fast training with fallback | ✅ | 20 (default) |
| `demo_sumo_gui.py` | Interactive demonstration | ✅ | 3 (default) |

### Usage Examples

```bash
# Complete training with GUI
python complete_sumo_training.py --gui

# Fast training with custom episodes
python complete_sumo_training.py --episodes 50

# Quick demo with single episode
python demo_sumo_gui.py --single

# Fast mode training
python complete_sumo_training.py --fast
```

### Basic Simulation
```python
from src.environment.traffic_env import TrafficEnvironment

# Headless simulation
env = TrafficEnvironment(use_gui=False)

# GUI simulation
env = TrafficEnvironment(use_gui=True)

obs = env.reset()

for step in range(1000):
    # Random action for demonstration
    actions = env.action_space.sample()
    obs, reward, done, info = env.step(actions)
    
    if done:
        obs = env.reset()
```

### Training with RL
```python
from src.environment.traffic_env import TrafficEnvironment
from src.environment.sarsa_agent import SarsaAgent

# Create environment (with or without GUI)
env = TrafficEnvironment(use_gui=True)

# Create SARSA agent
agent = SarsaAgent(
    state_size=27,
    action_size=4,
    learning_rate=0.1,
    discount_factor=0.95
)

# Training loop
for episode in range(100):
    state, info = env.reset()
    action = agent.get_action(state, training=True)
    
    while True:
        next_state, reward, done, truncated, info = env.step(action)
        
        if done or truncated:
            agent.end_episode(reward)
            break
        else:
            next_action = agent.get_action(next_state, training=True)
            agent.update(reward, next_state, next_action, False)
            state, action = next_state, next_action
```

## Configuration

- Modify `config/simulation.sumocfg` to adjust simulation parameters
- Edit `src/network/network_generator.py` to change network topology
- Adjust reward function in `src/environment/traffic_env.py`

## Troubleshooting

### SUMO Path Issues

If you encounter errors like "SUMO not found" or "netconvert not found":

1. **Check SUMO Installation**: Verify that SUMO binaries are in your PATH:
   ```bash
   which sumo
   which netconvert
   ```

2. **Manual Path Configuration**: If SUMO is installed in a non-standard location, you may need to:
   - Add SUMO's `bin` directory to your system PATH
   - Update the hard-coded paths in the project files if necessary

3. **macOS Homebrew Users**: The project expects SUMO at `/opt/homebrew/opt/sumo/bin/`. If using Intel Mac, the path might be `/usr/local/opt/sumo/bin/`.

4. **Environment Variable**: You can set the `SUMO_HOME` environment variable to point to your SUMO installation directory:
   ```bash
   export SUMO_HOME="/path/to/your/sumo/installation"
   ```

### Common Error Messages

- **"netconvert not found"**: SUMO tools are not in PATH or not installed
- **"SUMO simulation failed"**: Check that `config/simulation.sumocfg` exists and is valid
- **"TraCI connection failed"**: SUMO binary might not be executable or path is incorrect

## License

MIT License
