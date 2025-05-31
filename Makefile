# SUMO Traffic Light Control Project Makefile
# Author: SUMO SARSA Training Project  
# Date: 2025-05-25

# Python interpreter
PYTHON = python3

# Project directories
CONFIG_DIR = config
MODELS_DIR = models
PLOTS_DIR = plots

# Default values
EPISODES = 30
GUI_SPEED = 1.0
PAUSE_FREQ = 50

.PHONY: all help install clean setup train train-gui train-fast train-gui-fast train-step demo test status

# Default target
all: help

help:
	@echo "SUMO Traffic Light Control Project"
	@echo "Available commands:"
	@echo "  help              - Show this help message"
	@echo "  install           - Install Python dependencies"
	@echo "  setup             - Setup project directories and check SUMO"
	@echo "  clean             - Clean generated files"
	@echo ""
	@echo "Training Commands:"
	@echo "  train             - Run headless training (50 episodes, 600 steps)"
	@echo "  train-gui         - Run GUI training (50 episodes, 600 steps)"
	@echo "  train-fast        - Run fast training (20 episodes)"
	@echo "  train-extended    - Run extended training (150 episodes)"
	@echo "  train-long        - Run long training (300 episodes)"
	@echo "  train-ultra       - Run ultra training (500 episodes)"
	@echo "  train-marathon    - Run marathon training (1000 episodes)"
	@echo ""
	@echo "Episode Duration Options:"
	@echo "  train-short       - Short episodes (400 steps)"
	@echo "  train-normal      - Normal episodes (600 steps, default)"
	@echo "  train-long-ep     - Long episodes (800 steps)"
	@echo "  train-extended-ep - Extended episodes (1200 steps)"
	@echo ""
	@echo "GUI Training:"
	@echo "  train-gui-fast    - Run fast GUI training (20 episodes)"
	@echo "  train-step        - Run step-by-step GUI training"
	@echo "  demo              - Run quick demo (2 episodes)"
	@echo ""
	@echo "Utilities:"
	@echo "  test              - Test SUMO installation"
	@echo "  test-reward       - Test enhanced reward function features"
	@echo "  status            - Show project status"

install:
	@echo "Installing dependencies..."
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt
	@echo "Dependencies installed"

setup:
	@echo "Setting up project..."
	@mkdir -p $(CONFIG_DIR) $(MODELS_DIR) $(PLOTS_DIR)
	@$(PYTHON) -c "from complete_sumo_training import check_sumo_installation; print('SUMO is installed' if check_sumo_installation() else 'SUMO not found')"
	@echo "Project setup complete"

clean:
	@echo "Cleaning project..."
	@rm -rf $(CONFIG_DIR)/*.xml $(CONFIG_DIR)/*.cfg 2>/dev/null || true
	@rm -rf $(MODELS_DIR)/*.pkl 2>/dev/null || true
	@rm -rf $(PLOTS_DIR)/*.png 2>/dev/null || true
	@find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	@find . -name "*.pyc" -delete 2>/dev/null || true
	@echo "Project cleaned"

train:
	@echo "Starting headless training (50 episodes, 600 steps per episode)..."
	$(PYTHON) complete_sumo_training.py --episodes 50

train-gui:
	@echo "Starting GUI training (50 episodes, 600 steps per episode)..."
	$(PYTHON) complete_sumo_training.py --gui --episodes 50 --gui-speed $(GUI_SPEED)

train-fast:
	@echo "Starting fast headless training (20 episodes)..."
	$(PYTHON) complete_sumo_training.py --fast

train-extended:
	@echo "Starting extended training (150 episodes)..."
	$(PYTHON) complete_sumo_training.py --extended

train-long:
	@echo "Starting long training (300 episodes)..."
	$(PYTHON) complete_sumo_training.py --long

train-ultra:
	@echo "Starting ultra training (500 episodes)..."
	$(PYTHON) complete_sumo_training.py --ultra

train-marathon:
	@echo "Starting marathon training (1000 episodes)..."
	$(PYTHON) complete_sumo_training.py --marathon

# Episode duration variants
train-short:
	@echo "Starting training with short episodes (400 steps)..."
	$(PYTHON) complete_sumo_training.py --episode-duration short

train-normal:
	@echo "Starting training with normal episodes (600 steps)..."
	$(PYTHON) complete_sumo_training.py --episode-duration normal

train-long-ep:
	@echo "Starting training with long episodes (800 steps)..."
	$(PYTHON) complete_sumo_training.py --episode-duration long

train-extended-ep:
	@echo "Starting training with extended episodes (1200 steps)..."
	$(PYTHON) complete_sumo_training.py --episode-duration extended

# Adaptive training
train-adaptive:
	@echo "Starting adaptive training with episode extension..."
	$(PYTHON) complete_sumo_training.py --adaptive-episodes

train-gui-fast:
	@echo "Starting fast GUI training..."
	$(PYTHON) complete_sumo_training.py --gui --fast --gui-speed $(GUI_SPEED)

train-step:
	@echo "Starting step-by-step GUI training..."
	$(PYTHON) complete_sumo_training.py --gui --gui-step --gui-speed 0.5 --gui-pause-freq $(PAUSE_FREQ) --episodes $(EPISODES)

demo:
	@echo "Running quick demo..."
	$(PYTHON) complete_sumo_training.py --gui --episodes 2 --gui-speed 1.5

test:
	@echo "Testing SUMO installation..."
	@$(PYTHON) -c "from complete_sumo_training import check_sumo_installation; print('SUMO Available' if check_sumo_installation() else 'SUMO Not Found')"

test-reward:
	@echo "Testing enhanced reward function features..."
	@echo "This will test the 11 advanced reward components including:"
	@echo "  - Progressive queue management with momentum tracking"
	@echo "  - Enhanced throughput optimization with flow consistency"
	@echo "  - Proactive congestion prevention scoring"
	@echo "  - Network harmony tracking and coordination"
	@echo "  - Peak traffic detection with adaptive multipliers"
	@echo "  - Bounded rewards for numerical stability"
	$(PYTHON) test_enhanced_reward.py

status:
	@echo "Project Status:"
	@echo "Python: $$($(PYTHON) --version)"
	@echo "SUMO: $$($(PYTHON) -c 'from complete_sumo_training import check_sumo_installation; print("Available" if check_sumo_installation() else "Not Found")')"
	@echo "Config files: $$(ls $(CONFIG_DIR)/*.xml $(CONFIG_DIR)/*.cfg 2>/dev/null | wc -l | tr -d ' ') files"
	@echo "Models: $$(ls $(MODELS_DIR)/*.pkl 2>/dev/null | wc -l | tr -d ' ') files"
	@echo "Plots: $$(ls $(PLOTS_DIR)/*.png 2>/dev/null | wc -l | tr -d ' ') files"
