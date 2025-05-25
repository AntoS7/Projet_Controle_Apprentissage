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
	@echo "  train             - Run headless training (30 episodes)"
	@echo "  train-gui         - Run GUI training (30 episodes)"
	@echo "  train-fast        - Run fast training (10 episodes)"
	@echo "  train-gui-fast    - Run fast GUI training (10 episodes)"
	@echo "  train-step        - Run step-by-step GUI training"
	@echo "  demo              - Run quick demo (2 episodes)"
	@echo "  test              - Test SUMO installation"
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
	@echo "Starting headless training ($(EPISODES) episodes)..."
	$(PYTHON) complete_sumo_training.py --episodes $(EPISODES)

train-gui:
	@echo "Starting GUI training ($(EPISODES) episodes)..."
	$(PYTHON) complete_sumo_training.py --gui --episodes $(EPISODES) --gui-speed $(GUI_SPEED)

train-fast:
	@echo "Starting fast headless training..."
	$(PYTHON) complete_sumo_training.py --fast

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

status:
	@echo "Project Status:"
	@echo "Python: $$($(PYTHON) --version)"
	@echo "SUMO: $$($(PYTHON) -c 'from complete_sumo_training import check_sumo_installation; print("Available" if check_sumo_installation() else "Not Found")')"
	@echo "Config files: $$(ls $(CONFIG_DIR)/*.xml $(CONFIG_DIR)/*.cfg 2>/dev/null | wc -l | tr -d ' ') files"
	@echo "Models: $$(ls $(MODELS_DIR)/*.pkl 2>/dev/null | wc -l | tr -d ' ') files"
	@echo "Plots: $$(ls $(PLOTS_DIR)/*.png 2>/dev/null | wc -l | tr -d ' ') files"
