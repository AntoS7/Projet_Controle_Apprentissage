# Makefile for SUMO Traffic Light Control Training
# Author: Traffic Control System
# Date: 2025

.PHONY: help setup check-sumo build-network install-deps clean train-quick train-robust train-complete evaluate test-agent gui-sumo

# Default target
help:
	@echo "=== SUMO Traffic Light Control Training ==="
	@echo ""
	@echo "Available targets:"
	@echo "  setup            - Complete setup: install dependencies and build network"
	@echo "  check-sumo       - Check if SUMO is properly installed"
	@echo "  install-deps     - Install Python dependencies"
	@echo "  build-network    - Build SUMO network from configuration files"
	@echo "  test-sumo        - Test SUMO simulation (headless)"
	@echo "  preflight        - Complete pre-flight check for SUMO training"
	@echo "  train-quick      - Run quick SUMO training with fallback"
	@echo "  train-real       - Run comprehensive real SUMO training"
	@echo "  train-complete   - Complete training with network setup"
	@echo "  train-gui        - Training with SUMO GUI (visual, auto-start episodes)"
	@echo "  demo-gui         - Interactive GUI demonstration (auto-start episodes)"
	@echo "  evaluate         - Evaluate trained SARSA agent"
	@echo "  gui-sumo         - Launch SUMO GUI for manual inspection"
	@echo "  clean            - Clean generated files"
	@echo "  help             - Show this help message"
	@echo ""
	@echo "🎮 GUI Features:"
	@echo "  - SUMO stays open between episodes (no restart)"
	@echo "  - Press Enter to auto-start next episode"
	@echo "  - Smooth visual training experience"
	@echo ""

# Variables
PYTHON = python3
PIP = pip3
CONFIG_DIR = config
NETWORK_FILE = $(CONFIG_DIR)/network.net.xml
EDGE_FILE = $(CONFIG_DIR)/network.edg.xml
NODE_FILE = $(CONFIG_DIR)/network.nod.xml
SUMO_CONFIG = $(CONFIG_DIR)/simulation.sumocfg

# Complete setup target
setup: check-sumo install-deps build-network
	@echo "✅ Setup complete! You can now run training scripts."

# Check SUMO installation
check-sumo:
	@echo "🔍 Checking SUMO installation..."
	@if [ -f "/opt/homebrew/opt/sumo/bin/sumo" ]; then \
		echo "✅ SUMO found at /opt/homebrew/opt/sumo/bin/sumo"; \
	else \
		echo "❌ SUMO not found at /opt/homebrew/opt/sumo/bin/sumo"; \
		echo "   - macOS: brew install sumo"; \
		echo "   - Ubuntu: sudo apt-get install sumo sumo-tools sumo-doc"; \
		exit 1; \
	fi
	@if [ -f "/opt/homebrew/opt/sumo/bin/netconvert" ]; then \
		echo "✅ netconvert found at /opt/homebrew/opt/sumo/bin/netconvert"; \
	else \
		echo "❌ netconvert not found at /opt/homebrew/opt/sumo/bin/netconvert"; \
		exit 1; \
	fi
	@if command -v traci >/dev/null 2>&1 || $(PYTHON) -c "import traci" 2>/dev/null; then \
		echo "✅ TraCI available"; \
	else \
		echo "⚠️  TraCI not found, will install with dependencies"; \
	fi

# Install Python dependencies
install-deps:
	@echo "📦 Installing Python dependencies..."
	@if [ -f requirements.txt ]; then \
		$(PIP) install -r requirements.txt; \
		echo "✅ Dependencies installed"; \
	else \
		echo "⚠️  requirements.txt not found, installing basic packages..."; \
		$(PIP) install sumo-tools traci numpy gymnasium matplotlib networkx; \
	fi

# Build SUMO network
build-network: $(NETWORK_FILE)

$(NETWORK_FILE): $(EDGE_FILE) $(NODE_FILE)
	@echo "🏗️  Building SUMO network..."
	@if [ ! -f $(EDGE_FILE) ] || [ ! -f $(NODE_FILE) ]; then \
		echo "❌ Missing network definition files: $(EDGE_FILE) or $(NODE_FILE)"; \
		echo "   Generating basic network files..."; \
		$(PYTHON) -c "import sys; sys.path.append('src'); from network.network_generator import generate_simple_network; generate_simple_network()"; \
	fi
	@echo "   Converting network files to SUMO format..."
	/opt/homebrew/opt/sumo/bin/netconvert --node-files=$(NODE_FILE) --edge-files=$(EDGE_FILE) --output-file=$(NETWORK_FILE)
	@if [ -f $(NETWORK_FILE) ]; then \
		echo "✅ Network built successfully: $(NETWORK_FILE)"; \
	else \
		echo "❌ Failed to build network"; \
		exit 1; \
	fi

# Quick training with automatic fallback
train-quick: preflight
	@echo "🚀 Starting quick SUMO training..."
	$(PYTHON) quick_sumo_training.py

# Comprehensive real SUMO training
train-real: preflight
	@echo "🚀 Starting comprehensive real SUMO training..."
	$(PYTHON) real_sumo_training.py

# Evaluate trained agent
evaluate: 
	@echo "📊 Evaluating SARSA agent..."
	$(PYTHON) evaluate_sarsa.py

# Test SUMO simulation (headless)
test-sumo: build-network
	@echo "🧪 Testing SUMO simulation..."
	@if [ -f $(SUMO_CONFIG) ]; then \
		echo "   Running headless SUMO test..."; \
		/opt/homebrew/opt/sumo/bin/sumo -c $(SUMO_CONFIG) --no-step-log --no-warnings --duration-log.disable true --step-length 1 --quit-on-end true --start true -e 10; \
		if [ $$? -eq 0 ]; then \
			echo "✅ SUMO simulation test passed"; \
		else \
			echo "❌ SUMO simulation test failed"; \
			exit 1; \
		fi; \
	else \
		echo "❌ SUMO configuration file not found: $(SUMO_CONFIG)"; \
		exit 1; \
	fi

# Pre-flight check: ensure SUMO is ready for training
preflight: build-network test-sumo
	@echo "✈️  Pre-flight check complete - SUMO is ready for training!"

# Training with GUI (for debugging)
train-gui: preflight
	@echo "🎮 Starting SUMO training with GUI..."
	$(PYTHON) complete_sumo_training.py --gui

# Complete training setup
train-complete: 
	@echo "🚀 Starting complete SUMO training..."
	$(PYTHON) complete_sumo_training.py

# Complete training with GUI
train-complete-gui:
	@echo "🎮 Starting complete SUMO training with GUI..."
	$(PYTHON) complete_sumo_training.py --gui --episodes 10

# Quick training with GUI
train-quick-gui: preflight
	@echo "🎮 Starting quick SUMO training with GUI..."
	$(PYTHON) quick_sumo_training.py --gui --episodes 5

# Interactive GUI demonstration
demo-gui: build-network
	@echo "🎮 Starting interactive GUI demonstration..."
	$(PYTHON) demo_sumo_gui.py

# Quick GUI demo
demo-quick:
	@echo "🔍 Running quick GUI demonstration..."
	$(PYTHON) demo_sumo_gui.py --single

# Launch SUMO GUI for manual inspection
gui-sumo: build-network
	@echo "🖥️  Launching SUMO GUI for manual inspection..."
	@if [ -f $(SUMO_CONFIG) ]; then \
		/opt/homebrew/opt/sumo/bin/sumo-gui -c $(SUMO_CONFIG); \
	else \
		echo "❌ SUMO configuration file not found: $(SUMO_CONFIG)"; \
		exit 1; \
	fi

# Clean generated files
clean:
	@echo "🧹 Cleaning generated files..."
	@rm -f $(CONFIG_DIR)/network.net.xml
	@rm -f $(CONFIG_DIR)/*.log
	@rm -f tripinfo.xml
	@rm -f *.log
	@rm -f models/*.pkl
	@rm -f *.png
	@rm -rf __pycache__
	@rm -rf src/__pycache__
	@rm -rf src/*/__pycache__
	@echo "✅ Cleanup complete"

# Show current status
status:
	@echo "📋 Project Status:"
	@echo "  SUMO Network: $$([ -f $(NETWORK_FILE) ] && echo '✅ Built' || echo '❌ Missing')"
	@echo "  Config Files: $$([ -f $(SUMO_CONFIG) ] && echo '✅ Present' || echo '❌ Missing')"
	@echo "  Dependencies: $$($(PYTHON) -c 'import traci, numpy, matplotlib; print("✅ Installed")' 2>/dev/null || echo '❌ Missing')"
	@echo "  Trained Models: $$(ls models/*.pkl 2>/dev/null | wc -l | tr -d ' ') files"
	@echo ""

# Quick start for new users
quickstart: setup train-quick evaluate
	@echo ""
	@echo "🎉 Quickstart complete!"
	@echo "Your SARSA agent has been trained and evaluated."
	@echo ""
	@echo "Next steps:"
	@echo "  - Run 'make train-real' for comprehensive training"
	@echo "  - Run 'make gui-sumo' to inspect the simulation visually"
	@echo "  - Check 'models/' directory for saved agent models"
	@echo ""

# Show available training options
training-options:
	@echo "🎯 Available Training Options:"
	@echo ""
	@echo "BASIC TRAINING:"
	@echo "  make train-quick        - Fast training with automatic fallback"
	@echo "  make train-real         - Comprehensive training with real SUMO"
	@echo "  make train-complete     - Complete setup and training"
	@echo ""
	@echo "GUI TRAINING (Visual):"
	@echo "  make train-gui          - Complete training with GUI"
	@echo "  make train-quick-gui    - Quick training with GUI (5 episodes)"
	@echo "  make train-complete-gui - Complete training with GUI (10 episodes)"
	@echo ""
	@echo "DEMONSTRATION:"
	@echo "  make demo-gui           - Interactive GUI demonstration"
	@echo "  make demo-quick         - Quick single episode GUI test"
	@echo "  make gui-sumo           - Manual SUMO GUI inspection"
	@echo ""
	@echo "All training scripts will automatically:"
	@echo "  ✓ Check SUMO installation"
	@echo "  ✓ Build network if needed"
	@echo "  ✓ Fall back to mock environment if SUMO fails"
	@echo "  ✓ Save trained models and results"
	@echo ""
