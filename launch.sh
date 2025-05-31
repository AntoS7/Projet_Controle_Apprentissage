#!/bin/bash

# SUMO Traffic Light Control Project Launcher
# Simple version for immediate use

PYTHON="python3"

print_help() {
    echo "SUMO Traffic Light Control Project Launcher"
    echo "Usage: ./launch.sh [command]"
    echo ""
    echo "Available commands:"
    echo "  help              - Show this help message"
    echo "  setup             - Setup project directories"
    echo "  install           - Install Python dependencies"
    echo ""
    echo "Training Commands:"
    echo "  train             - Run headless training (50 episodes, 600 steps)"
    echo "  train-gui         - Run GUI training (50 episodes, 600 steps)"
    echo "  train-fast        - Run fast training (20 episodes)"
    echo "  train-extended    - Run extended training (150 episodes)"
    echo "  train-long        - Run long training (300 episodes)"
    echo "  train-ultra       - Run ultra training (500 episodes)"
    echo "  train-marathon    - Run marathon training (1000 episodes)"
    echo ""
    echo "Episode Duration:"
    echo "  train-short       - Short episodes (400 steps)"
    echo "  train-long-ep     - Long episodes (800 steps)"
    echo "  train-extended-ep - Extended episodes (1200 steps)"
    echo ""
    echo "GUI & Demo:"
    echo "  train-gui-fast    - Run fast GUI training (20 episodes)"
    echo "  train-step        - Run step-by-step GUI training"
    echo "  demo              - Run quick demo (2 episodes)"
    echo ""
    echo "Utilities:"
    echo "  test              - Test SUMO installation"
    echo "  test-reward       - Test enhanced reward function features"
    echo "  status            - Show project status"
    echo "  clean             - Clean generated files"
    echo "  plots             - Show plots directory"
    echo "  models            - Show saved models"
}

case "${1:-help}" in
    "help"|"-h"|"--help")
        print_help
        ;;
    "setup")
        echo "Setting up project..."
        mkdir -p config models plots
        echo "✓ Directories created"
        ;;
    "install")
        echo "Installing dependencies..."
        $PYTHON -m pip install --upgrade pip
        $PYTHON -m pip install -r requirements.txt
        echo "✓ Dependencies installed"
        ;;
    "train")
        echo "Starting headless training (50 episodes, 600 steps)..."
        $PYTHON complete_sumo_training.py --episodes 50
        ;;
    "train-gui")
        echo "Starting GUI training (50 episodes, 600 steps)..."
        $PYTHON complete_sumo_training.py --gui --episodes 50 --gui-speed 1.0
        ;;
    "train-fast")
        echo "Starting fast training (20 episodes)..."
        $PYTHON complete_sumo_training.py --fast
        ;;
    "train-extended")
        echo "Starting extended training (150 episodes)..."
        $PYTHON complete_sumo_training.py --extended
        ;;
    "train-long")
        echo "Starting long training (300 episodes)..."
        $PYTHON complete_sumo_training.py --long
        ;;
    "train-ultra")
        echo "Starting ultra training (500 episodes)..."
        $PYTHON complete_sumo_training.py --ultra
        ;;
    "train-marathon")
        echo "Starting marathon training (1000 episodes)..."
        $PYTHON complete_sumo_training.py --marathon
        ;;
    "train-short")
        echo "Starting training with short episodes (400 steps)..."
        $PYTHON complete_sumo_training.py --episode-duration short
        ;;
    "train-long-ep")
        echo "Starting training with long episodes (800 steps)..."
        $PYTHON complete_sumo_training.py --episode-duration long
        ;;
    "train-extended-ep")
        echo "Starting training with extended episodes (1200 steps)..."
        $PYTHON complete_sumo_training.py --episode-duration extended
        ;;
    "train-gui-fast")
        echo "Starting fast GUI training..."
        $PYTHON complete_sumo_training.py --gui --fast --gui-speed 1.0
        ;;
    "train-step")
        echo "Starting step-by-step GUI training..."
        $PYTHON complete_sumo_training.py --gui --gui-step --gui-speed 0.5 --gui-pause-freq 50
        ;;
    "demo")
        echo "Running quick demo..."
        $PYTHON complete_sumo_training.py --gui --episodes 2 --gui-speed 1.5
        ;;
    "test")
        echo "Testing SUMO installation..."
        $PYTHON -c "from complete_sumo_training import check_sumo_installation; print('✓ SUMO Available' if check_sumo_installation() else '✗ SUMO Not Found')"
        ;;
    "status")
        echo "Project Status:"
        echo "Python: $($PYTHON --version)"
        echo "Config files: $(ls config/*.xml config/*.cfg 2>/dev/null | wc -l | tr -d ' ') files"
        echo "Models: $(ls models/*.pkl 2>/dev/null | wc -l | tr -d ' ') files"
        echo "Plots: $(ls plots/*.png 2>/dev/null | wc -l | tr -d ' ') files"
        ;;
    "clean")
        echo "Cleaning project..."
        rm -f config/*.xml config/*.cfg 2>/dev/null || true
        rm -f models/*.pkl 2>/dev/null || true
        rm -f plots/*.png 2>/dev/null || true
        find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
        find . -name "*.pyc" -delete 2>/dev/null || true
        echo "✓ Project cleaned"
        ;;
    "plots")
        echo "Plot files:"
        ls -la plots/*.png 2>/dev/null || echo "No plots found"
        open plots 2>/dev/null || true
        ;;
    "models")
        echo "Saved models:"
        ls -la models/*.pkl 2>/dev/null || echo "No models found"
        ;;
    *)
        echo "Unknown command: $1"
        echo "Use './launch.sh help' for available commands"
        exit 1
        ;;
esac
