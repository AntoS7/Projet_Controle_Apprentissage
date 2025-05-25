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
    echo "  train             - Run headless training (30 episodes)"
    echo "  train-gui         - Run GUI training (30 episodes)"
    echo "  train-fast        - Run fast training (10 episodes)"
    echo "  train-gui-fast    - Run fast GUI training (10 episodes)"
    echo "  train-step        - Run step-by-step GUI training"
    echo "  demo              - Run quick demo (2 episodes)"
    echo "  test              - Test SUMO installation"
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
        echo "Starting headless training..."
        $PYTHON complete_sumo_training.py --episodes 30
        ;;
    "train-gui")
        echo "Starting GUI training..."
        $PYTHON complete_sumo_training.py --gui --episodes 30 --gui-speed 1.0
        ;;
    "train-fast")
        echo "Starting fast training..."
        $PYTHON complete_sumo_training.py --fast
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
