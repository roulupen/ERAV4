## Environemnt Setup

### Prerequisites

Install `uv` (if not already installed):
```bash
# On macOS and Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Or via pip
pip install uv
```

### Installation

```bash
# Clone or download the project
git clone <repository-url>
cd ERAV4/assignment-5

# Install dependencies and create virtual environment
uv sync

# Activate the virtual environment
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### Alternative: Install with development dependencies

```bash
# Install with development tools (jupyter, black, flake8, pytest)
uv sync --extra dev

# Or install everything
uv sync --extra all
```

## Running the Project

### Method 1: Jupyter Notebooks (Recommended)

```bash
# Activate virtual environment
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Start Jupyter
jupyter notebook

# Run any notebook:
# - MNIST_Classifier_v1.ipynb
# - MNIST_Classifier_v2.ipynb  
# - MNIST_Classifier_Final.ipynb
```

### Method 2: Using uv run (No activation needed)

```bash
# Run Jupyter with uv
uv run jupyter notebook

# Run any Python script
uv run python -c "import torch; print('PyTorch version:', torch.__version__)"
```

## Project Structure

```
ERAV4/
├── assignment-5/                    # Main project directory
│   ├── MNIST_Classifier_v1.ipynb   # Dropout in layers 3 & 4
│   ├── MNIST_Classifier_v2.ipynb   # No dropout in layers 3 & 4
│   ├── MNIST_Classifier_Final.ipynb # Batch normalization + strategic dropout
│   ├── TrainingHelper.py           # Training utilities
│   ├── Visualization.py            # Plotting utilities
│   ├── pyproject.toml              # Project configuration and dependencies
│   ├── uv.lock                     # Locked dependency versions
│   ├── checkpoints/                # Model checkpoints
│   ├── data/                       # MNIST dataset
│   └── *.png                       # Generated plots
└── README.md                       # This file
```
