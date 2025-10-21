# PLMTune

## Project Description

PLMTune is a project focused on protein language model fine-tuning for intrinsically disordered regions (IDR) and variant effect prediction (VEP).

## Project Architecture

```
PLMTune/
│
├── configs/                    # Configuration files for experiments and models
│
├── src/                        # Source code
│   └── idr_vep/               # Main package
│       ├── data/              # Data loading and preprocessing
│       ├── models/            # Model architectures and definitions
│       ├── train/             # Training scripts and logic
│       ├── eval/              # Evaluation metrics and scripts
│       ├── interp/            # Model interpretation and analysis
│       └── utils/             # Utility functions and helpers
│
├── scripts/                   # Standalone scripts for various tasks
│
├── sbatch/                    # SLURM batch job scripts for HPC
│
├── logs/                      # Training logs and outputs
│
├── data/                      # Data directory
│   ├── raw/                   # Raw, unprocessed data
│   └── processed/             # Processed and cleaned data
│
├── notebooks/                 # Jupyter notebooks for exploration and analysis
│
└── .vscode/                   # VS Code settings and configurations

```

## Getting Started

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/PLMTune.git
cd PLMTune

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Usage

Coming soon...

## Requirements

- Python 3.8+
- PyTorch
- Additional dependencies listed in `requirements.txt`

## License

Coming soon...

## Contributors

Coming soon...
