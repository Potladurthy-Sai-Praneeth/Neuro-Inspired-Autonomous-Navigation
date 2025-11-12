# F1Tenth Neuromorphic Autonomous Navigation

This project implements autonomous navigation for neuromorphic-inspired algorithms on F1Tenth race tracks using OpenAI Gym environment.


# Requirements 
- Linux Os
- Python 3.10

### Local Setup Instructions

1. **Install Python**
```sh
# Add deadsnakes PPA for Python 3.10
sudo apt-get update
sudo apt-get install -y software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa

# Install Python 3.10 and required system packages
sudo apt-get update --fix-missing
sudo apt-get install -y \
    python3.10 \
    python3.10-distutils \
    python3.10-dev \
    python3-pip \
    python3.10-venv \
    git \
    build-essential \
    libgl1-mesa-dev \
    mesa-utils \
    libglu1-mesa-dev \
    fontconfig \
    libfreetype6-dev
```
2. **Create and activate a virtual environment:**
```sh
python3.10 -m venv f1_tenth_environment
source f1_tenth_environment/bin/activate
```

3. **Install pip and required Python packages:**
```sh
pip install pip==23.0.1
# Change the directory to the location where requiremenst.txt exist or use the <path>/requirements.txt
pip install -r requirements.txt
```

### Algorithm Selection

To switch between algorithms, simply modify the `ALGORITHM` variable in `main.py`:

```python
# In main.py
ALGORITHM = 'BTSP'    # or 'SARSA'
```

### Configuration

Modify hyperparameters in `config.py`:

```python
# Algorithm-specific parameters
BTSP_PARAMS = {
    'learning_rate': 1e-3,
    'et_decay_rate': 0.9,
    'is_decay_rate': 0.7,
    'max_weight': 5
}

SARSA_PARAMS = {
    'learning_rate': 1e-3,
    'discount_factor': 0.95,
    'decay_rate': 0.9
}
```

### Training

```bash
python main.py
```

### Inference

Uncomment and modify the inference section in `main.py`:

```python
run_inference(algorithm='BTSP', weights_file='weights.npy')
```


## File Descriptions

### `main.py`
- Entry point for training and inference
- Algorithm selection and configuration
- Map setup and path management

### `trainer.py`
- Main training loop supporting both algorithms
- Handles environment interaction and episode management
- Manages checkpointing and progress saving

### `algorithms.py`
- `BTSPAlgorithm`: Implements BTSP weight updates and eligibility traces
- `SARSAAlgorithm`: Implements SARSA weight updates and eligibility traces
- `ActionSelector`: Handles action selection with epsilon-greedy policy

### `environment.py`
- `EnvironmentManager`: Handles F1Tenth environment setup and map switching
- `StateProcessor`: Processes LiDAR data and manages state representation

### `reward.py`
- `Reward`: Calculates centerline, speed, and progress rewards
- Handles reward function parameters and track-specific calculations

### `index_selector.py`
- `IndexSelector`: Manages randomized training position selection
- Ensures balanced exploration of track positions

### `config.py`
- `Config`: Centralized configuration management
- Contains all hyperparameters and settings
- Provides utility functions for map management

### `explanation.md`
- Detailed explanation of the algorithms, reward functions, and weight updates.

# Demo
[Demo](https://drive.google.com/file/d/1pCYcmITV0wzrNcmtZDqXbTRRGbY_bX7R/view)
