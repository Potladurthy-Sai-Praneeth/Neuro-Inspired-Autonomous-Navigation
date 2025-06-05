"""
Example usage script showing how to switch between BTSP and SARSA algorithms.
"""

from main import main, run_inference
from config import Config

def train_btsp():
    """Train using BTSP algorithm."""
    print("Training with BTSP algorithm...")
    # Modify main.py to use BTSP
    # You can also directly use F1TenthTrainer here
    main()

def train_sarsa():
    """Train using SARSA algorithm."""
    print("Training with SARSA algorithm...")
    # Modify main.py to use SARSA
    # You can also directly use F1TenthTrainer here
    main()

def run_comparison():
    """Run both algorithms for comparison."""
    print("Running algorithm comparison...")
    
    # Train BTSP
    print("\n=== BTSP Training ===")
    # Set algorithm in main.py to 'BTSP' and run
    
    # Train SARSA  
    print("\n=== SARSA Training ===")
    # Set algorithm in main.py to 'SARSA' and run

def demonstrate_config_changes():
    """Demonstrate how to modify algorithm parameters."""
    print("Algorithm Parameters:")
    print(f"BTSP Parameters: {Config.BTSP_PARAMS}")
    print(f"SARSA Parameters: {Config.SARSA_PARAMS}")
    
    # Example of modifying parameters
    Config.BTSP_PARAMS['learning_rate'] = 5e-4
    Config.SARSA_PARAMS['discount_factor'] = 0.99
    
    print("\nModified Parameters:")
    print(f"BTSP Parameters: {Config.BTSP_PARAMS}")
    print(f"SARSA Parameters: {Config.SARSA_PARAMS}")

if __name__ == "__main__":
    # Demonstrate configuration
    demonstrate_config_changes()
    
    # Choose what to run
    print("\nChoose an option:")
    print("1. Train BTSP")
    print("2. Train SARSA") 
    print("3. Run inference")
    
    choice = input("Enter choice (1-3): ")
    
    if choice == "1":
        # Modify main.py ALGORITHM = 'BTSP' before running
        print("Set ALGORITHM = 'BTSP' in main.py and run main()")
    elif choice == "2":
        # Modify main.py ALGORITHM = 'SARSA' before running  
        print("Set ALGORITHM = 'SARSA' in main.py and run main()")
    elif choice == "3":
        weights_file = input("Enter path to weights file: ")
        algorithm = input("Enter algorithm (BTSP/SARSA): ")
        run_inference(algorithm=algorithm, weights_file=weights_file)
    else:
        print("Invalid choice")
