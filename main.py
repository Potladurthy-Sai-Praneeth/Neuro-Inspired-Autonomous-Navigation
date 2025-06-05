"""
Main entry point for F1Tenth navigation training with algorithm selection.
"""

import os
import numpy as np
from trainer import F1TenthTrainer
from config import Config


def main():
    """Main function to run training with specified algorithm."""
    
    # Algorithm selection - Change this to switch between algorithms
    ALGORITHM = 'BTSP'  # Options: 'BTSP' or 'SARSA'
    
    print(f"Starting training with {ALGORITHM} algorithm...")
    
    # Get map paths and data
    all_map_paths, map_centers, map_names, track_lengths = Config.get_map_paths()
    
    if not all_map_paths:
        print("Warning: No maps found. Please check the racetracks path in config.py")
        # Fallback for development - you can modify these paths
        all_map_paths = ['./maps/example_map']
        map_centers = ['./maps/example_centerline.csv']
        map_names = ['example']
    
    # Get train/test split
    train_maps, test_maps = Config.get_train_test_split(map_names)
    print(f'Train Maps: {train_maps}')
    print(f'Test Maps: {test_maps}')
    
    # Filter maps for training
    indices = [idx for idx, name in enumerate(map_names) if name in train_maps]
    map_path_subset = [all_map_paths[i] for i in indices]
    map_centers_subset = [map_centers[i] for i in indices]
    map_names_subset = [map_names[i] for i in indices]
    
    # Setup paths
    save_path = f'{ALGORITHM}_Fewer_actions/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # Configuration for training
    # Set these paths according to your setup
    inference_file = None  # Set to path of pre-trained weights if continuing training
    reward_file = None     # Set to existing reward file if continuing training
    collision_file = None  # Set to existing collision file if continuing training
    
    # Example of loading previous training (uncomment and modify as needed):
    # inference_file = f'{ALGORITHM}_Fewer_actions/Shanghai_10989.npy'
    # reward_file = f'{ALGORITHM}_Fewer_actions/rewards.npy'
    # collision_file = f'{ALGORITHM}_Fewer_actions/times.npy'
    
    # Set random seed for reproducibility
    np.random.seed(Config.RANDOM_SEED)
    
    # Initialize trainer
    trainer = F1TenthTrainer(
        algorithm=ALGORITHM,
        map_path=map_path_subset,
        map_centers_file=map_centers_subset,
        track_name=map_names_subset,
        save_path=save_path,
        inference_file=inference_file,
        reward_file=reward_file,
        collision_file=collision_file
    )
    
    # Start training
    print(f"Training initialized. Starting {ALGORITHM} training loop...")
    trainer.train()


def run_inference(algorithm='BTSP', weights_file=None):
    """
    Run inference with a trained model.
    
    Args:
        algorithm (str): Algorithm used ('BTSP' or 'SARSA').
        weights_file (str): Path to trained weights file.
    """
    if weights_file is None:
        print("Please specify a weights file for inference")
        return
    
    print(f"Running inference with {algorithm} algorithm...")
    
    # Get map paths (you can modify this for specific test maps)
    all_map_paths, map_centers, map_names, _ = Config.get_map_paths()
    
    # Use first map for inference (modify as needed)
    map_path_subset = [all_map_paths[0]]
    map_centers_subset = [map_centers[0]]
    map_names_subset = [map_names[0]]
    
    # Initialize trainer for inference
    trainer = F1TenthTrainer(
        algorithm=algorithm,
        map_path=map_path_subset,
        map_centers_file=map_centers_subset,
        track_name=map_names_subset,
        save_path=None,
        inference_file=weights_file,
        reward_file=None,
        collision_file=None
    )
    
    # Run inference
    trainer.inference()


if __name__ == "__main__":
    # For training
    main()
    
    # For inference (uncomment and modify as needed)
    # run_inference(algorithm='BTSP', weights_file='BTSP_Fewer_actions/Shanghai_10989.npy')
