"""
Configuration and constants for F1Tenth navigation training.
"""

import os


class Config:
    """Configuration class containing all hyperparameters and settings."""
    
    # Environment settings
    GYM_ENV_CODE = 'f110_gym:f110-v0'
    NUM_AGENTS = 1
    MAP_EXT = '.png'
    
    # Initial position
    SX = 0.
    SY = 0.
    
    # Track headings for different tracks
    TRACK_HEADINGS = {
        'Hockenheim': 2.02,
        'Mexico City': -0.15,
        'Oschersleben': 2.86,
        'Shanghai': -2.93,
        'BrandsHatch': 0.42,
        'Monza': 1.47,
        'Catalunya': -2.14,
        'SaoPaulo': -1.31,
        'Sepang': -3.06,
        'Silverstone': 0.94,
        'Nuerburgring': -2.38,
        'YasMarina': 0.13,
        'Spa': 2.13,
        'Sochi': -2.14,
        'Montreal': -1.35,
        'Austin': -0.65,
        'Melbourne': 2.37,
        'Budapest': 2.45,
        'Spielberg': -2.88,
        'Zandvoort': 1.2,
        'Sakhir': 1.53,
        'MoscowRaceway': 1.46
    }
    
    # LiDAR and state processing
    NUM_BEAMS = 1080
    N_FEATURES = 11
    ANGLE = 220
    N_SECTORS = 22
    
    # Action space
    NUM_ANGLES = 22  # Same as n_sectors
    NUM_SPEEDS = 5
    MIN_SPEED = 0.8
    MAX_SPEED = 1.8
    
    # State space
    NUM_STATES = 2 ** N_FEATURES
    
    # Random seed
    RANDOM_SEED = 42
    
    # Algorithm parameters
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
    
    # Training parameters
    ACTION_THRESHOLD_DECAY = 0.9998
    CHECKPOINT_INTERVAL = 500
    MAP_SWITCH_COLLISIONS = None  # Set based on track_center_counter
    
    # Projection matrix parameters
    ZERO_PROB = 0.5
    ONE_PROB = 0.5
    
    @classmethod
    def get_map_paths(cls, base_path='./../f1tenth_racetracks'):
        """
        Get all available map paths.
        
        Args:
            base_path (str): Base path to racetracks folder.
            
        Returns:
            tuple: (all_map_paths, map_centers, map_names, track_lengths)
        """
        all_map_paths = []
        map_centers = []
        map_names = []
        track_lengths = []
        
        if not os.path.exists(base_path):
            # Fallback for local development
            base_path = './f1tenth_racetracks'
            
        if os.path.exists(base_path):
            for folder in os.listdir(base_path):
                if folder not in ['README.md', '.gitignore', 'convert.py', 'LICENSE', 'rename.py', '.git']:
                    folder_name = folder
                    file_name = folder_name.replace(' ', '') + '_map'
                    map_center = folder_name.replace(' ', '') + '_centerline.csv'
                    
                    map_center_path = f'{base_path}/{folder_name}/{map_center}'
                    if os.path.exists(map_center_path):
                        import pandas as pd
                        track_lengths.append(len(pd.read_csv(map_center_path)))
                        map_names.append(folder_name)
                        all_map_paths.append(f'{base_path}/{folder_name}/{file_name}')
                        map_centers.append(map_center_path)
        
        return all_map_paths, map_centers, map_names, track_lengths
    
    @classmethod
    def get_train_test_split(cls, map_names, train_maps=None):
        """
        Get train/test split of maps.
        
        Args:
            map_names (list): List of all map names.
            train_maps (list): List of training map names. If None, uses default.
            
        Returns:
            tuple: (train_maps, test_maps)
        """
        if train_maps is None:
            train_maps = ['Shanghai', 'Nuerburgring', 'Montreal', 'Austin', 'Hockenheim','Mexico City'] # 'Spa'
        
        test_maps = [name for name in map_names if name not in train_maps]
        return train_maps, test_maps
