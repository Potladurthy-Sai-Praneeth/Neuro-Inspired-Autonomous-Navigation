"""
Environment handling and utilities for F1Tenth navigation.
"""

import numpy as np
import gym
import os
import pandas as pd
from f110_gym.envs.base_classes import Integrator
from f110_gym.envs.f110_env import F110Env
from sklearn.preprocessing import normalize


class EnvironmentManager:
    def __init__(self, gym_env_code='f110_gym:f110-v0', num_agents=1, map_path=None, map_ext='.png', 
                 sx=0., sy=0., map_centers_file=None, track_name=None, logger=None):
        """
        Environment manager for F1Tenth navigation.
        
        Args:
            gym_env_code (str): Gym environment code.
            num_agents (int): Number of agents in the environment.
            map_path (list): List of map paths.
            map_ext (str): Map extension.
            sx (float): Initial x position of the agent.
            sy (float): Initial y position of the agent.
            map_centers_file (list): List of paths to map centers files.
            track_name (list): List of track names.
            logger: Logger instance.
        """
        self.path_counter = 0
        self.sx, self.sy = sx, sy
        self.num_agents = num_agents
        self.map_path = map_path
        self.map_ext = map_ext
        self.map_centers_file = map_centers_file
        self.track_name = track_name
        self.logger = logger
        
        # Initialize environment
        self.env = gym.make(gym_env_code, map=self.map_path[self.path_counter], 
                           map_ext=self.map_ext, num_agents=self.num_agents, 
                           timestep=0.01, integrator=Integrator.RK4)
        self.env.add_render_callback(self.render_callback)
        
        # Load map centers
        self._load_map_data()
        
    def _load_map_data(self):
        """Load map centers and track data."""
        file = pd.read_csv(self.map_centers_file[self.path_counter])
        file.columns = ['x', 'y', 'w_r', 'w_l']
        file.index = file.index.astype(int)
        self.map_centers = file.values[:, :2]
        self.track_width = file.loc[0, 'w_r'] + file.loc[0, 'w_l']
        self.track_headings = self.calculate_track_headings(self.map_centers)
        self.track_center_counter =  self.map_centers.shape[0]
        
    def calculate_track_headings(self, track_centers, window_size=5):
        """
        Calculate orientations for track traversal.
        
        Args:
            track_centers (np.ndarray): Shape (N, 2) array of track center points (x, y)
            window_size (int): Number of points to consider for smoothing
        
        Returns:
            np.ndarray: Shape (N,) array of orientation angles in radians
        """
        num_points = track_centers.shape[0]
        half_window = window_size // 2
        
        # Create indices for the future points (with wraparound)
        future_indices = (np.arange(num_points) + half_window) % num_points
        
        # Get the future points
        future_points = track_centers[future_indices]
        
        # Calculate direction vectors
        direction_vectors = future_points - track_centers
        
        # Calculate angles using arctan2
        orientations = np.arctan2(direction_vectors[:, 1], direction_vectors[:, 0])
        
        return orientations

    def update_map(self):
        """
        Update the map of the environment to the next map in the list.
        This function is called after fixed number of collisions with the environment.
        """
        if self.env.renderer is not None:
            self.env.renderer.close()
        self.path_counter += 1
        if self.path_counter == len(self.map_path):
            self.path_counter = 0
        self.env.map_name = self.map_path[self.path_counter]
        self.env.update_map(f'{self.map_path[self.path_counter]}.yaml', self.map_ext)
        F110Env.renderer = None
        
        # Load new map data
        self._load_map_data()
        
        print(f'Map updated to {self.track_name[self.path_counter]}')
        if self.logger:
            self.logger.info('-------' * 20)
            self.logger.info(f'Map updated to: {self.map_path[self.path_counter]}')
        
    def render_callback(self, env_renderer):
        """
        Render callback function to update the map of the environment.
        Do not modify this function.
        """
        e = env_renderer
        x = e.cars[0].vertices[::2]
        y = e.cars[0].vertices[1::2]
        top, bottom, left, right = max(y), min(y), min(x), max(x)
        e.score_label.x = left
        e.score_label.y = top - 700
        e.left = left - 800
        e.right = right + 800
        e.top = top + 800
        e.bottom = bottom - 800


class StateProcessor:
    def __init__(self, n_features=11, n_sectors=22, num_beams=1080, angle=220, random_seed=42):
        """
        State processing utilities for LiDAR data.
        
        Args:
            n_features (int): Number of features for state representation.
            n_sectors (int): Number of LiDAR sectors after downsampling.
            num_beams (int): Number of LiDAR beams.
            angle (int): LiDAR field of view angle.
            random_seed (int): Random seed for projection matrix.
        """
        self.n_features = n_features
        self.n_sectors = n_sectors
        self.num_beams = num_beams
        self.angle = angle
        self.random_seed = random_seed
        
        # Binary powers for state calculation
        self.binary_powers = np.array([2 ** i for i in range(self.n_features)])
        
        # Projection matrix and bias
        self.projection_matrix = self.get_projection_matrix()
        self.bias = np.linspace(-1, 1, self.n_features).reshape(1, -1)
        
        # Normalized LiDAR storage
        self.normalized_lidar = np.zeros((1, self.n_sectors))
        
    def get_projection_matrix(self, zero_prob=0.5, one_prob=0.5):
        """
        Generate or load the projection matrix for state representation.
        
        Args:
            zero_prob (float): Probability of selecting 0.
            one_prob (float): Probability of selecting 1.
            
        Returns:
            np.ndarray: Projection matrix.
        """
        if not os.path.exists('Projection_matrices'):
            os.mkdir('Projection_matrices')
            
        matrix_path = os.path.join('Projection_matrices', 
                                  f'projection_{self.n_features}f_{self.n_sectors}a_s{self.random_seed}.npy')
        
        if not os.path.exists(matrix_path):
            std = np.sqrt(1/self.n_features)
            matrix = np.random.normal(loc=0.0, scale=1/std, size=(self.n_sectors, self.n_features))
            np.save(matrix_path, matrix)
        else:
            matrix = np.load(matrix_path)
        return matrix

    def get_statistical_properties(self, lidar_input):
        """
        Downsample the LiDAR input by calculating median values for sectors.
        
        Args:
            lidar_input (np.ndarray): LiDAR input.
            
        Returns:
            np.ndarray: Downsampled LiDAR data.
        """
        # Select rays corresponding to specified field of view
        sector_size = np.asarray(lidar_input[100:-100], dtype=np.float32).shape[0] // self.n_sectors
        sectors = lidar_input[:sector_size * self.n_sectors].reshape(self.n_sectors, sector_size)
        return np.median(sectors, axis=1).reshape(1, -1)
    
    def binarize_vector(self, vector):
        """
        Binarize the projected LiDAR vector.
        
        Args:
            vector (np.ndarray): Projected LiDAR input.
            
        Returns:
            np.ndarray: Binary representation of the LiDAR data.
        """
        return np.where(vector > 0, 1, 0)

    def get_binary_representation(self, lidar_input):
        """
        Get the binary representation of the LiDAR input.
        
        Args:
            lidar_input (np.ndarray): LiDAR input.
            
        Returns:
            np.ndarray: Binary representation of the LiDAR input.
        """
        self.normalized_lidar = normalize(lidar_input, axis=1)
        return self.binarize_vector(np.dot(lidar_input, self.projection_matrix) + self.bias)
    
    def get_state(self, binary):
        """
        Convert binary representation to state index.
        
        Args:
            binary (np.ndarray): Binary representation of the LiDAR input.
            
        Returns:
            int: State index.
        """
        return np.dot(binary[0], self.binary_powers)
