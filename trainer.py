"""
Main trainer class for F1Tenth navigation with support for both BTSP and SARSA algorithms.
"""

import numpy as np
import os
import time
import logging
from reward import Reward
from index_selector import IndexSelector
from environment import EnvironmentManager, StateProcessor
from algorithms import BTSPAlgorithm, SARSAAlgorithm, ActionSelector
from config import Config


class F1TenthTrainer:
    """Main trainer class supporting both BTSP and SARSA algorithms."""
    
    def __init__(self, algorithm='BTSP', map_path=None, map_centers_file=None, track_name=None,
                 save_path=None, inference_file=None, reward_file=None, collision_file=None):
        """
        Initialize F1Tenth trainer.
        
        Args:
            algorithm (str): Algorithm to use ('BTSP' or 'SARSA').
            map_path (list): List of map paths.
            map_centers_file (list): List of map centers file paths.
            track_name (list): List of track names.
            save_path (str): Path to save weights and results.
            inference_file (str): Path to load pre-trained weights.
            reward_file (str): Path to existing reward file.
            collision_file (str): Path to existing collision file.
        """
        self.algorithm_name = algorithm
        self.save_path = save_path
        self.track_name = track_name
        self.reward_file = reward_file
        self.collision_file = collision_file
        
        # Setup logging
        self._setup_logging()
        
        # Initialize environment manager
        self.env_manager = EnvironmentManager(
            gym_env_code=Config.GYM_ENV_CODE,
            num_agents=Config.NUM_AGENTS,
            map_path=map_path,
            map_ext=Config.MAP_EXT,
            sx=Config.SX,
            sy=Config.SY,
            map_centers_file=map_centers_file,
            track_name=track_name,
            logger=self.logger
        )
        
        # Initialize state processor
        self.state_processor = StateProcessor(
            n_features=Config.N_FEATURES,
            n_sectors=Config.N_SECTORS,
            num_beams=Config.NUM_BEAMS,
            angle=Config.ANGLE,
            random_seed=Config.RANDOM_SEED
        )
        
        # Action space setup
        self.angles_deg = np.linspace(-Config.ANGLE // 2, Config.ANGLE // 2, Config.NUM_ANGLES)[::-1]
        self.angles = np.radians(self.angles_deg)
        self.speeds = np.linspace(Config.MIN_SPEED, Config.MAX_SPEED, Config.NUM_SPEEDS)
        
        # Initialize weights
        self._initialize_weights(inference_file)
        
        # Initialize algorithm
        self._initialize_algorithm()
        
        # Initialize action selector
        self.action_selector = ActionSelector(
            num_angles=Config.NUM_ANGLES,
            num_speeds=Config.NUM_SPEEDS,
            action_threshold_decay=Config.ACTION_THRESHOLD_DECAY
        )
        self.action_selector.update_threshold(self.num_collisions)
        
        # Initialize reward class
        self.reward_class = Reward(
            min_speed=Config.MIN_SPEED,
            max_speed=Config.MAX_SPEED,
            num_speeds=Config.NUM_SPEEDS,
            map_centers=self.env_manager.map_centers,
            track_width=self.env_manager.track_width,
            logger=self.logger
        )
        
        # Initialize index selector
        self.index_selector = IndexSelector(self.env_manager.map_centers.shape[0])
        
        # Training variables
        self.curr_state = None
        self.next_state = None
        self.reward = 0
        self.episode_reward = 0
        self.episodic_rewards = [0]
        self.collision_times = [0]
        
    def _setup_logging(self):
        """Setup logging configuration."""
        log_filename = f'{self.algorithm_name}_logs.log'
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            filename=log_filename,
            filemode='a'
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info(f'Initializing {self.algorithm_name} trainer')
        
    def _initialize_weights(self, inference_file):
        """Initialize or load weights."""
        if inference_file is not None and os.path.exists(inference_file):
            self.weights = np.load(inference_file)
            self.num_collisions = int(inference_file.split('_')[-1].split('.')[0])
            print(f'Loaded weights from {inference_file}')
            self.logger.info(f'Loaded weights from {inference_file}')
        else:
            self.weights = np.random.randn(Config.NUM_STATES, Config.NUM_ANGLES, Config.NUM_SPEEDS)
            self.num_collisions = 0
            
    def _initialize_algorithm(self):
        """Initialize the specified algorithm."""
        if self.algorithm_name == 'BTSP':
            self.algorithm = BTSPAlgorithm(
                num_states=Config.NUM_STATES,
                num_angles=Config.NUM_ANGLES,
                num_speeds=Config.NUM_SPEEDS,
                **Config.BTSP_PARAMS
            )
        elif self.algorithm_name == 'SARSA':
            self.algorithm = SARSAAlgorithm(
                num_states=Config.NUM_STATES,
                num_angles=Config.NUM_ANGLES,
                num_speeds=Config.NUM_SPEEDS,
                **Config.SARSA_PARAMS
            )
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm_name}")
            
    def save_reward_time(self):
        """Save episodic rewards and collision times."""
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        
        if self.reward_file is not None and os.path.exists(self.reward_file):
            r = np.append(np.load(self.reward_file), self.episodic_rewards)
            t = np.append(np.load(self.collision_file), self.collision_times)
        else:
            r = np.array(self.episodic_rewards)
            t = np.array(self.collision_times)
            
        np.save(os.path.join(self.save_path, 'rewards.npy'), r)
        np.save(os.path.join(self.save_path, 'times.npy'), t)

    def save_weights(self):
        """Save model weights."""
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        filename = f'{self.track_name[self.env_manager.path_counter]}_{self.num_collisions + 1}.npy'
        np.save(os.path.join(self.save_path, filename), self.weights)
        self.logger.info(f'Weights saved: {filename}')

    def train(self):
        """Main training loop."""
        # Reset environment and get initial observation
        obs, _, done, _ = self.env_manager.env.reset(
            np.array([[Config.SX, Config.SY, self.env_manager.track_headings[0]]])
        )
        
        # Process initial state
        lidar = obs['scans'][0]
        lidar_downsampled = self.state_processor.get_statistical_properties(lidar)
        self.curr_state = self.state_processor.get_state(
            self.state_processor.get_binary_representation(lidar_downsampled)
        )
        
        # Reset reward and select initial action
        self.reward_class.reset(np.array([[Config.SX, Config.SY]]))
        angle_index, speed_index = self.action_selector.select_action(self.curr_state, self.weights)
        
        start_time = time.time()
        
        while True:
            # Execute action
            steering_angle, speed = self.angles[angle_index], self.speeds[speed_index]
            curr_x, curr_y = obs['poses_x'][0], obs['poses_y'][0]
            
            # Step environment
            obs, _, done, _ = self.env_manager.env.step(np.array([[steering_angle, speed]]))
            
            # Process new state
            lidar = obs['scans'][0]
            lidar_downsampled = self.state_processor.get_statistical_properties(lidar)
            self.next_state = self.state_processor.get_state(
                self.state_processor.get_binary_representation(lidar_downsampled)
            )
            
            # Calculate reward
            if done:
                self.reward = -1000
            else:
                self.reward = self.reward_class.calculate_reward(
                    np.array([curr_x, curr_y]),
                    np.array([obs['poses_x'][0], obs['poses_y'][0]]),
                    self.speeds[speed_index]
                )
            
            self.episode_reward += self.reward
            
            # Algorithm-specific updates
            if self.algorithm_name == 'BTSP':
                self.algorithm.set_eligibility_traces(self.curr_state, angle_index, speed_index, self.state_processor.normalized_lidar)
                self.algorithm.weight_update(self.weights, self.curr_state, angle_index, speed_index, self.reward )
                
            elif self.algorithm_name == 'SARSA':
                self.algorithm.set_eligibility_traces(self.curr_state, angle_index, speed_index)
                if not done:
                    angle_index, speed_index = self.algorithm.weight_update(
                        self.weights, self.curr_state, self.next_state, 
                        angle_index, speed_index, self.reward, self.action_selector
                    )

              # Handle collision
            if done:
                collision_handled = self._handle_collision(start_time)
                if collision_handled:
                    # Get new action for the reset state
                    angle_index, speed_index = self.action_selector.select_action(self.curr_state, self.weights)
                    start_time = time.time()
                    continue
            
            # Select next action (for BTSP) or use SARSA's returned action
            if self.algorithm_name == 'BTSP':
                angle_index, speed_index = self.action_selector.select_action(self.next_state, self.weights)
            
            # Decay eligibility traces and update state
            self.algorithm.decay_eligibility_traces()
            self.curr_state = self.next_state
            
            # Render environment
            # self.env_manager.env.render(mode='human')
            
    def _handle_collision(self, start_time):
        """Handle collision event and environment reset."""
        self.env_manager.track_center_counter -= 1
        self.logger.info(f'Collision: {self.num_collisions+1}, State: {self.curr_state}, Reward: {self.episode_reward}')
        
        # Update tracking variables
        self.action_selector.decay_threshold()
        self.episodic_rewards.append(self.episode_reward)
        self.episode_reward = 0
        end_time = time.time()
        self.collision_times.append(end_time - start_time)
        self.num_collisions += 1
        
        # Reset algorithm traces
        self.algorithm.reset_traces()
        
        # Get new random position on track
        random_idx = self.index_selector.select_index()
        n_x, n_y = self.env_manager.map_centers[random_idx]
        delta_x, delta_y = np.random.uniform(-0.75, 0.75), np.random.uniform(-0.3, 0.3)
        delta_theta = np.random.uniform(-0.2, 0.2)
        n_theta = self.env_manager.track_headings[random_idx]
        
        # Reset environment with new position
        obs, _, done, _ = self.env_manager.env.reset(
            np.array([[n_x + delta_x, n_y + delta_y, n_theta + delta_theta]])
        )
        
        # Process new state
        lidar = obs['scans'][0]
        lidar_downsampled = self.state_processor.get_statistical_properties(lidar)
        self.curr_state = self.state_processor.get_state(
            self.state_processor.get_binary_representation(lidar_downsampled)
        )
        
        # Reset reward class
        self.reward_class.reset(np.array([[n_x + delta_x, n_y + delta_y]]))
        
        # Checkpoint saving
        if (self.num_collisions + 1) % Config.CHECKPOINT_INTERVAL == 0:
            print(f'Collision: {self.num_collisions+1}, Time: {sum(self.collision_times)}, Reward: {sum(self.episodic_rewards)/len(self.episodic_rewards)}')
            self.save_reward_time()
            self.episodic_rewards.clear()
            self.collision_times.clear()
            self.reward_file = os.path.join(self.save_path, 'rewards.npy')
            self.collision_file = os.path.join(self.save_path, 'times.npy')
        
        # Map switching
        if self.env_manager.track_center_counter == 0:
            print(f'Training on {self.track_name[self.env_manager.path_counter]} Completed')
            self.save_weights()
            self.env_manager.update_map()
            self.reward_class.set_parameters(self.env_manager.map_centers, self.env_manager.track_width)
            self.reward_class.reset(np.array([[Config.SX, Config.SY]]))
            self.index_selector.set_parameters(self.env_manager.map_centers.shape[0])
        
        return True  # Indicate that collision was handled
        
    def inference(self):
        """Run inference with trained model."""
        obs, _, done, _ = self.env_manager.env.reset(
            np.array([[Config.SX, Config.SY, self.env_manager.track_headings[0]]])
        )
        
        lidar = obs['scans'][0]
        lidar_downsampled = self.state_processor.get_statistical_properties(lidar)
        self.curr_state = self.state_processor.get_state(
            self.state_processor.get_binary_representation(lidar_downsampled)
        )
        
        self.reward_class.reset(np.array([[Config.SX, Config.SY]]))
        angle_index, speed_index = self.action_selector.select_action_inference(self.curr_state, self.weights)
        
        while not done:
            steering_angle, speed = self.angles[angle_index], self.speeds[speed_index]
            obs, _, done, _ = self.env_manager.env.step(np.array([[steering_angle, speed]]))
            
            lidar = obs['scans'][0]
            lidar_downsampled = self.state_processor.get_statistical_properties(lidar)
            self.next_state = self.state_processor.get_state(
                self.state_processor.get_binary_representation(lidar_downsampled)
            )
            
            angle_index, speed_index = self.action_selector.select_action_inference(self.next_state, self.weights)
            self.curr_state = self.next_state
            
            self.env_manager.env.render(mode='human')
