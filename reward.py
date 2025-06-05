"""
Reward function for F1Tenth autonomous navigation.
Calculates centerline, speed, and progress rewards.
"""

import numpy as np


class Reward:
    def __init__(self, min_speed=0.8, max_speed=2, num_speeds=5, map_centers=None, track_width=2.2, logger=None):
        """
        Initialize reward function.
        
        Args:
            min_speed (float): Minimum speed for reward calculation
            max_speed (float): Maximum speed for reward calculation  
            num_speeds (int): Number of discrete speeds
            map_centers (np.ndarray): Array of map centers
            track_width (float): Width of the track
            logger: Logger instance
        """
        # Keep existing parameters
        self.min_speed = min_speed
        self.max_speed = max_speed
        self.num_speeds = num_speeds
        self.speeds = np.linspace(min_speed, max_speed, num_speeds)
        self.mean_speed = np.mean(self.speeds)
        self.std_speed = np.std(self.speeds)
        self.set_parameters(map_centers, track_width)
        
        # Hyperparameters
        self.epsilon = 1e-5
        self.distance_travelled = 0
        self.milestone = 0
        self.centerline_scale = 5
        self.progress_scale = 1.0
        self.logger = logger

    def set_parameters(self, map_centers, track_width):
        """
        Helper function to set the parameters of the reward function externally from the class instance.
        
        Args:
            map_centers (np.ndarray): Array of map centers.
            track_width (float): Width of the track.
        """
        self.map_centers = map_centers
        # Initial point and center that determines the position at the start of episode
        self.initial_point = np.array([[0, 0]])
        self.initial_center_idx, _ = self.__calculate_distance_from_center(self.map_centers, self.initial_point)
        self.initial_center = self.map_centers[self.initial_center_idx]

        # Race Track parameters
        self.distance_between_centers = np.hstack([[0.], np.linalg.norm(self.map_centers[:-1, :] - self.map_centers[1:, :], axis=1)])
        self.total_track_length = np.sum(self.distance_between_centers)
        self.track_width = track_width
    
    def __calculate_distance_from_center(self, centers, curr):
        """
        Helper function to calculate the distance from all centers of the track to the current position.
        
        Args:
            centers (np.ndarray): Array of map centers.
            curr (np.ndarray): Current position of the agent.
            
        Returns:
            idx (int): Index of the closest center.
            distance (float): Distance to the closest center.
        """
        distances = np.linalg.norm(centers - curr, axis=1)
        idx = np.argmin(distances)
        return idx, distances[idx]
    
    def reset(self, pos):
        """
        Reset the reward function state, supporting arbitrary starting position and heading.
        
        Args:
            pos (np.ndarray): Starting position of the agent.
        """
        self.distance_travelled = 0
        self.milestone = 0
        self.initial_point = pos
        
        # Initialize starting center reference
        self.initial_center_idx, _ = self.__calculate_distance_from_center(self.map_centers, self.initial_point)
        self.initial_center = self.map_centers[self.initial_center_idx]
    
    def centerline_reward(self, curr_pos, next_pos, curr_center_idx, next_center_idx, curr_dist, next_dist, threshold_angle=0.2, threshold_dist=0.35):
        """
        Calculate the centerline reward based on the distance from the centerline. 
        Angles are computed in radians.
        Restricting the large negative reward to -1000 and positive to 100.
        
        Args:
            curr_pos (np.ndarray): Current position of the agent.
            next_pos (np.ndarray): Next position of the agent.
            curr_center_idx (int): Current center index.
            next_center_idx (int): Next center index.
            curr_dist (float): Current distance from centerline.
            next_dist (float): Next distance from centerline.
            threshold_angle (float): Threshold angle for reward calculation.
            threshold_dist (float): Threshold distance for reward calculation.
        
        Returns:
            float: Centerline reward.
        """
        if curr_center_idx == next_center_idx:
            next_center_idx = (next_center_idx + 1) % len(self.map_centers)
        
        movement_vector = next_pos - curr_pos
        movement_vector /= (np.linalg.norm(movement_vector) + self.epsilon)

        centerline_vector = self.map_centers[next_center_idx] - self.map_centers[curr_center_idx]
        centerline_vector /= (np.linalg.norm(centerline_vector) + self.epsilon)

        angle = np.arctan2(centerline_vector[1], centerline_vector[0]) - np.arctan2(movement_vector[1], movement_vector[0])
        angle = np.arctan2(np.sin(angle), np.cos(angle))  # Normalize angle to [-pi, pi]

        angle = abs(angle)  # Consider only the absolute angle deviation

        if angle <= threshold_angle:
            reward = 2.0 - (angle / threshold_angle)
        else:
            # Exponential decay
            reward = -4.3 * np.exp(3 * (angle - threshold_angle))
        
        if next_dist <= threshold_dist:
            # Reward increases as distance gets closer to 0
            dist_reward = 2.0 * (1 - next_dist / threshold_dist)
        else:
            # Penalize exponentially as distance increases beyond threshold
            dist_reward = -2.0 * np.exp(1.5 * (next_dist - threshold_dist))
        
        # Combine rewards
        total_reward = reward + dist_reward
        
        # Add bonus if the agent is improving its distance to centerline
        if next_dist < curr_dist:
            total_reward += 1  # Small bonus for getting closer to centerline

        return np.clip(total_reward, -1000, 100)

    def speed_reward(self, speed):
        """
        Calculate the speed reward based on the speed of the agent.
        Uses gaussian distribution of the speeds and rewards agent positively for speeds that fall in the center of the distribution and negative on both sides.
        
        Args:
            speed (float): Speed of the agent.
        
        Returns:
            float: Speed reward.
        """
        # Calculate the Gaussian probability density function
        reward = 4 * ((1/np.sqrt(2*np.pi*self.std_speed**2) * np.exp(-0.5/(self.std_speed**2) * (speed-self.mean_speed)**2)) - self.mean_speed/self.speeds[-1])
        return reward
        
    def progress_reward(self, curr_pos, next_pos, curr_center_idx, next_center_idx):
        """
        Calculate the progress reward based on the distance travelled along the track. 
        Restricting the positive reward to +100.
        
        Args:
            curr_pos (np.ndarray): Current position of the agent.
            next_pos (np.ndarray): Next position of the agent.
            curr_center_idx (int): Index of the current center.
            next_center_idx (int): Index of the next center.
        
        Returns:
            float: Progress reward.
        """
        if curr_center_idx == next_center_idx:
            next_center_idx = (next_center_idx + 1) % len(self.map_centers)
        
        distance = np.linalg.norm(next_pos - curr_pos)
        self.distance_travelled += distance

        reward = self.distance_travelled
        if self.distance_travelled > 100:
            self.distance_travelled = 0
        
        return reward
    
    def calculate_reward(self, curr_pos, next_pos, speed):
        """
        Calculate the total reward based on centerline and speed rewards.
        
        Args:
            curr_pos (np.ndarray): Current position of the agent.
            next_pos (np.ndarray): Next position of the agent.
            speed (float): Speed of the agent.
        
        Returns:
            float: Total reward.
        """
        curr_center_idx, curr_dist = self.__calculate_distance_from_center(self.map_centers, curr_pos)
        next_center_idx, next_dist = self.__calculate_distance_from_center(self.map_centers, next_pos)

        centerline_reward = self.centerline_reward(curr_pos, next_pos, curr_center_idx, next_center_idx, curr_dist, next_dist)
        speed_reward = self.speed_reward(speed)
        
        total_reward = centerline_reward + speed_reward

        return total_reward
