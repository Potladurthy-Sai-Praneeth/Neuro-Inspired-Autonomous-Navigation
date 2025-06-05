"""
BTSP and SARSA algorithms for reinforcement learning.
"""

import numpy as np
from scipy.sparse import csr_matrix


class BTSPAlgorithm:
    """BTSP (Behavioural Time Scale Synaptic Plasticity) Algorithm implementation."""
    
    def __init__(self, num_states, num_angles, num_speeds, learning_rate=1e-3, 
                 et_decay_rate=0.9, is_decay_rate=0.7, max_weight=5):
        """
        Initialize BTSP algorithm.
        
        Args:
            num_states (int): Number of states.
            num_angles (int): Number of angle actions.
            num_speeds (int): Number of speed actions.
            learning_rate (float): Learning rate for weight updates.
            et_decay_rate (float): Eligibility trace decay rate.
            is_decay_rate (float): IS decay rate.
            max_weight (float): Maximum weight value.
        """
        self.num_states = num_states
        self.num_angles = num_angles
        self.num_speeds = num_speeds
        self.learning_rate = learning_rate
        self.et_decay_rate = et_decay_rate
        self.is_decay_rate = is_decay_rate
        self.max_weight = max_weight
        
        # Eligibility Trace and IS matrices
        self.ET = np.zeros((1, self.num_states))
        self.IS = np.zeros((self.num_angles, self.num_speeds))
        
    def set_eligibility_traces(self, curr_state, angle_idx, speed_idx, normalized_lidar):
        """
        Set eligibility traces for the current state-action pair.
        
        Args:
            curr_state (int): Current state index.
            angle_idx (int): Angle index of the selected action.
            speed_idx (int): Speed index of the selected action.
            normalized_lidar (np.ndarray): Normalized LiDAR data.
        """
        self.ET[0, curr_state] = 1
        self.IS[angle_idx, speed_idx] = normalized_lidar[0, angle_idx]

    def decay_eligibility_traces(self):
        """
        Decay the eligibility traces based on the decay rates.
        """
        self.ET *= self.et_decay_rate
        self.IS *= self.is_decay_rate
        self.IS[self.IS < 1e-6] = 0
        self.ET[self.ET < 1e-4] = 0

    def weight_update(self, weights, curr_state, angle_idx, speed_idx, point_reward):
        """
        Update weights using BTSP algorithm.
        
        Args:
            weights (np.ndarray): Weight matrix.
            curr_state (int): Current state index.
            angle_idx (int): Angle index of the selected action.
            speed_idx (int): Speed index of the selected action.
            point_reward (float): Reward received from the environment.
        """
        # Using distance at specific location with IS decay
        ET_IS_product = (csr_matrix(self.ET).T.dot(self.IS.reshape(1, -1))).reshape(
            self.num_states, self.num_angles, self.num_speeds)
        non_zero_indices = np.argwhere(self.ET != 0)[:, -1]
        weights[curr_state, angle_idx, speed_idx] += point_reward 
        delta = (self.max_weight - weights[non_zero_indices]) * ET_IS_product[non_zero_indices]
        np.add(weights[non_zero_indices], self.learning_rate * delta, out=weights[non_zero_indices])

    def reset_traces(self):
        """Reset eligibility traces."""
        self.ET.fill(0)
        self.IS.fill(0)


class SARSAAlgorithm:
    """SARSA Algorithm implementation."""
    
    def __init__(self, num_states, num_angles, num_speeds, learning_rate=1e-3, 
                 discount_factor=0.95, decay_rate=0.9):
        """
        Initialize SARSA algorithm.
        
        Args:
            num_states (int): Number of states.
            num_angles (int): Number of angle actions.
            num_speeds (int): Number of speed actions.
            learning_rate (float): Learning rate for weight updates.
            discount_factor (float): Discount factor for future rewards.
            decay_rate (float): Decay rate for eligibility traces.
        """
        self.num_states = num_states
        self.num_angles = num_angles
        self.num_speeds = num_speeds
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.decay_rate = decay_rate
        
        # Eligibility trace matrix
        self.ET_IS = np.zeros((self.num_states, self.num_angles, self.num_speeds))
        
    def set_eligibility_traces(self, curr_state, angle_idx, speed_idx):
        """
        Set eligibility traces for the current state-action pair.
        
        Args:
            curr_state (int): Current state index.
            angle_idx (int): Angle index of the selected action.
            speed_idx (int): Speed index of the selected action.
        """
        self.ET_IS[curr_state, angle_idx, speed_idx] = 1

    def decay_eligibility_traces(self):
        """
        Decay the eligibility traces based on the decay rate.
        """
        self.ET_IS *= self.discount_factor * self.decay_rate

    def weight_update(self, weights, curr_state, next_state, angle_idx, speed_idx, reward, action_selector):
        """
        Update weights using SARSA algorithm and return next action.
        
        Args:
            weights (np.ndarray): Weight matrix.
            curr_state (int): Current state index.
            next_state (int): Next state index.
            angle_idx (int): Angle index of the selected action.
            speed_idx (int): Speed index of the selected action.
            reward (float): Reward received from the environment.
            action_selector: Object with select_action method.
            
        Returns:
            tuple: Next action indices (angle_idx, speed_idx).
        """
        next_angle_idx, next_speed_idx = action_selector.select_action(next_state)
        delta = (reward + self.discount_factor * weights[next_state, next_angle_idx, next_speed_idx] - 
                weights[curr_state, angle_idx, speed_idx])
        
        weights += self.learning_rate * delta * self.ET_IS
        return next_angle_idx, next_speed_idx

    def reset_traces(self):
        """Reset eligibility traces."""
        self.ET_IS.fill(0)


class ActionSelector:
    """Action selection utilities."""
    
    def __init__(self, num_angles, num_speeds, action_threshold_decay=0.9998):
        """
        Initialize action selector.
        
        Args:
            num_angles (int): Number of angle actions.
            num_speeds (int): Number of speed actions.
            action_threshold_decay (float): Decay rate for exploration threshold.
        """
        self.num_angles = num_angles
        self.num_speeds = num_speeds
        self.action_threshold_decay = action_threshold_decay
        self.action_threshold = 0.1
        
    def update_threshold(self, num_collisions):
        """
        Update action threshold based on number of collisions.
        
        Args:
            num_collisions (int): Number of collisions so far.
        """
        self.action_threshold = 0.1 * (self.action_threshold_decay ** num_collisions)

    def select_action(self, state, weights):
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state (int): Current state index.
            weights (np.ndarray): Weight matrix.
            
        Returns:
            tuple: Action indices (angle_index, speed_index).
        """
        random_number = np.random.rand()
        if random_number < self.action_threshold:
            angle_index = np.random.randint(0, self.num_angles)
            speed_index = np.random.randint(0, self.num_speeds)
        else:
            max_value = np.max(weights[state])
            max_indices = np.argwhere(weights[state] == max_value)
            angle_index, speed_index = max_indices[np.random.choice(np.arange(len(max_indices)))]
        
        return angle_index, speed_index

    def select_action_inference(self, state, weights):
        """
        Select action for inference (no exploration).
        
        Args:
            state (int): Current state index.
            weights (np.ndarray): Weight matrix.
            
        Returns:
            tuple: Action indices (angle_index, speed_index).
        """
        max_indices = np.argwhere(weights[state] == np.max(weights[state]))
        angle_index, speed_index = max_indices[np.random.choice(np.arange(len(max_indices)))]
        return angle_index, speed_index

    def decay_threshold(self):
        """Decay the action threshold."""
        self.action_threshold *= self.action_threshold_decay
