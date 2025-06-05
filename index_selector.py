"""
Index selector for randomized training positions.
"""

import numpy as np


class IndexSelector:
    def __init__(self, num_indices):
        """
        IndexSelector class is used to introduce the randomized training of the agent.
        After each collision, the agent will select a random index from all the track centers and adjusts its new position to a random position on the track.
        Each index is selected based on the probability of the index being selected. The probability is updated after each selection.
        
        Args:
            num_indices (int): Number of indices to select from.
        """
        self.set_parameters(num_indices)
    
    def set_parameters(self, num_indices):
        """
        Helper function to set the parameters of the index selector externally from the class instance.
        
        Args:
            num_indices (int): Number of indices to select from.
        """
        self.num_indices = num_indices
        self.visited_indices = set()
        self.probabilities = np.ones(num_indices) / num_indices
        self.all_indices = np.arange(self.num_indices)
    
    def select_index(self):
        """
        Select an index based on the current probabilities.
        This function is called at each collision with the environment.
        If all indices have been visited, the probabilities are reset and a new index is selected.
        
        Returns:
            int: Selected index.
        """
        if len(self.visited_indices) == self.num_indices:
            # Reset the probabilities and visited indices
            print('Visited all indices, resetting')
            self.visited_indices = set()
            self.probabilities = np.ones(self.num_indices) / self.num_indices

        # Select an index based on the current probabilities
        random_idx = np.random.choice(self.all_indices, p=self.probabilities)

        # Update the probabilities
        self.visited_indices.add(random_idx)
        if len(self.visited_indices) < self.num_indices:
            self.probabilities[random_idx] = 0
            remaining_prob = 1 - np.sum(self.probabilities)
            self.probabilities[self.probabilities > 0] += remaining_prob / np.sum(self.probabilities > 0)

        return random_idx
