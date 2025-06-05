# Simulation Process Flow

## 1. Environment

* The simulation uses an Open AI Gym environment customized by F1 Tenth. 
* The environment provides observations in the form of 1D LiDAR data. 
* The 1D LiDAR has a 270° field of view.
* Each 1D vector consists of 1080 beams, where each beam represents the distance to a wall or obstacle. 

## 2. State Space Representation

* The state space represents the agent's location and surroundings on the racetrack, using observations from the environment. 
* Observations (LiDAR readings) are continuous values. 
* To use continuous state space in constrained environments, we discretize it
* In this simulation, LiDAR information is represented in a binary format. 
* An 11-bit binary representation is used, offering a balance between having a finite number of states (2^11) and sufficient resolution for LiDAR readings. 

### 2.1 Discretizing Observations

* The process of discretizing the observations is as follows:
    1.  The first and last 100 values are removed from the 1080-value observation vector, resulting in 880 values (to focus on a 220° field of view). 
    2.  The remaining observations are grouped into *p* bins (where *p* is user-defined). The number of bins determines the number of steering angles in the action space. 
    3.  A trade-off exists in choosing the number of bins:
        * More bins provide finer control over steering angles but increase the action space. 
        * Fewer bins lead to a smaller action space but reduce steering angle granularity. 
    4.  The median distance to an obstacle/wall within each bin is calculated. The median is used because it preserves the distribution of LiDAR data better than the mean. 
    5.  The resulting p-dimensional vector is projected into an 11-dimensional space using a random binary projection matrix. The projection matrix is generated randomly with a 4:1 ratio of 0s and 1s. 
    6.  This projection is converted to binary values using a threshold. The projection aims to preserve similarity between data points while providing broad coverage across the 2^11 states. 

## 3. Action Space

* The action space consists of:
    * **Steering Angle:**
        * As the LiDAR data is downsampled to *p* beams, the angle between two beams is approximately (220/p) degrees.
        * Each of the *p* beams corresponds to a specific angle between -110° and 110°. 
        * A vector of angles (in radians) is precomputed to represent the possible steering angles. 
    * **Speed:**
        * Speed is included as a real value, uniformly sampled between 0.8 and 1.8, to allow the agent to learn optimal speeds. 
        * 's' different speed levels are considered. 
        * The number of steering angles and speed levels can be adjusted to control granularity. 

## 4. Weight Representation

* The algorithm adjusts weights to determine actions, so weights are represented considering that each state has a set of possible actions (steering angle, speed).
* Each state has synapses connected to each of the p steering angles and s speeds.
* This can be visualized as a weight matrix of shape (p, s) for each state.
* A 3D weight representation is used to account for all 2^11 states. 

## 5. Reward

* The reward function is crucial for the agent's learning process.
* Rewards can be positive or negative, depending on the chosen action. 
* The reward function includes the following components:
    * **Progress Reward:**
        * This reward quantitatively measures the agent's progress from the starting point.
        * It is normalized using track properties (track length) to prevent biasing the agent.
        * Progress Reward = (Distance Traveled) / (Track Length) 
    * **Centerline Reward:**
        * This reward penalizes the agent for deviating from the track's centerline.
        * It is calculated using track center coordinates provided by the environment.
        * The angle (ϴ) between the agent's direction (v1) and the direction towards the nearest track center (v2) is used to determine the reward
        * If the deviation angle is greater than 90°, a negative reward is given. 
        * If the deviation angle is between 0° and 90°, an exponential decay function is used to calculate the reward. 
    * **Milestone Reward:**
        * A small positive reward is given for reaching every new track center
        * This reinforces the agent for correct actions. 
        * It is calculated based on the distance traveled from the starting point.
    * **Collision Reward:**
        * A large negative reward is given upon collision.
    * The final reward for each iteration is the sum of all the above reward components.
    * Speed reward can also be included in the reward function. 

## 6. Weight Updates

### 6.1 SARSA λ Weight Updates

* Sarsa λ is a reinforcement learning algorithm used for decision-making in uncertain environments
* It helps the agent learn optimal policies to maximize rewards.
* It is based on Temporal Difference (TD) learning. 
* It uses eligibility traces to weight recent experiences more heavily.


### 6.2 BTSP Weight Updates

* In BTSP, synaptic changes are modulated by Eligibility Trace (ET) and Instructive Signal (IS). 
* **Eligibility Trace (ET):**
    * ET is a decaying signal generated by a visual/sensory cue (LiDAR information in this case). 
    * It is a 1D vector with the same shape as the number of states (1, 2^11). 
    * The magnitude of ET indicates the change in synaptic weight for each state. 
    * Older states have lower ET values (less change in synaptic weights) compared to current states. 
    * The state index from LiDAR is used to update ET values. 
    * The ET value for the current state is set to 1 and decays over time. 
    * Example: ET \[0, 0, 0.81, 0, 0, 0, 0, 0.9, 0, 0, ..., 1] at timestep 3 indicates state 3 (timestep 0) has an eligibility of 0.81, state 7 (timestep 1) has 0.9, and the current state (2048) has 1. 
* **Instructive Signal (IS):**
    * IS is a supervisory signal for synaptic modulation, arising from dendritic calcium spikes. 
    * In this simulation, LiDAR information (distances to obstacles) is used as the supervisory signal.
    * IS is a 2D vector with the same shape as the action space (steering angles, speeds). 
    * The down-sampled LiDAR information with median values represents distance information. 
    * Distance information is updated only at the specific (state, action) pair chosen by the agent. 
    * To track agent history, the current pair is updated with the current distance, and all old pairs are decayed with a rate of 0.7. 
* **BTSP Weight Update Rules:**
    * Two operations are used to modify the weights:
        * W<sub>angle, speed</sub> += Reward 
        * dW = W<sub>max</sub> - W ⊙ ET \* IS 
        * W = W + η \* dW 
    * Where:
        * η is the learning rate. 
        * \* represents matrix multiplication. 
        * ⊙ represents element-wise multiplication. 
