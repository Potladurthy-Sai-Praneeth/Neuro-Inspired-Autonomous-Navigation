# F1 Tenth Autonomous Navigation

## Overview
This project focuses on autonomous navigation using the F1 Tenth OpenAI Gym environment. It employs LiDAR-based state representation, reinforcement learning, and weight update mechanisms for efficient decision-making.

## Environment
- **Observations**: 1D LiDAR with a 270° field of view (1080 beams).
- **Actions**: Steering angle and speed.

## State Representation
- LiDAR data is discretized into a binary 11-bit representation using random binary projection.
- Observations are divided into 30 bins, with median values extracted and projected into an 11-dimensional binary space.

## Action Space
- **Steering Angle**: 31 discrete angles mapped to a 270° field of view.
- **Speed**: 10 discrete levels ranging from 0.8 to 2.

## Reward Function
- **Progress Reward**: Encourages forward movement along the track.
- **Centerline Reward**: Penalizes deviation from the track center.
- **Milestone Reward**: Rewards reaching new track centers.
- **Collision Reward**: Large negative reward for crashes.

## Learning Algorithms
### SARSA(λ)
- Temporal Difference (TD) learning with eligibility traces.
- Updates weight matrix using:
  ``` math
  Q(S,A) ← Q(S,A) + α [R + γ Q(S', A') - Q(S,A)] E(S,A)
  ```
- Eligibility trace (E) decays over time, prioritizing recent experiences.

### BTSP (Behavioral Time Scale Plasticity)
- Uses **Eligibility Trace (ET)**. Encodes state representation from LiDAR.
- **Instructive Signal (IS)** from down-sampled LiDAR distances guides weight updates.
- Weight updates follow:
  ```math
  Q[s, a] += R
  ```
  ```math
  \delta = (Q_{max} - Q) \odot (ET * IS)
  ```
  ```math
  Q += \eta \cdot \delta
  ```
  ```math
  ET *= \lambda_1 ET
  ```
  ```math
  IS *= \lambda_2 IS
  ```

## Implementation
1. **State Representation**: Convert LiDAR data to a binary format.
2. **Reward Calculation**: Compute rewards based on progress, track adherence, milestones, and collisions.
3. **Weight Updates**: Apply SARSA(λ) or BTSP to adjust state-action weights.

## Future Work
- Improve discretization techniques for better state representation.
- Optimize action selection strategies for more efficient navigation.
