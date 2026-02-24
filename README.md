# AI-Based Autonomous Car Racing Simulation with PPO & Imitation Learning
**A Safe Reinforcement Learning Framework for Competitive Racing**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)
[![Algorithm](https://img.shields.io/badge/Algorithm-PPO_%2B_Imitation-orange)](https://openai.com/research/proximal-policy-optimization)

> **Note:** This project implements a hybrid learning architecture (Imitation Learning + PPO) integrated with Action Mapping and State Mapping safety mechanisms for high-speed autonomous driving.

---

## Project Overview

Autonomous racing requires a delicate balance between aggressive speed and physical safety constraints. This project addresses the "cold start" problem in Reinforcement Learning by using a two-stage training process:

1.  **Warm Start (Imitation Learning):** The agent initially learns to clone expert driving behaviors to establish a baseline safe policy.
2.  **Optimization (PPO):** The policy is fine-tuned using Proximal Policy Optimization to surpass the expert's performance.

To ensure safety during this aggressive optimization, the system implements an Action Mapping mechanism to enforce traction limits and a State Mapping mechanism to handle dynamic overtaking scenarios.

---

## Demo

![Simulation Demo Placeholder](https://github.com/metadriverse/metadrive/raw/main/metadrive/assets/metadrive_teaser.gif)
*Agent utilizing State Mapping to identify feasible overtaking corridors while adhering to friction constraints.*

---

## Installation (Standard Python)

### Prerequisites
* **Python 3.10** (Strict requirement).
    * **Windows:** Download the [Python 3.10 Installer](https://www.python.org/downloads/release/python-31011/) (Scroll to "Files" -> "Windows installer (64-bit)").
    * **Mac/Linux:** Use your package manager (e.g., `brew install python@3.10` or `sudo apt install python3.10`).

### Setup Instructions

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/tarasathwik/MetaDrive_Project.git](https://github.com/tarasathwik/MetaDrive_Project.git)
    cd MetaDrive_Project
    ```

2.  **Create a Virtual Environment:**
    We must explicitly use the Python 3.10 executable to create the environment.

    **Windows:**
    ```bash
    # If you installed Python 3.10 correctly, the 'py' launcher handles versions:
    py -3.10 -m venv venv
    
    # Activate it:
    .\venv\Scripts\activate
    ```

    **Mac/Linux:**
    ```bash
    # Point to the specific 3.10 binary
    python3.10 -m venv venv
    
    # Activate it:
    source venv/bin/activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Verify Installation:**
    Check that the simulator launches:
    ```bash
    python scripts/play_game.py
    ```

## Mathematical Methodology

This framework combines modern RL algorithms with rigorous safety constraints.

## 1. Hybrid Learning Architecture

### Stage I: Imitation Learning (Behavioral Cloning)

We minimize the divergence between the agent’s policy and expert actions using the Behavioral Cloning (BC) loss:

```math
L_{\text{BC}}(\theta)
=
\mathbb{E}_{(s, a^*) \sim \mathcal{D}_{\text{expert}}}
\left[
\left\| \pi_\theta(s) - a^* \right\|_2^2
\right]
```

---

### Stage II: Proximal Policy Optimization (PPO)

After imitation pretraining, the policy is fine-tuned using Proximal Policy Optimization with a clipped surrogate objective to ensure stable updates:

```math
L^{\text{CLIP}}(\theta)
=
\mathbb{E}_t
\left[
\min \left(
r_t(\theta)\,\hat{A}_t,\;
\text{clip}\!\left(r_t(\theta), 1-\epsilon, 1+\epsilon\right)\hat{A}_t
\right)
\right]
```

where

```math
r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)}
```

---

## 2. Safety Mechanisms

### Action Mapping (Traction Control)

To prevent skidding, the agent’s actions are constrained by the **Friction Circle Constraint**, ensuring adherence to tire friction limits:

```math
\sqrt{F_x^2 + F_y^2} \le \mu_{\text{max}} F_z
```

Unsafe actions are projected back onto the feasible set using an action-mapping function:

```math
a_{\text{safe}} = H_{\text{AM}}(a)
```

---

### State Mapping (Dynamic Perception)

For competitive overtaking, raw track observations are transformed into a feasible state representation based on ego and opponent dynamics:

```math
\hat{V}_E = H_{\text{SM}}(V_E, V_C)
```
  
  This allows the agent to perceive the track as a dynamic corridor, ignoring blocked lanes.

---

## Usage

### 1. Collect Expert Data
Run the manual control script to generate expert demonstrations for Imitation Learning.
```bash
python scripts/collect_data.py --episodes 10
```

### 2. Train the Agent (Hybrid Pipeline)
After collecting the expert data, the training pipeline executes in two stages:
```bash
# Stage 1 & 2: Runs BC pre-training followed by PPO optimization
python scripts/train_rl_agent.py
```

### 3. Evaluate the Model
To watch the fully trained agent navigate the MetaDrive environment with 3D rendering enabled:
```bash
python scripts/test_final_agent.py
```

---

## Training Dynamics & Findings

The agent was trained for **500,000 steps** using the Constrained PPO architecture (integrating the Friction Circle `ActionMapWrapper` and `StateMapWrapper`). A deep data analysis of the TensorBoard metrics revealed unique learning dynamics characteristic of optimizing high-speed autonomous racing under strict mathematical and physical constraints:

### 1. The Exploration-Exploitation Shift (Entropy Loss)
Unlike standard RL convergence where entropy strictly decreases, the **Entropy Loss** showed a steady incline during the transition from the Imitation Learning (BC) warm-start to PPO. 
* **Analysis**: The agent began with artificially low entropy due to being heavily biased by the safe, conservative BC expert data. As PPO optimization engaged, the agent intentionally increased its action uncertainty to actively explore higher-speed trajectories while strictly obeying the Action Mapping boundaries.

![Constrained Entropy Loss Curve](./results/constrained_entropy_loss.png)
*Figure 1: Entropy expansion as the policy breaks away from the Imitation baseline to mathematically explore optimal racing lines within the predefined friction limits.*

### 2. Policy Stability in High-Variance Environments
The **Approximate KL Divergence** exhibited significant jitter, reflecting the extreme sensitivity of the MetaDrive physics engine at high speeds.
* **Analysis**: In a racing context, minor steering deviations at the edge of the friction circle can lead to catastrophic traction loss. The PPO Clipped Surrogate Objective effectively bounded these massive policy updates, ensuring the neural network did not collapse when testing the absolute limits of the vehicle's grip.

![Constrained KL Divergence](./results/constrained_kl_divergence.png)
*Figure 2: Approximate KL Divergence demonstrating the high-variance environment and the algorithm's active bounding of policy updates.*

### 3. Safe Optimization Convergence (Reward vs. Episode Length)
A comparative data analysis of the baseline Unconstrained PPO model versus the Action-Mapped (Constrained) model revealed a critical insight into the environment's reward dynamics. Both models successfully climbed to a stable mean reward asymptote of **~230**, but exhibited fundamentally different geometric and temporal behaviors:
* **Unconstrained Model:** Gathered points rapidly through high-variance, chaotic driving, resulting in frequent collisions and early episode terminations.
* **Constrained Model:** Because the `ActionMapWrapper` mathematically blocked physically impossible, chaotic swerving, the agent gathered points at a slower rate per frame with significantly lower variance. The model achieved the exact same final expected value (~230) by drastically increasing its overall **Episode Length**. The data proves the agent learned that maintaining stable, kinematically valid geometric arcs ensures long-term survival and consistent reward accumulation.

![Constrained Episode Reward Mean](./results/constrained_mean_reward.png)
*Figure 3: Episode Reward Mean showing the steady, low-variance upward gradient ascent to the ~230 asymptote over 500k steps.*

![Constrained Episode Length](./results/constrained_episode_length.png)
*Figure 4: Episode Length Mean proving that the Action Mapping safety layer forces the probability distribution to optimize for long-term vehicle survivability rather than short-term point-gathering.*

---

## Phase 2: Architectural Pivot & Advanced Mathematical Modeling (In Progress)

To push the agent to high-performance racing speeds, the architecture is transitioning to a multi-objective optimization framework rooted in advanced probability and pure calculus.

### The Upgraded Training Pipeline (Estimated 1M - 2M Steps Convergence)
The next phase fundamentally upgrades the model's perception and strictly enforces safety guarantees to maximize sample efficiency.

**1. Perception & Brain Transfer (Knowledge Distillation)**
* **Temporal Frame Stacking:** Expanding the observation vector to 4 stacked frames to capture relative velocity and acceleration matrices of dynamic traffic.
* **Teacher-Student Distillation:** Seamlessly transferring the 500k-step expert policy into a larger `[256, 256]` neural network. This is achieved by mathematically extracting the probability distributions from the frozen Teacher model and training the new Student architecture to minimize the Kullback-Leibler (KL) Divergence:
$$D_{KL}(P \parallel Q) = \int P(x) \log\left(\frac{P(x)}{Q(x)}\right) dx$$

This solves the tensor dimension mismatch without losing the foundational driving policy.

**2. Safe RL: Dynamic Action Masking & Racing Calculus**
* **Probability Shielding (Action Masking):** Shifting from post-action penalties to proactive probability shielding. By analyzing the simulated LiDAR arrays, a strict mathematical mask intercepts the PPO probability distribution. It dynamically forces the probability of kinematically impossible or collision-bound actions to exactly `0.0` before the policy executes. 

This physically blocks the neural network from catastrophic exploration, yielding massive improvements in sample efficiency (convergence expected in 1-2 million steps rather than 10 million).
* **Mathematical Reward Shaping:** Implementing the composite continuous reward function from the MetaDrive base paper to penalize steering jerk and optimize the forward velocity vector:
$$R_t = c_1 R_{disp} + c_2 R_{speed} - c_3 P_{steering} + R_{term}$$

---

## Future Work
* **Multi-Agent Racing:** Expanding the framework to handle adversarial RL vehicles rather than standard traffic.
* **Data-Driven Dynamic Friction Adapting:** Modifying the Action Mapping constraint to account for variable weather or track surfaces dynamically by feeding real-time telemetry analytics into the wrapper.

## Acknowledgments & References

This project was developed to explore safe reinforcement learning architectures. To ensure academic integrity and open-source compliance, the following libraries, frameworks, and research papers are explicitly acknowledged:

### Core Frameworks
* **MetaDrive Simulator**: The core autonomous driving environment, LiDAR physics, and 3D rendering were provided by the [MetaDrive](https://github.com/metadriverse/metadrive) open-source project.
* **RL & Deep Learning**: The implementation of Proximal Policy Optimization (PPO) and the underlying neural network architectures were built using [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3) and PyTorch.
* **Environment Wrappers**: The Action Mapping and State Mapping safety mechanisms were engineered by extending the standard `gymnasium` (formerly OpenAI Gym) API.

### Academic Literature
* **Proximal Policy Optimization**: Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). *Proximal Policy Optimization Algorithms*. arXiv preprint arXiv:1707.06347.
* **Behavioral Cloning**: Pomerleau, D. A. (1989). *ALVINN: An Autonomous Land Vehicle in a Neural Network*. Advances in Neural Information Processing Systems.
* **MetaDrive Architecture**: Li, Q., Peng, Z., Zhou, B., et al. (2022). *MetaDrive: Composing Diverse Driving Scenarios for Generalizable Reinforcement Learning*. IEEE TPAMI.
* **Safe RL & Action Masking**: Recent advancements (2024-2025) in continuous action space reduction and dynamic masking for safety-critical autonomous control.

---

## License

This project's original code is licensed under the **MIT License**. 

*Note: Third-party libraries and environments (such as MetaDrive and Stable Baselines3) remain under their respective original open-source licenses.*
