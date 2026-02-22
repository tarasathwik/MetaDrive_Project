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

### 1. Hybrid Learning Architecture

* **Stage I: Imitation Learning (Behavioral Cloning):**
  We minimize the divergence between the agent's policy and the expert's actions using the Behavioral Cloning loss:
  
  $$L_{BC}(\theta) = \mathbb{E}_{(s, a^*)\sim \mathcal{D}_{expert}} \left[ ||\pi_{\theta}(s) - a^*||^2_2 \right]$$

* **Stage II: Proximal Policy Optimization (PPO):**
  We optimize the policy using the clipped surrogate objective to ensure stable updates:
  
  $$L^{CLIP}(\theta) = \hat{\mathbb{E}}_t \left[ \min(r_t(\theta)\hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t) \right]$$

### 2. Safety Mechanisms

* **Action Mapping (Traction Control):**
  To prevent skidding, we enforce the **Friction Circle Constraint**. The total force applied by the agent must strictly adhere to the physical limits of tire friction.
  
  $$|\vec{F}_{x} + \vec{F}_{y}| < \mu_{max} F_{z}$$
  
  We implement a projection function $H_{AM}$ that maps unsafe actions back onto this safe boundary.

* **State Mapping (Dynamic Perception):**
  For competitive overtaking, we transform raw track observations into a "Feasible Area" vector based on opponent positions.
  
  $$\hat{V}_{E} = H_{SM}(V_{E}, V_{C})$$
  
  This allows the agent to perceive the track as a dynamic corridor, ignoring blocked lanes.

---

## Usage

### 1. Collect Expert Data
Run the manual control script to generate expert demonstrations for Imitation Learning.
```bash
python scripts/collect_data.py --episodes 10
2. Train the Agent (Hybrid Pipeline)After collecting the expert data, the training pipeline executes in two stages:Bash# Stage 1 & 2: Runs BC pre-training followed by PPO optimization
python scripts/train_rl_agent.py
3. Evaluate the ModelTo watch the fully trained agent navigate the MetaDrive environment with 3D rendering enabled:Bashpython scripts/test_final_agent.py
📊 Training Dynamics & FindingsThe model was trained for 500,000 steps following the initial Imitation Learning phase. The TensorBoard metrics revealed unique learning dynamics characteristic of high-speed autonomous racing:1. The Exploration-Exploitation Shift (Entropy)Unlike standard RL convergence where entropy strictly decreases, our Entropy Loss showed a steady incline during the transition from BC to PPO.Analysis: The agent began with artificially low entropy (high confidence) due to being heavily biased by the BC expert data. As PPO optimization engaged, the agent intentionally increased its action uncertainty to "unlearn" the conservative expert paths and actively explore aggressive, high-reward racing trajectories.Figure 1: Entropy expansion as the policy breaks away from the Imitation baseline to explore optimal racing lines.2. Policy Stability in High-Variance EnvironmentsThe Approximate KL Divergence exhibited high instability and significant jitter, reflecting the extreme sensitivity of the MetaDrive physics engine at high speeds.Analysis: In a racing context, minor steering deviations easily lead to "out-of-road" crashes, creating high-variance gradients. The PPO Clipped Surrogate Objective ($\epsilon = 0.2$) successfully bounded these massive policy updates, ensuring the policy did not completely collapse during complex, high-speed traffic negotiations.Figure 2: Approximate KL Divergence demonstrating the high-variance environment and active bounding of policy updates.3. Overall Performance GainDespite the aggressive exploration phase and high KL variance, the agent successfully optimized its racing lines, moving from a safe-but-slow BC baseline (~193 Mean Reward) to a highly optimized PPO peak.Figure 3: Episode Reward Mean showing the steady upward gradient ascent over 500k steps.Future WorkMulti-Agent Racing: Expanding the framework to handle adversarial RL vehicles rather than standard traffic.Dynamic Friction Adapting: Modifying the Action Mapping constraint to account for variable weather or track surfaces dynamically.

