# AI-Based Autonomous Driving Simulation with PPO & Imitation Learning
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
  
  <div align="center">
    <img src="https://latex.codecogs.com/svg.latex?L_{BC}(\theta)%20=%20\mathbb{E}_{(s,%20a^*)\sim%20\mathcal{D}_{expert}}%20\left[%20||\pi_{\theta}(s)%20-%20a^*||^2_2%20\right]" title="Behavioral Cloning Loss" />
  </div>

* **Stage II: Proximal Policy Optimization (PPO):**
  We optimize the policy using the clipped surrogate objective to ensure stable updates:
  
  <div align="center">
    <img src="https://latex.codecogs.com/svg.latex?L^{CLIP}(\theta)%20=%20\hat{\mathbb{E}}_t%20\left[%20\min(r_t(\theta)\hat{A}_t,%20\text{clip}(r_t(\theta),%201-\epsilon,%201+\epsilon)\hat{A}_t)%20\right]" title="PPO Loss" />
  </div>

### 2. Safety Mechanisms

* **Action Mapping (Traction Control):**
  To prevent skidding, we enforce the **Friction Circle Constraint**. The total force applied by the agent must strictly adhere to the physical limits of tire friction .
  
  <div align="center">
    <img src="https://latex.codecogs.com/svg.latex?|\vec{F}_{x}%20+%20\vec{F}_{y}|%20<%20\mu_{max}%20F_{z}" title="Friction Circle Constraint" />
  </div>
  
  We implement a projection function **H_AM** that maps unsafe actions back onto this safe boundary.

* **State Mapping (Dynamic Perception):**
  For competitive overtaking, we transform raw track observations into a "Feasible Area" vector based on opponent positions .
  
  <div align="center">
    <img src="https://latex.codecogs.com/svg.latex?\hat{V}_{E}%20=%20H_{SM}(V_{E},%20V_{C})" title="State Mapping Function" />
  </div>
  
  This allows the agent to perceive the track as a dynamic corridor, ignoring blocked lanes.

---

## Usage

### 1. Collect Expert Data
Run the manual control script to generate expert demonstrations for Imitation Learning.
```bash
python scripts/collect_data.py --episodes 10

