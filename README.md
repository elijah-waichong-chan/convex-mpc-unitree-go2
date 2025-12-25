# MPC Locomotion Control of the Unitree Go2 Quadruped Robot in MuJoCo
This repository implements a full **Convex Model Predictive Controller (MPC)** for the **Unitree Go2 quadruped robot**, including **contact-force optimization**, **centroidal dynamics**, and **MuJoCo simulation**.

Developed as part of the **UC Berkeley Master of Engineering (MEng)** capstone project in Mechanical Engineering.

> **Under development:** A C++ implementation targeting real-time performance is in progress. See the **Updates** section below for the latest status.

---

## Introduction

This repository contains a full implementation of a **Convex Model Predictive Controller (MPC)** in MuJoCo simulation with a Unitree Go2 quadruped robot .

The controller is designed following the methodology described in the following publication:

> **"Dynamic Locomotion in the MIT Cheetah 3 Through Convex Model-Predictive Control"**  
> https://dspace.mit.edu/bitstream/handle/1721.1/138000/convex_mpc_2fix.pdf

The objective of this project is to reproduce the main ideas presented in the paper — particularly the **contact-force MPC formulation**, convex optimization structure, and robust locomotion behavior—while integrating them into a modern, modular robotics control pipeline.

---
## Locomotion Capabilities

The controller achieves the following performance in MuJoCo simulation:

### Linear Motion
- **Forward speed:** up to **0.8 m/s**
- **Backward speed:** up to **0.8 m/s**
- **Lateral (sideways) speed:** up to **0.4 m/s**
<p align="center">
  <img src="media/forward_walking.gif" width="300">
  <img src="media/side_walking.gif" width="300">
</p>


### Rotational Motion
- **Yaw rotational speed:** up to **4.0 rad/s**
<p align="center"> <img src="media/yaw_rotation.gif" width="600"> </p>


### Supported Gaits
- Trot gait (tested at 3.0 Hz with 0.6 duty cycle)

## Controller Overview

Our motion control stack includes:

- **Centroidal MPC (~30-50 Hz)**  
Contact-force–based MPC implemented via **CasADi** using OSQP, solving a convex QP each cycle. The prediction horizon spans one full gait cycle, divided into 16 time steps.

- **Reference Trajectory Generator (~30-50 Hz)**  
Generates centroidal trajectory for MPC based on user input.

- **Swing/Stance Leg Controller (200 Hz)**  
    - Swing-phase: impedance control with foot trajectory and force tracking
    - Stance-phase: joint torque computation to realize MPC contact forces

- **Gait Scheduler and Foot Trajectory Generator (200 Hz)**  
    - Determines stance/swing timing
    - Compute touchdown position for swing-foot using Raibert style foot placement method and - - Compute swing-leg trajectory using minimal jerk quintic polynomial with adjustable apex height

---

## Libraries Used

### Physics Simulation
- **MuJoCo** — https://github.com/google-deepmind/mujoco *(tested: 3.2.7)*

### Kinematics and Dynamics
- **Pinocchio** — https://github.com/stack-of-tasks/pinocchio *(tested: 3.6.x)*

---

## Installation and Dependencies
### 1. Clone the repository
```bash
git clone https://github.com/elijah-waichong-chan/convex-mpc-unitree-go2.git
cd convex-mpc-unitree-go2
```

### 2. Create a Conda environment (Optional)
```bash
conda create -n go2 python=3.10 -y
conda activate go2
```

### 3. Install Dependencies

```bash
conda install pinocchio -c conda-forge
conda install casadi -c conda-forge
pip install mujoco==3.2.7
```

## Version Requirements

- **NumPy:** `>=1.24,<2.0`  


---

## Updates

12/24/2025
- Added URDF and MJCF model to the repo.
- Simplified installation steps, no longer need to download URDF models and unitree_mujoco library.

12/21/2025
- Reduced the overall controller loop from 1000 Hz → 200 Hz in preparation for real-time deployment; no observed performance degradation.
- The MPC update rate remains ~30–50 Hz, depending on gait frequency.

11/28/2025:
- Significantly faster model update and solving time per MPC iteration. Better matrix construction, implemented warm start, reduced redundant matrix update.
- Updated solve time plot style
- Updated motion demo in testMPC.py

11/26/2025:
- The controller is capable of full 2D motion and yaw rotation.
- The QP solve time for each MPC iteration are currently not capable of real-time control yet. This will be address in future updates with restructuring of the QP and more efficient matrix update.
- To adjust the cost matrix, go to centroidal_mpc.py
- To adjust the gait frequency and duty cycle, go to test_MPC.py
- To adjust the friction coefficient, go to centroidal_mpc.py, remember to change MuJoCo setting too.
- To adjust swing leg trajectory height, go to gait.py
- To adjust gait(phase offset), go to gait.py
- To adjust the desired motion, go to Trajectory Reference Setting in test_MPC.py
- To run the simulation and see the plotted results, run test_MPC.py
