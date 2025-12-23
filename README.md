# MPC Locomotion Control of the Unitree Go2 Quadruped Robot in MuJoCo
This repository implements a full **Convex Model Predictive Controller (MPC)** for the **Unitree Go2 quadruped robot**, including **contact-force optimization**, **centroidal dynamics**, and **MuJoCo simulation**.

Developed as part of the **UC Berkeley Master of Engineering (MEng)** capstone project in Mechanical Engineering.

*This repo is still under development. Breaking changes are expected. Check "Updates" section below for latest updates.

---

## 🐾 Introduction

This repository contains a full implementation of a **Convex Model Predictive Controller (MPC)** for the Unitree Go2 quadruped robot in MuJoCo simulation.

The controller is designed following the methodology described in the MIT publication:

> **"Dynamic Locomotion in the MIT Cheetah 3 Through Convex Model-Predictive Control"**  
> https://dspace.mit.edu/bitstream/handle/1721.1/138000/convex_mpc_2fix.pdf

The objective of this project is to reproduce the main ideas presented in the paper — particularly the **contact-force MPC formulation**, convex optimization structure, and robust locomotion behavior—while integrating them into a modern, modular robotics control pipeline.

---
## ⚡ Locomotion Capabilities

The controller achieves the following performance in MuJoCo simulation using the convex MPC + leg controller pipeline:

### 🏃 Linear Motion
- **Forward speed:** up to **0.8 m/s**
- **Backward speed:** up to **0.8 m/s**
- **Lateral (sideways) speed:** up to **0.4 m/s**
<p align="center">
  <img src="media/forward_walking.gif" width="300">
  <img src="media/side_walking.gif" width="300">
</p>


### 🔄 Rotational Motion
- **Yaw rotational speed:** up to **4.0 rad/s**
<p align="center"> <img src="media/yaw_rotation.gif" width="600"> </p>


### 🐾 Supported Gaits
- Trot gait (tested at 3.0 Hz with 0.6 duty cycle)

## 🦿 Controller Overview

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

## 🔧 Libraries Used

- **MuJoCo** — fast, stable **physics simulation** used for testing locomotion, foot contacts, and dynamic behaviors.
- **Pinocchio** — efficient **kinematics and dynamics library** for:
  - forward kinematics  
  - Jacobians  
  - frame placements
  - dynamics terms (M, C, g)

- **unitree_mujoco** — Unitree’s MuJoCo asset + URDF package 
https://github.com/unitreerobotics/unitree_mujoco

Together, these libraries form the computational backbone of the control and simulation environment.

---

## Installation and Dependencies
### 1. Clone the repository
```bash
git clone https://github.com/elijah-waichong-chan/convex-mpc-unitree-go2.git
cd convex-mpc-unitree-go2
```

### 2. Create a Conda environment
```bash
conda create -n go2 python=3.10.15 -y
conda activate go2
```

### 3. Download & copy Unitree MuJoCo assets into the repo
This project depends on the official Unitree MuJoCo models to run simulation.
```bash
git clone https://github.com/unitreerobotics/unitree_mujoco.git
cp -r unitree_mujoco ./third_party/unitree_mujoco
```

Your repo structure should now look like:
```
convex-mpc-unitree-go2/
└── third_party/
    └── unitree_mujoco/
```

Then update the GO2 foot friction in the MuJoCo model:
#### 1.Open:
```
third_party/unitree_mujoco/unitree_robots_go2/go2.xml
```
#### 2. Go to line 33 (the contact/geom friction definition for the feet) and change it to:
```xml
friction="0.8 0.02 0.01"/>
```

### 4. Download & copy Unitree GO2 URDF into the repo

The Pinocchio model requires the official GO2 URDF and its meshes.  
Unitree provides them as a downloadable ZIP archive.

Download the GO2 URDF package:

```bash
wget https://oss-global-cdn.unitree.com/static/Go2_URDF.zip
unzip Go2_URDF.zip
```

Copy the GO2 URDF into the project:
```bash
cp -r Go2_URDF/go2_description ./third_party/unitree_go2_description
```

Your directory structure should now include:
```
convex-mpc-unitree-go2/
└── third_party/
    ├── unitree_mujoco/
    └── unitree_go2_description/
```

### 5. Install MuJoCo
MuJoCo 3.2.7, instruction to be added.

### 6. Install Pinocchio
Pinocchio is required for kinematics, dynamics, and centroidal model computations.

Install via conda:
```bash
conda install pinocchio -c conda-forge
```

### 7. Install CasAdi
CasADi is used for building and solving the MPC optimization problems.

Install via conda:
```bash
conda install casadi -c conda-forge
```

## 🐍 Version Recommendation

- **Python:** `3.10.15`  
- **CasADi:** `3.6.7`  
- **NumPy:** `1.26.4`  
- **SciPy:** `1.15.2`  
- **Matplotlib:** `3.8.4`  
- **Pinocchio:** `3.6.0`  
- **MuJuCo:** `3.2.7`  

---

## Updates

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
