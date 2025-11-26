# Motion Control of Unitree Go2 Quadruped Robot

A **contact-force–optimization MPC locomotion controller** for the Unitree Go2 quadruped robot.

Developed as part of the **UC Berkeley Master of Engineering (MEng)** capstone project in Mechanical Engineering.

As of 11/26/2025, the controller is capable acheive full 2D motion and yaw rotation


---

## 🐾 Introduction

This repository contains a full implementation of a **Convex Model Predictive Controller (MPC)** for the Unitree Go2 quadruped robot.  
The controller is designed following the methodology described in the MIT publication:

> **"Dynamic Locomotion in the MIT Cheetah 3 Through Convex Model-Predictive Control"**  
> https://dspace.mit.edu/bitstream/handle/1721.1/138000/convex_mpc_2fix.pdf

The objective of this project is to reproduce the main ideas presented in the paper—particularly the **contact-force MPC formulation**, convex optimization structure, and robust locomotion behavior—while integrating them into a modern, modular robotics control pipeline.

---
## ⚡ Locomotion Capabilities

The controller achieves the following performance in MuJoCo simulation using the convex MPC + leg controller pipeline:

### 🏃 Linear Motion
- **Forward speed:** up to **0.8 m/s**
- **Backward speed:** up to **0.8 m/s**
- **Lateral (sideways) speed:** up to **0.4 m/s**

### 🔄 Rotational Motion
- **Yaw rotational speed:** up to **4.0 rad/s**

### 🐾 Supported Gaits
- Trot gait (default: tested at 3.0 Hz and 0.6 duty)

## 🔧 Libraries Used

- **MuJoCo** — fast, stable **physics simulation** used for testing locomotion, foot contacts, and dynamic behaviors.
- **Pinocchio** — efficient **kinematics and dynamics library** for:
  - forward kinematics  
  - Jacobians  
  - frame placements
  - dynamics terms (M, C, g)

These libraries form the computational backbone of the control and simulation environment.

---

## 🦿 Controller Overview

Our motion control stack includes:

- **Centroidal MPC (~30-50 Hz)**  
    Contact-force–based MPC implemented via **CasADi**, solving a convex quadratic program each control cycle. The prediction horizon is set at one full gait cycle and divide into 16 time-step.

- **Reference Trajectory Generator (~30-50 Hz)**  
    Generates centroidal trajectory for MPC according to user input (simuilate a controller joystick input)

- **Swing/Stance Leg Controller (1000 Hz)**  
    Performs foot trajectory tracking using a PD controller during swing-phase and contact-force realization during stance-phase.

- **Gait Scheduler and Foot Trajectory Generator (1000 Hz)**  
    Determines stance/swing timing, compute touchdown position for swing-foot using Raibert style foot placement method and compute swing-leg trajectory using minimal jerk quintic polynomial with adjustable apex height.

---

## 🐍 Version Recommendation

- **Python:** `3.10.15`  
- **CasADi:** `3.6.7`  
- **NumPy:** `1.26.4`  
- **SciPy:** `1.15.2`  
- **Matplotlib:** `3.8.4`  
- **Pinocchio:** `3.6.0`  
- **MuJuCo:** `3.2.7`  

---
