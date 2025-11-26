import time
import mujoco as mj
import numpy as np
from dataclasses import dataclass

from go2_robot_data import PinGo2Model
from mujoco_model import MuJoCo_GO2_Model
from com_trajectory import ComTraj
from centroidal_mpc import CentroidalMPC
from leg_controller import LegController
from gait import Gait

from plot_helper import plot_mpc_result, plot_swing_foot_traj, plot_full_traj, plot_solve_time

# --------------------------------------------------------------------------------
# Parameters
# --------------------------------------------------------------------------------

# Simulation Setting
INITIAL_X_POS = -5                  # The initial x-position of the robot
INITIAL_Y_POS = 0                   # The initial y-position of the robot
RUN_SIM_LENGTH_S = 5                # Adjust this to change the duration of simulation in seconds
RENDER_HZ = 120.0                   # Adjust this to change the replay redering rate
RENDER_DT = 1.0 / RENDER_HZ         # Time step of the simulation replay
REALTIME_FACTOR = 1                 # Adjust this to change the replay speed (1 = realtime)

# Gait Setting
GAIT_HZ = 3             # Adjust this to change the frequency of the gait
GAIT_DUTY = 0.6         # Adjust this to change the duty of the gait
GAIT_T = 1.0 / GAIT_HZ  # Peirod of the gait

# Trajectory Reference Setting
x_vel_des_body = 0         # Adjust this to change the desired forward velocity
y_vel_des_body = 0.4           # Adjust this to change the desired lateral velocity
z_pos_des_body = 0.27        # Adjust this to change the desired height
yaw_rate_des_body = 0       # Adjust this to change the desired roatation velocity

# Leg Controller Loop Setting
LEG_CTRL_HZ = 1000                                      # Leg controller (output torque) update rate
LEG_CTRL_DT = 1.0 / LEG_CTRL_HZ                         # Time-step of the leg controller (1000 Hz)
LEG_CTRL_I_END = int(RUN_SIM_LENGTH_S/LEG_CTRL_DT)      # Last iteration number of the simulation
leg_ctrl_i = 0                                          # Iteration counter

# Relation between MPC loop and Leg controller loop
MPC_DT = GAIT_T / 16                                    # Time step of the MPC Controlnqler as 1/16 of the gait period
MPC_HZ = 1/MPC_DT                                       # MPC update rate
STEPS_PER_MPC = max(1, int(LEG_CTRL_HZ // MPC_HZ))      # Number of steps the leg controller runs before the MPC is called

TAU_MAX = 45

LEG_SLICE = {
    "FL": slice(0, 3),
    "FR": slice(3, 6),
    "RL": slice(6, 9),
    "RR": slice(9, 12),
}

# --------------------------------------------------------------------------------
# Storage Variables
# --------------------------------------------------------------------------------

# Centriodal State x = [px, py, pz, roll, pitch, yaw, vx, vy, vz, wx, wy, wz]
# frame: world, units: m, rad, m/s, rad/s
x_vec = np.zeros((12, LEG_CTRL_I_END))

# MPC contact force log: [FLx, FLy, FLz, FRx, FRy, FRz, RLx, RLy, RLz, RRx, RRy, RRz]
# frame: world, units: N
mpc_force_world = np.zeros((12, LEG_CTRL_I_END))

# Leg controller output (before saturation): joint torques per leg
# layout: [FL_hip, FL_thigh, FL_calf, FR_hip, ..., RR_calf], units: Nm
tau_raw = np.zeros((12, LEG_CTRL_I_END))

# Applied motor torques after saturation (what we actually send to MuJoCo)
# same layout as tau_raw, units: N·m
tau_cmd = np.zeros((12, LEG_CTRL_I_END))

# Storage variables for MuJoCo for replaying purpose
time_log_s = np.zeros(LEG_CTRL_I_END)           # Log simulation time
q_log = np.zeros((LEG_CTRL_I_END, 19))          # Log robot configuration
tau_log_Nm = np.zeros((LEG_CTRL_I_END, 12))     # Log robot joint torque

# Storage variables for foot trajectory
@dataclass
class FootTraj:
        pos_des = np.zeros((12, LEG_CTRL_I_END))
        pos_now = np.zeros((12, LEG_CTRL_I_END))
        vel_des = np.zeros((12, LEG_CTRL_I_END))
        vel_now = np.zeros((12, LEG_CTRL_I_END))
foot_traj = FootTraj()

mpc_solve_time_s = []     # Time takes to solve the MPC QP
X_opt = []                # Optimal trajectory from the MPC
U_opt = []                # Optimal contact force from the MPC

# --------------------------------------------------------------------------------
# Simulation Initialization
# --------------------------------------------------------------------------------

# Create classes instance
go2 = PinGo2Model()                 # Current robot object in Pinocchio
mujoco_go2 = MuJoCo_GO2_Model()     # Current robot object in MuJoCo
leg_controller = LegController()    # Leg controller
traj = ComTraj(go2)           # Reference trajectory over the horizon for each MPC iteration
gait = Gait(GAIT_HZ, GAIT_DUTY)     # Gait setup and swing-leg trajectory planning


# Initialize the robot configuration
q_init = go2.current_config.get_q()                     # Get the current robot configuration
q_init[0], q_init[1] = INITIAL_X_POS, INITIAL_Y_POS     # Set the initial x-position
# q_init[3:7] = [0.0, 0.0, -0.159318, 0.987219]              # [qx, qy, qz, qw]
mujoco_go2.update_with_q_pin(q_init)                    # Update the MuJoCo model with the current Pinocchio configration
mujoco_go2.model.opt.timestep = LEG_CTRL_DT             # Time-step of the MuJoCo environment (1000 Hz) 

# Create a sparsity MPC QP solver
traj.generate_traj(go2, gait, 0, x_vel_des_body, y_vel_des_body, z_pos_des_body, yaw_rate_des_body, time_step=MPC_DT)
mpc = CentroidalMPC(go2, traj)      # Creates the mpc object

# Start simulation
print(f"Running simulation for {RUN_SIM_LENGTH_S}s")
sim_start_time = time.perf_counter()

while leg_ctrl_i < LEG_CTRL_I_END:

    time_now_s = mujoco_go2.data.time   # Current time in simulation

    # if(time_now_s) < 2:
    #     x_vel_des_body = 0
    #     y_vel_des_body = 0
    #     z_pos_des_body = 0.27
    #     yaw_rate_des_body = 0.3

    # elif(time_now_s) < 3:
    #     x_vel_des_body = 0
    #     y_vel_des_body = 0
    #     z_pos_des_body = 0.27
    #     yaw_rate_des_body = 0

    # if(time_now_s) < 4:
    #     x_vel_des_body = 0.5
    #     y_vel_des_body = 0
    #     z_pos_des_body = 0.27
    #     yaw_rate_des_body = 0

    # elif(time_now_s) < 6:
    #     x_vel_des_body = 0
    #     y_vel_des_body = 0
    #     z_pos_des_body = 0.27
    #     yaw_rate_des_body = 0

    # elif(time_now_s) < 5:
    #     x_vel_des = 0
    #     y_vel_des = 0
    #     z_pos_des = 0.27
    #     yaw_rate_des = -0.6

    # elif(time_now_s) < 6:
    #     x_vel_des = 0
    #     y_vel_des = 0
    #     z_pos_des = 0.27
    #     yaw_rate_des = 0

    # elif(time_now_s) < 7:
    #     x_vel_des = 0
    #     y_vel_des = 0
    #     z_pos_des = 0.27
    #     yaw_rate_des = 0

    # elif(time_now_s) < 8:
    #     x_vel_des = 0.3
    #     y_vel_des = -0.3
    #     z_pos_des = 0.27
    #     yaw_rate_des = 0

    # elif(time_now_s) < 9:
    #     x_vel_des = 0
    #     y_vel_des = 0
    #     z_pos_des = 0.27
    #     yaw_rate_des = 0

    # 1) Update Pinocchio model with MuJuCo data
    mujoco_go2.update_pin_with_mujoco(go2)
    x_vec[:, leg_ctrl_i] = go2.compute_com_x_vec().reshape(-1)

    # 2) Log current robot configuration in MuJoCo
    time_log_s[leg_ctrl_i]    = time_now_s
    q_log[leg_ctrl_i, :] = mujoco_go2.data.qpos
    # ------------------------------

    # 3) Update MPC if needed
    if (leg_ctrl_i % STEPS_PER_MPC) == 0:
        print("Time:", time_now_s)

        # 3.1) Update reference trajectory 
        traj.generate_traj(go2, gait, time_now_s, 
                                x_vel_des_body, y_vel_des_body,
                                z_pos_des_body, yaw_rate_des_body, 
                                time_step=MPC_DT)
        
        # 3.2) Solve the QP with the latest states
        sol = mpc.solve_QP(go2, traj)
        mpc_solve_time_s.append(mpc.solve_time)

        # 3.3) Retrieve results
        N = traj.N
        w_opt = sol["x"].full().flatten()
        X_opt = w_opt[: 12*(N)].reshape((12, N), order='F')
        U_opt = w_opt[12*(N):].reshape((12, N), order='F')


    # 4) Extract the first optimized GRF
    mpc_force_world[:, leg_ctrl_i] = U_opt[:, 0] 

    # 5) Compute joint torques
    FL_leg_output = leg_controller.compute_leg_torque("FL", go2, gait, mpc_force_world[LEG_SLICE["FL"], leg_ctrl_i], time_now_s)
    tau_raw[LEG_SLICE["FL"], leg_ctrl_i] = FL_leg_output.tau
    foot_traj.pos_des[LEG_SLICE["FL"], leg_ctrl_i] = FL_leg_output.pos_des
    foot_traj.pos_now[LEG_SLICE["FL"], leg_ctrl_i] = FL_leg_output.pos_now
    foot_traj.vel_des[LEG_SLICE["FL"], leg_ctrl_i] = FL_leg_output.vel_des
    foot_traj.vel_now[LEG_SLICE["FL"], leg_ctrl_i] = FL_leg_output.vel_now

    FR_leg_output = leg_controller.compute_leg_torque("FR", go2, gait, mpc_force_world[LEG_SLICE["FR"], leg_ctrl_i], time_now_s)
    tau_raw[LEG_SLICE["FR"], leg_ctrl_i] = FR_leg_output.tau
    foot_traj.pos_des[LEG_SLICE["FR"], leg_ctrl_i] = FR_leg_output.pos_des
    foot_traj.pos_now[LEG_SLICE["FR"], leg_ctrl_i] = FR_leg_output.pos_now
    foot_traj.vel_des[LEG_SLICE["FR"], leg_ctrl_i] = FR_leg_output.vel_des
    foot_traj.vel_now[LEG_SLICE["FR"], leg_ctrl_i] = FR_leg_output.vel_now

    RL_leg_output = leg_controller.compute_leg_torque("RL", go2, gait, mpc_force_world[LEG_SLICE["RL"], leg_ctrl_i], time_now_s)
    tau_raw[LEG_SLICE["RL"], leg_ctrl_i] = RL_leg_output.tau
    foot_traj.pos_des[LEG_SLICE["RL"], leg_ctrl_i] = RL_leg_output.pos_des
    foot_traj.pos_now[LEG_SLICE["RL"], leg_ctrl_i] = RL_leg_output.pos_now
    foot_traj.vel_des[LEG_SLICE["RL"], leg_ctrl_i] = RL_leg_output.vel_des
    foot_traj.vel_now[LEG_SLICE["RL"], leg_ctrl_i] = RL_leg_output.vel_now

    RR_leg_output = leg_controller.compute_leg_torque("RR", go2, gait, mpc_force_world[LEG_SLICE["RR"], leg_ctrl_i], time_now_s)
    tau_raw[LEG_SLICE["RR"], leg_ctrl_i] = RR_leg_output.tau
    foot_traj.pos_des[LEG_SLICE["RR"], leg_ctrl_i] = RR_leg_output.pos_des
    foot_traj.pos_now[LEG_SLICE["RR"], leg_ctrl_i] = RR_leg_output.pos_now
    foot_traj.vel_des[LEG_SLICE["RR"], leg_ctrl_i] = RR_leg_output.vel_des
    foot_traj.vel_now[LEG_SLICE["RR"], leg_ctrl_i] = RR_leg_output.vel_now

    # 6) Apply motor saturation
    tau_cmd[:, leg_ctrl_i] = np.clip(tau_raw[:, leg_ctrl_i], -TAU_MAX, TAU_MAX)
    # 7) Apply the computed torque to MuJoCo
    mj.mj_step1(mujoco_go2.model, mujoco_go2.data)
    mujoco_go2.set_joint_torque(tau_cmd[:, leg_ctrl_i])
    mj.mj_step2(mujoco_go2.model, mujoco_go2.data)
    # 8) Log the applied torque values
    tau_log_Nm[leg_ctrl_i,:] = tau_cmd[:, leg_ctrl_i]

    leg_ctrl_i += 1

sim_end_time = time.perf_counter()
print(f"Simulation ended. Duration: {sim_end_time - sim_start_time}s")

# --------------------------------------------------------------------------------
# Simulation Results
# --------------------------------------------------------------------------------

# Plot results
t_vec = np.arange(LEG_CTRL_I_END) * LEG_CTRL_DT
plot_mpc_result(t_vec, mpc_force_world, tau_cmd, x_vec, block=False)
plot_swing_foot_traj(t_vec, foot_traj, False)
plot_solve_time(mpc_solve_time_s, MPC_DT, MPC_HZ, False)

# Replay simulation
mujoco_go2.replay_simulation(time_log_s, q_log, tau_log_Nm, RENDER_DT, REALTIME_FACTOR)

# # # # # 5) Run simulation with optimal input
x0_col = go2.compute_com_x_vec()
traj_ref = np.hstack([x0_col, traj.compute_x_ref_vec()])
traj_act = np.hstack([x0_col, X_opt])
plot_full_traj(traj_act, traj_ref, block=True)

# [x_now, x_sim] = go2.run_simulation(U_opt)
# pos_traj_sim = x_sim[0:3, :]
# pos_traj_opt = X_opt[0:3, :]
# pos_traj_ref = np.hstack([x0_col[0:3, :], traj.compute_x_ref_vec()[0:3, :]])
# plot_traj_tracking(pos_traj_ref, pos_traj_sim, block=True)
# plot_traj_tracking(pos_traj_ref, pos_traj_opt, block=True)

