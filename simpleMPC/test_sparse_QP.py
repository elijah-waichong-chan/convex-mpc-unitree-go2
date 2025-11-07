from go2_model import Pin_Go2_Model
from reference_trajectory import Reference_Traj
from dynamics import Dynamics
from locomotion_mpc import Locomotion_MPC
from plot_helper import plot_contact_forces, plot_traj_tracking
import numpy as np

# 1) Compute the dynamics and desired trajectory
go2 = Pin_Go2_Model()
traj = Reference_Traj()
dynamics = Dynamics()
dt = 0.03125

traj.generateConstantTraj(go2, x_vel_des=0.5, y_vel_des=0.2, z_pos_des=0.27, yaw_rate_des=0, 
                          t0=0, time_step=dt, time_horizon=0.5, frequency=2, duty=0.5)
dynamics.update_dynamics(go2, traj, dt)

# 2) Build an empty QP solver object
mpc = Locomotion_MPC(dynamics, traj)

# 3) Solve the QP with the latest states
sol = mpc.solve_QP(dynamics, traj)

# 4) Retrieve results
N = traj.N
w_opt = sol["x"].full().flatten()  # (n_w,)

X_opt = w_opt[: 12*(N)].reshape((12, N), order='F')
U_opt = w_opt[12*(N):].reshape((12, N), order='F')

# 5) Run simulation with optimal input
[x_now, x_sim] = dynamics.run_simulation(go2, U_opt)

# print(x_sim[:, 1:25] , X_opt)
pos_traj_sim = x_sim[0:3, :]
pos_traj_opt = X_opt[0:3, :]
x0_col = go2.current_config.get_simplified_full_state().compute_x_vec().reshape(-1, 1)
pos_traj_ref = np.hstack([x0_col[0:3, :], traj.compute_x_ref_vec()[0:3, :]])

# print(np.hstack([x0_col, traj.compute_x_ref_vec()]), x_traj)
plot_traj_tracking(pos_traj_ref, pos_traj_sim, block=False)
plot_traj_tracking(pos_traj_ref, pos_traj_opt, block=False)
plot_contact_forces(U_opt, traj.contact_schedule, dt, block=True)

