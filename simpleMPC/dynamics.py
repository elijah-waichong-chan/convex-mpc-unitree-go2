import numpy as np
from numpy import sin, cos
from reference_trajectory import Reference_Traj
from scipy.signal import cont2discrete
from go2_model import Pin_Go2_Model
from scipy.linalg import expm

class Dynamics:

    def __init__(self):
        # Gravity Vector
        self.gc = np.array([
            0, 0, 0, 
            0, 0, 0, 
            0, 0, -9.81, 
            0, 0, 0 
            ])
        
    def update_dynamics(self, go2: Pin_Go2_Model, traj: Reference_Traj, dt):
        self._continuousDynamics(go2, traj)
        self._discreteDynamics(dt)

    def run_simulation(self, go2: Pin_Go2_Model, u_vec):

        current_state = go2.current_config.get_simplified_full_state()

        N_input = u_vec.shape[1] # Sequence of input given
        assert N_input == self.N, f"Expected {N_input=} to equal {self.N=}"

        x_traj = np.zeros((12, N_input+1))
        x_now = current_state.compute_x_vec()
        x_traj[:, [0]] = x_now

        for i in range(N_input):
            u_i   = u_vec[:, i].reshape(-1, 1)
            x_traj[:, i+1] = (self.Ad @ x_traj[:, [i]] + self.Bd[i] @ u_i + self.gd).flatten()

        return x_now, x_traj
        
    def _skew(self,vector):

        return np.array([
            [0, -vector[2], vector[1]],
            [vector[2], 0, -vector[0]],
            [-vector[1], vector[0], 0]
        ])

    def _continuousDynamics(self, go2: Pin_Go2_Model, traj: Reference_Traj):
        
        m = go2.data.Ig.mass
        I_world = go2.data.Ig.inertia # Get current rotational inertia and freeze for the horizon
        I_inv = np.linalg.inv(I_world)
        self.N = traj.N

        psi_avg = np.average(traj.yaw_ref)
        R_z = np.array([
            [cos(psi_avg), -sin(psi_avg), 0],
            [sin(psi_avg),  cos(psi_avg), 0],
            [0,             0,            1]
        ])

        self.Ac = np.block([
            [np.zeros((3, 3)), np.zeros((3, 3)), np.eye(3),        np.zeros((3, 3))],
            [np.zeros((3, 3)), np.zeros((3, 3)), np.zeros((3, 3)), R_z.T           ],
            [np.zeros((3, 3)), np.zeros((3, 3)), np.zeros((3, 3)), np.zeros((3, 3))],
            [np.zeros((3, 3)), np.zeros((3, 3)), np.zeros((3, 3)), np.zeros((3, 3))],
        ])

        self.Bc = np.zeros((self.N, 12, 12))
        for i in range(self.N):

            skew_r1 = self._skew(traj.fl_foot_placement_body[:, i])
            skew_r2 = self._skew(traj.fr_foot_placement_body[:, i])
            skew_r3 = self._skew(traj.rl_foot_placement_body[:, i])
            skew_r4 = self._skew(traj.rr_foot_placement_body[:, i])

            self.Bc[i] = np.block([
                [np.zeros((3, 3)),    np.zeros((3, 3)),    np.zeros((3, 3)),    np.zeros((3, 3))],
                [np.zeros((3, 3)),    np.zeros((3, 3)),    np.zeros((3, 3)),    np.zeros((3, 3))],
                [(1/m) * np.eye(3), (1/m) * np.eye(3), (1/m) * np.eye(3), (1/m) * np.eye(3)],
                [I_inv @ skew_r1,   I_inv @ skew_r2,   I_inv @ skew_r3,   I_inv @ skew_r4],

            ])

    def _discreteDynamics(self, dt):

        self.Bd = np.zeros((self.N, 12, 12))

        # Discretize Ac and Bc
        for i in range(self.N): 
            self.Ad, self.Bd[i], *_ = cont2discrete((self.Ac, self.Bc[i], np.eye(12), np.zeros((12, 12))), dt, method='zoh')

        # Discretize gc
        n_steps = 50
        tau = np.linspace(0, dt, n_steps)
        exp_terms = [expm(self.Ac * t) @ self.gc for t in tau]
        gd = np.trapz(np.stack(exp_terms, axis=1), tau, axis=1)

        self.gd = gd.reshape(-1, 1)