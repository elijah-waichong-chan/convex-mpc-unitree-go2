from __future__ import annotations

from dataclasses import dataclass
import numpy as np

# Local imports (same modules you used in the test file)
from .go2_robot_data import PinGo2Model
from .com_trajectory import ComTraj
from .centroidal_mpc import CentroidalMPC
from .leg_controller import LegController
from .gait import Gait

@dataclass
class BodyCmd:
    """Body command in the body frame."""
    x_vel: float = 0.0
    y_vel: float = 0.0
    z_pos: float = 0.27
    yaw_rate: float = 0.0

class Go2TorqueMPC:
    """
    Stateful wrapper around your existing stack:
      PinGo2Model + ComTraj + CentroidalMPC + LegController + Gait

    Call step() at CTRL_HZ. Internally it updates MPC at a slower rate and holds the latest solution.
    Returns joint torques in the same 12-dim ordering used by your test file:
      [FL hip, FL thigh, FL calf,
       FR hip, FR thigh, FR calf,
       RL hip, RL thigh, RL calf,
       RR hip, RR thigh, RR calf]
    """

    # Torque limits copied from your test file
    HIP_LIM = 23.7
    ABD_LIM = 23.7
    KNEE_LIM = 45.43

    LEG_SLICE = {
        "FL": slice(0, 3),
        "FR": slice(3, 6),
        "RL": slice(6, 9),
        "RR": slice(9, 12),
    }

    def __init__(
        self,
        gait_hz: float = 3.0,
        gait_duty: float = 0.6,
        ctrl_hz: float = 200.0,
        mpc_substeps_per_gait: int = 16,
        safety: float = 0.9,
        warm_start: bool = False,
    ):
        self.ctrl_hz = float(ctrl_hz)
        self.ctrl_dt = 1.0 / max(self.ctrl_hz, 1e-9)

        # Gait + MPC timing (matches your test logic)
        self.gait = Gait(gait_hz, gait_duty)
        self.gait_T = 1.0 / float(gait_hz)

        self.mpc_dt = self.gait_T / float(mpc_substeps_per_gait)
        self.mpc_hz = 1.0 / max(self.mpc_dt, 1e-9)
        self.steps_per_mpc = max(1, int(round(self.ctrl_hz / self.mpc_hz)))

        self.safety = float(safety)
        self.tau_lim = self.safety * np.array(
            [
                self.HIP_LIM, self.ABD_LIM, self.KNEE_LIM,  # FL
                self.HIP_LIM, self.ABD_LIM, self.KNEE_LIM,  # FR
                self.HIP_LIM, self.ABD_LIM, self.KNEE_LIM,  # RL
                self.HIP_LIM, self.ABD_LIM, self.KNEE_LIM,  # RR
            ],
            dtype=float,
        )

        # Core objects (persist across calls)
        self.go2 = PinGo2Model()
        self.leg_controller = LegController()
        self.traj = ComTraj(self.go2)
        self.mpc = CentroidalMPC(self.go2, self.traj)

        # Internal state
        self.ctrl_i = 0
        self.warm_start = bool(warm_start)

        # Safe defaults until first solve
        self.U_opt = None  # shape (12, N)
        self._init_solution()

    def _init_solution(self):
        # Create a "do-nothing" solution until first solve
        # We need traj.N; if ComTraj hasn't been generated yet, we will generate on first step
        self.U_opt = None

    def reset(self):
        """Reset control tick counters and clear any held MPC solution."""
        self.ctrl_i = 0
        self._init_solution()

    def update_model_from_pin(self, q_pin: np.ndarray, dq_pin: np.ndarray):
        """
        Update PinGo2Model from Pinocchio-format generalized coordinates.
        q_pin should be length 19 for floating base:
          [p(3), quat_xyzw(4), joint(12)]
        dq_pin should be length 18:
          [v_body(3), w_body(3), joint_vel(12)]
        """
        q_pin = np.asarray(q_pin, dtype=float).reshape(-1)
        dq_pin = np.asarray(dq_pin, dtype=float).reshape(-1)
        self.go2.update_model(q_pin, dq_pin)

    def _maybe_update_mpc(self, t_now: float, cmd: BodyCmd):
        """
        Update trajectory + solve QP at the configured MPC rate.
        Holds the latest U_opt and uses it until next solve.
        """
        do_update = (self.ctrl_i % self.steps_per_mpc) == 0

        if not do_update:
            return

        # Generate reference trajectory (same call signature as your test)
        self.traj.generate_traj(
            self.go2,
            self.gait,
            float(t_now),
            float(cmd.x_vel),
            float(cmd.y_vel),
            float(cmd.z_pos),
            float(cmd.yaw_rate),
            time_step=float(self.mpc_dt),
        )

        # Solve QP and unpack (same as your test)
        sol = self.mpc.solve_QP(self.go2, self.traj, self.warm_start)

        N = int(self.traj.N)
        w_opt = sol["x"].full().flatten()
        U_opt = w_opt[12 * (N):].reshape((12, N), order="F")  # (12, N)

        self.U_opt = U_opt

    def _compute_tau_from_grf(self, grf_world: np.ndarray, t_now: float) -> np.ndarray:
        """
        Map GRF (world) to joint torques using your leg controller (per-leg).
        Returns tau_raw (12,).
        """
        tau_raw = np.zeros(12, dtype=float)

        # FL
        FL = self.leg_controller.compute_leg_torque(
            "FL", self.go2, self.gait, grf_world[self.LEG_SLICE["FL"]], float(t_now)
        )
        tau_raw[self.LEG_SLICE["FL"]] = FL.tau

        # FR
        FR = self.leg_controller.compute_leg_torque(
            "FR", self.go2, self.gait, grf_world[self.LEG_SLICE["FR"]], float(t_now)
        )
        tau_raw[self.LEG_SLICE["FR"]] = FR.tau

        # RL
        RL = self.leg_controller.compute_leg_torque(
            "RL", self.go2, self.gait, grf_world[self.LEG_SLICE["RL"]], float(t_now)
        )
        tau_raw[self.LEG_SLICE["RL"]] = RL.tau

        # RR
        RR = self.leg_controller.compute_leg_torque(
            "RR", self.go2, self.gait, grf_world[self.LEG_SLICE["RR"]], float(t_now)
        )
        tau_raw[self.LEG_SLICE["RR"]] = RR.tau

        return tau_raw

    def step(
        self,
        t_now: float,
        q_pin: np.ndarray,
        dq_pin: np.ndarray,
        cmd: BodyCmd | None = None,
    ) -> np.ndarray:
        """
        One control tick. Call at CTRL_HZ.

        Inputs:
          t_now: time in seconds (monotonic or robot time)
          q_pin, dq_pin: floating-base state in Pin format (see update_model_from_pin doc)
          cmd: desired body motion (vx, vy, z, yaw_rate)

        Output:
          tau_cmd: (12,) torques clipped to limits
        """
        if cmd is None:
            cmd = BodyCmd()

        # Update model
        self.update_model_from_pin(q_pin, dq_pin)

        # Make sure we have an initialized trajectory/solution at least once
        if self.U_opt is None:
            # Generate an initial traj + solve once so U_opt exists
            self._maybe_update_mpc(t_now, cmd)
            if self.U_opt is None:
                # If solve failed somehow, safe zeros
                self.ctrl_i += 1
                return np.zeros(12, dtype=float)

        # Update MPC on schedule
        self._maybe_update_mpc(t_now, cmd)

        # Use first GRF from latest solution
        grf0_world = self.U_opt[:, 0] if self.U_opt is not None else np.zeros(12, dtype=float)

        # Map GRF -> joint torques
        tau_raw = self._compute_tau_from_grf(grf0_world, t_now)

        # Saturate
        tau_cmd = np.clip(tau_raw, -self.tau_lim, self.tau_lim)

        self.ctrl_i += 1
        return tau_cmd


# --- Convenience singleton API (easy import for your ROS2 bridge) ---
_solver: Go2TorqueMPC | None = None


def get_solver() -> Go2TorqueMPC:
    global _solver
    if _solver is None:
        _solver = Go2TorqueMPC()
    return _solver


def solve_torques(
    t_now: float,
    q_pin: np.ndarray,
    dq_pin: np.ndarray,
    cmd: BodyCmd | None = None,
) -> np.ndarray:
    """
    Convenience function: uses a singleton solver instance.
    """
    return get_solver().step(t_now=t_now, q_pin=q_pin, dq_pin=dq_pin, cmd=cmd)
