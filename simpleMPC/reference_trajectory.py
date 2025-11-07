import numpy as np
from go2_model import Pin_Go2_Model
from dataclasses import dataclass

@dataclass
class Reference_Traj:
    time: np.ndarray = np.empty(0)
    x_pos_ref: np.ndarray = np.empty(0)
    y_pos_ref: np.ndarray = np.empty(0)
    z_pos_ref: np.ndarray = np.empty(0)
    x_vel_ref: np.ndarray = np.empty(0)
    y_vel_ref: np.ndarray = np.empty(0)
    z_vel_ref: np.ndarray = np.empty(0)
    roll_ref: np.ndarray = np.empty(0)
    pitch_ref: np.ndarray = np.empty(0)
    yaw_ref: np.ndarray = np.empty(0)
    roll_rate_ref: np.ndarray = np.empty(0)
    pitch_rate_ref: np.ndarray = np.empty(0)
    yaw_rate_ref: np.ndarray = np.empty(0)
    fl_foot_placement: np.ndarray = np.empty(0)
    fr_foot_placement: np.ndarray = np.empty(0)
    rl_foot_placement: np.ndarray = np.empty(0)
    rr_foot_placement: np.ndarray = np.empty(0)
    fl_foot_placement_vec: np.ndarray = np.empty(0)
    fr_foot_placement_vec: np.ndarray = np.empty(0)
    rl_foot_placement_vec: np.ndarray = np.empty(0)
    rr_foot_placement_vec: np.ndarray = np.empty(0)

    def compute_x_ref_vec(self):
        refs = [
            self.x_pos_ref,
            self.y_pos_ref,
            self.z_pos_ref,
            self.roll_ref,
            self.pitch_ref,
            self.yaw_ref,
            self.x_vel_ref,
            self.y_vel_ref,
            self.z_vel_ref,
            self.roll_rate_ref,
            self.pitch_rate_ref,
            self.yaw_rate_ref,
        ]
        # stack into shape (12, N)
        N = min(len(r) for r in refs)
        ref_traj = np.vstack([r[:N] for r in refs])
        return ref_traj
    
    def debug_print_ref_sizes(self):

        refs = {
            "x_pos_ref": self.x_pos_ref,
            "y_pos_ref": self.y_pos_ref,
            "z_pos_ref": self.z_pos_ref,
            "x_vel_ref": self.x_vel_ref,
            "y_vel_ref": self.y_vel_ref,
            "z_vel_ref": self.z_vel_ref,
            "roll_ref": self.roll_ref,
            "pitch_ref": self.pitch_ref,
            "yaw_ref": self.yaw_ref,
            "roll_rate_ref": self.roll_rate_ref,
            "pitch_rate_ref": self.pitch_rate_ref,
            "yaw_rate_ref": self.yaw_rate_ref,
        }

        print("🔍 Reference variable sizes:")
        for name, r in refs.items():
            arr = np.asarray(r)
            print(f"{name:15s}  type={type(r).__name__:10s}  shape={arr.shape}  size={arr.size}")


    def gait_planner(self,
                    frequency_hz: float,
                    duty: float,
                    time_step: float,
                    time_now: float,
                    time_horizon: float) -> np.ndarray:
        
        N = self.N
        t = time_now + np.arange(N) * time_step # time vector
        T = 1 / frequency_hz # Perioid

        phase_offset = np.array([0.0, 0.5, 0.5, 0.0])
        
        phases = np.mod(t[None, :] / T + phase_offset[:, None], 1.0)
        self.contact_schedule = (phases < duty).astype(np.int32)  # 4 x N
        print(self.contact_schedule)

    def raibertFootPlacement(self, p_com, x_vel, y_vel, frequency, duty):
        period = 1/frequency
        stanceTime = duty * period
        swingTime = (1-duty) * period

        p_com[2] = 0

        r_next_touchdown_world = p_com + np.array([x_vel*(swingTime + stanceTime/2), y_vel*(swingTime + stanceTime/2), 0])
        return r_next_touchdown_world

    def generateConstantTraj(self,
                             go2: Pin_Go2_Model,
                             x_vel_des: float,
                             y_vel_des: float,
                             z_pos_des: float,
                             yaw_rate_des: float,
                             t0: float,
                             time_step: float,
                             time_horizon: float,
                             frequency: float,
                             duty: float):
        
        self.initial_simplified_state = go2.current_config.get_simplified_full_state()
        
        self.N = int(time_horizon / time_step) # number of sequences to output
        N = self.N
        t_vec = np.arange(N) * time_step # time vector
        t_vec = t_vec + time_step

        self.time = t_vec

        self.x_pos_ref = go2.current_config.x + x_vel_des * t_vec
        self.x_vel_ref = np.full(N, x_vel_des, dtype=float)

        self.y_pos_ref = go2.current_config.y + y_vel_des * t_vec
        self.y_vel_ref = np.full(N, y_vel_des, dtype=float)

        self.z_pos_ref = np.full(N, z_pos_des, dtype=float)
        self.z_vel_ref = np.full(N, 0, dtype=float)

        [yaw, _, _] = go2.current_config.get_euler_angle()
        self.yaw_ref = yaw + yaw_rate_des * t_vec
        self.yaw_rate_ref = np.full(N, yaw_rate_des, dtype=float)

        self.pitch_ref = np.full(N, 0, dtype=float)
        self.pitch_rate_ref = np.full(N, 0, dtype=float)

        self.roll_ref = np.full(N, 0, dtype=float)
        self.roll_rate_ref = np.full(N, 0, dtype=float)

        [r_fl_next, r_fr_next, r_rl_next, r_rr_next] = go2.get_foot_placement_in_body()

        com_world = np.array([go2.current_config.x, go2.current_config.y, go2.current_config.z])

        self.gait_planner(frequency, duty, time_step, t0, time_horizon)
        # print(self.contact_schedule)

        r_fl = np.zeros((3,N))
        r_fr = np.zeros((3,N))
        r_rl = np.zeros((3,N))
        r_rr = np.zeros((3,N))

        r_fl_body = np.zeros((3,N))
        r_fr_body = np.zeros((3,N))
        r_rl_body = np.zeros((3,N))
        r_rr_body = np.zeros((3,N))

        fl_hip_offset = go2.FL_hip_offset
        fr_hip_offset = go2.FR_hip_offset
        rl_hip_offset = go2.RL_hip_offset
        rr_hip_offset = go2.RR_hip_offset

        mask_previous = np.array([2,2,2,2])


        for i in range(N):
            current_mask = self.contact_schedule[:, i]
            p_com = np.array([com_world[0] + x_vel_des*i*time_step, com_world[1] + y_vel_des*i*time_step, com_world[2]])
            # print(p_com)

            if current_mask[0] != mask_previous[0]:
                if current_mask[0] == 0:
                    r_fl_next = fl_hip_offset + self.raibertFootPlacement(p_com, x_vel_des, y_vel_des, frequency, duty)
                    r_fl[:,i] = np.array([0,0,0])
                    r_fl_body[:,i] = np.array([0,0,0])
                elif current_mask[0] == 1:
                    r_fl[:,i] = r_fl_next
            else:
                r_fl[:,i] = r_fl[:,i-1]

            if current_mask[0] == 1:
                r_fl_body[:,i] = r_fl[:,i] - p_com

            if current_mask[1] != mask_previous[1]:
                if current_mask[1] == 0:
                    r_fr_next = fr_hip_offset + self.raibertFootPlacement(p_com, x_vel_des, y_vel_des, frequency, duty)
                    r_fr[:,i] = np.array([0,0,0])
                    r_fr_body[:,i] = np.array([0,0,0])
                elif current_mask[1] == 1:
                    r_fr[:,i] = r_fr_next
            else:
                r_fr[:,i] = r_fr[:,i-1]
            if current_mask[1] == 1:
                r_fr_body[:,i] = r_fr[:,i] - p_com

            if current_mask[2] != mask_previous[2]:
                if current_mask[2] == 0:
                    r_rl_next = rl_hip_offset + self.raibertFootPlacement(p_com, x_vel_des, y_vel_des, frequency, duty)
                    r_rl[:,i] = np.array([0,0,0])
                    r_rl_body[:,i] = np.array([0,0,0])
                elif current_mask[2] == 1:
                    r_rl[:,i] = r_rl_next
            else:
                r_rl[:,i] = r_rl[:,i-1]
            if current_mask[2] == 1:
                r_rl_body[:,i] = r_rl[:,i] - p_com

            if current_mask[3] != mask_previous[3]:
                if current_mask[3] == 0:
                    r_rr_next = rr_hip_offset + self.raibertFootPlacement(p_com, x_vel_des, y_vel_des, frequency, duty)
                    r_rr[:,i] = np.array([0,0,0])
                    r_rr_body[:,i] = np.array([0,0,0])
                elif current_mask[3] == 1:
                    r_rr[:,i] = r_rr_next
            else:
                r_rr[:,i] = r_rr[:,i-1]
            if current_mask[3] == 1:
                r_rr_body[:,i] = r_rr[:,i] - p_com

            mask_previous = current_mask

        self.fl_foot_placement = r_fl
        self.fr_foot_placement = r_fr
        self.rl_foot_placement = r_rl
        self.rr_foot_placement = r_rr

        self.fl_foot_placement_body = r_fl_body
        self.fr_foot_placement_body = r_fr_body
        self.rl_foot_placement_body = r_rl_body
        self.rr_foot_placement_body = r_rr_body