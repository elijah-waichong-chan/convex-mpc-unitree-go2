import casadi as ca
import numpy as np
import scipy.sparse as sp
from com_trajectory import ComTraj
from go2_robot_data import PinGo2Model
import time

# --------------------------------------------------------------------------------
# Model Predictive Control Setting
# --------------------------------------------------------------------------------

COST_MATRIX_Q = np.diag([1, 1, 50,  10, 10, 1,  5, 5, 1,  1, 1, 1])     # State cost weight matrix
COST_MATRIX_R = np.diag([1e-6] * 12)                                    # Input cost weight matrix

MU = 0.8    # Friction coefficient
NX = 12     # State size (6-DOF 12 states)
NU = 12     # Input size (4 x 3D force)
M = 1000    # Used for trivial friction constraint for swing leg

# Solver Option
OPTS = {
    "osqp": {
        "eps_abs": 1e-4,
        "eps_rel": 1e-4,
        "max_iter": 10000,
        "polish": False,
        "verbose": False,
        "scaling": 10,
        "scaled_termination": True
    }
}

SOLVER_NAME: str = "osqp"


class CentroidalMPC:
    def __init__(self, go2:PinGo2Model, traj: ComTraj):
        self.Q = COST_MATRIX_Q 
        self.R = COST_MATRIX_R 
        self.nvars = traj.N * NX + traj.N * NU    # Total number of decision variables                          
        self.solve_time: float = 0 

        self.H_sp = None    # CasAdi Sparsity
        self.A_sp = None    # CasAdi Sparsity
        self.H_const = None # CasAdi Matrix
        self._build_QP(traj, True)

    def solve_QP(self, go2:PinGo2Model, traj: ComTraj, verbose: bool = False):

        t0 = time.perf_counter()

        # 1) Update the QP
        [g, A, lba, uba] = self._update_sparse_matrix(traj)        # update QP matrcies
        [lbx, ubx] = self._compute_bounds(traj)                    # Compute decision variables bounds
        t_compute = time.perf_counter() - t0
        # 2) Solve the QP
        sol = self.solver(h=self.H_const, g=g, a=A, lba=lba, uba=uba, lbx=lbx, ubx=ubx)
        t_solve = time.perf_counter() - t_compute - t0
        self.solve_time = (t_compute + t_solve) * 1e3

        # 3) Print Summary
        if verbose:
            stats = self.solver.stats()
            print(f"[QP SOLVER] update matrix takes {t_compute*1e3:.3f} ms")
            print(f"[QP SOLVER] solver takes {t_solve*1e3:.3f} ms")
            print(f"[QP SOLVER] total time = {(t_compute + t_solve)*1e3:.3f} ms  ({1.0/(t_compute + t_solve):.1f} Hz)")
            print(f"[QP SOLVER] status: {stats.get('return_status')}")
        return sol
    
    def _build_QP(self, traj: ComTraj, verbose: bool = False):

        # --------------------------------------------------------------------------------
        # This function builds a Quadratic Program solver with sparsity structure only.
        # Subsequent update of the matries are necessary to run the optimization.
        # --------------------------------------------------------------------------------
        
        t0 = time.perf_counter()

        # 1) Build and store the Hessian H matrix and constraint A matrix
        [H0, A0] = self._build_sparse_matrix(traj)
        self.H_sp = H0.sparsity()
        self.A_sp = A0.sparsity()
        self.H_const = H0

        # 2) Create a sparsity QP solver with the H, A sparsity structure
        qp = {'h': self.H_sp, 'a': self.A_sp}
        self.solver = ca.conic('S', SOLVER_NAME, qp, OPTS)   # osqp solver used
        t_build = time.perf_counter() - t0

        # 3) Print Summary
        if verbose:
            print("[QP BUILDER] ✅ QP solver built successfully.")
            print(f"[QP BUILDER] duration = {t_build*1e3:.3f} ms  ({1.0/t_build:.1f} Hz)")
            print(f"[QP BUILDER] Dimensions: #Decision Variables = {H0.size1()}, #Constraints = {A0.size1()}")
            print(f"[QP BUILDER] Sparsity: nnz(H) = {H0.nnz()}, nnz(A) = {A0.nnz()}\n")

    def _build_sparse_matrix(self, traj: ComTraj):

        # --------------------------------------------------------------------------------
        # This method only returns H and A matrix for building sparsity solver purpose
        # --------------------------------------------------------------------------------

        # 1) Extract Parameters
        N = traj.N
        nvars = self.nvars
        contact_table = traj.contact_table  # (4xN) Contact schedule mask
        Ad = np.asarray(traj.Ad)
        Bd = np.asarray(traj.Bd) 

        # 2) Build the Hessian H for the QP
        rows, cols, vals = [], [], []

        # Helper function to construct the matrix
        def add(i,j,v):
            rows.append(i)
            cols.append(j)
            vals.append(float(v))

        # X blocks: k=0..N-1
        for k in range(N):
            base = k*NX
            for i in range(NX):
                if self.Q[i,i] != 0:
                    add(base+i, base+i, 2*self.Q[i,i])

        # U blocks: k=0..N-1
        for k in range(N):
            base = N*NX + k*NU
            for i in range(NU):
                if self.R[i,i] != 0:
                    add(base+i, base+i, 2*self.R[i,i])

        H = ca.DM.triplet(rows, cols, ca.DM(vals), nvars, nvars)

        # 3) Build the constraint matrix A for the QP
        # Using Scipy for efficient construction
        I_n = sp.eye(NX, format='csc') # Identity matrix (12 x 12)
        I_N = sp.eye(N,  format='csc') # Identity matrix (N x N)
        S   = sp.diags([np.ones(N-1)], [-1], shape=(N,N), format='csc') # Matrix with ones only in the lower diagonal half

        # 4.1) Dynamics Constraints
        # Dynamics - X block (multiply with Ad)
        AX = sp.kron(I_N, I_n, format='csc') + sp.kron(S, -Ad, format='csc')
        # Dynamics - U block (muktiply with Bd)
        AU = sp.block_diag([ -sp.csc_matrix(Bd[k]) for k in range(N) ], format='csc')

        # Form the Dynamics Constaints as 
            # x_k+1 - g_d = Ad*x_k + B_d*u_k
        # LHS matrix of the equality that multiply with decision variables
        Aeq = sp.hstack([AX, AU], format='csc')

        # 4.2) Friction Constraints
        rows, cols, vals = [], [], []
        l_ineq, u_ineq = [], []

        # helper function to index contact forces for each leg
        def leg_cols(leg): return 3*leg+0, 3*leg+1, 3*leg+2

        baseU = N*NX    # Index helper
        r0 = 0
        # Building the inequality matrix
        for k in range(N):
            uk0 = baseU + k*NU
            for leg in range(4):

                c = 1 if contact_table[leg, k] == 1 else 0
                fx, fy, fz = leg_cols(leg)

                # fx - mu fz <= M*(1-c)
                rows += [r0, r0]
                cols += [uk0+fx, uk0+fz]
                vals += [1.0, -MU]
                l_ineq += [-np.inf]
                u_ineq += [M*(1-c)]
                r0 += 1

                # -fx - mu fz <= M*(1-c)
                rows += [r0, r0]
                cols += [uk0+fx, uk0+fz]
                vals += [-1.0, -MU]
                l_ineq += [-np.inf]
                u_ineq += [M*(1-c)]
                r0 += 1

                # fy - mu fz <= M*(1-c)
                rows += [r0, r0]
                cols += [uk0+fy, uk0+fz]
                vals += [1.0, -MU]
                l_ineq += [-np.inf]
                u_ineq += [M*(1-c)]
                r0 += 1

                # -fy - mu fz <= M*(1-c)
                rows += [r0, r0]
                cols += [uk0+fy, uk0+fz]
                vals += [-1.0, -MU]
                l_ineq += [-np.inf]
                u_ineq += [M*(1-c)]
                r0 += 1

        # The inequality matrix which multiply with decision variables
        Aineq = sp.csc_matrix((vals, (rows, cols)), shape=(r0, N*(NX+NU)))

        # 4.3) Stack equalities + inequalities together
        # The final constraint A matrix
        A = sp.vstack([Aeq, Aineq], format='csc')

        # 4.4) Convert SciPy to CasADi matrix
        # helper function
        def scipy_to_casadi_csc(M):
            M = M.tocsc()
            spz = ca.Sparsity(M.shape[0], M.shape[1], M.indptr.tolist(), M.indices.tolist())
            return ca.DM(spz, M.data)

        # CasAdi constraint A matrix
        A_dm = scipy_to_casadi_csc(A)

        return H, A_dm
    
    def _update_sparse_matrix(self, traj: ComTraj):

        # --------------------------------------------------------------------------------
        # This method returns time-varying cost g vector, constraint matrix A and its bounds
        # --------------------------------------------------------------------------------

        # 1) Extract updated parameters
        N = traj.N # Prediction horizon
        contact_table = traj.contact_table  # (4,N) Contact schedule mask
        Ad = np.asarray(traj.Ad)
        Bd = np.asarray(traj.Bd) 
        gd = np.asarray(traj.gd).reshape(NX,1)
        x0 = traj.initial_x_vec
        x_ref_traj = traj.compute_x_ref_vec()

        # 2) Build the linear term vector g for the QP
        gx_blocks = []

        # X block: k=0..N-1
        for k in range(N):
            gx_blocks.append(-2 * ca.DM(self.Q) @ x_ref_traj[:, k])
        gx = ca.vertcat(*gx_blocks)

        # U block: k=0..N-1
        gu = ca.DM.zeros(NU * N, 1)
        g  = ca.vertcat(gx, gu)

        # 3) Build the constraints matrix A for the QP
        # Using Scipy for efficient construction
        I_n = sp.eye(NX, format='csc') # Identity matrix (12 x 12)
        I_N = sp.eye(N,  format='csc') # Identity matrix (N x N)
        S   = sp.diags([np.ones(N-1)], [-1], shape=(N,N), format='csc') # Matrix with ones only in the lower diagonal half

        # 3.1) Dynamics Constraints
        # Dynamics - X block (multiply with Ad)
        AX = sp.kron(I_N, I_n, format='csc') + sp.kron(S, -Ad, format='csc')
        # Dynamics - U block (muktiply with Bd)
        AU = sp.block_diag([ -sp.csc_matrix(Bd[k]) for k in range(N) ], format='csc')

        # Form the Dynamics Constaints as 
            # x_k+1 - g_d = Ad*x_k + B_d*u_k
        # LHS matrix of the equality that multiply with decision variables
        Aeq = sp.hstack([AX, AU], format='csc')
        # RHS vector of the equality (first row is special because of initial condition)
        beq = np.vstack([Ad @ x0 + gd] + [gd]*(N-1)).ravel()

        # 3.2) Friction Constraints
        rows, cols, vals = [], [], []
        l_ineq, u_ineq = [], []

        # helper function to index contact forces for each leg
        def leg_cols(leg): return 3*leg+0, 3*leg+1, 3*leg+2

        baseU = N*NX
        r0 = 0
        # Building the inequality matrix
        for k in range(N):
            uk0 = baseU + k * NU
            for leg in range(4):

                c = 1 if contact_table[leg, k] == 1 else 0
                fx, fy, fz = leg_cols(leg)

                # fx - mu fz <= M*(1-c)
                rows.extend([r0, r0])
                cols.extend([uk0 + fx, uk0 + fz])
                vals.extend([1.0, -MU])
                l_ineq.append(-np.inf)
                u_ineq.append(M * (1 - c))
                r0 += 1

                # -fx - mu fz <= M*(1-c)
                rows.extend([r0, r0])
                cols.extend([uk0 + fx, uk0 + fz])
                vals.extend([-1.0, -MU])
                l_ineq.append(-np.inf)
                u_ineq.append(M * (1 - c))
                r0 += 1

                # fy - mu fz <= M*(1-c)
                rows.extend([r0, r0])
                cols.extend([uk0 + fy, uk0 + fz])
                vals.extend([1.0, -MU])
                l_ineq.append(-np.inf)
                u_ineq.append(M * (1 - c))
                r0 += 1

                # -fy - mu fz <= M*(1-c)
                rows.extend([r0, r0])
                cols.extend([uk0 + fy, uk0 + fz])
                vals.extend([-1.0, -MU])
                l_ineq.append(-np.inf)
                u_ineq.append(M * (1 - c))
                r0 += 1

        # The inequality matrix which multiply with decision variables
        Aineq = sp.csc_matrix((vals, (rows, cols)), shape=(r0, N*(NX+NU)))
        # Lower bound and Upper bound of the inequality constraints
        lineq = np.array(l_ineq)
        uineq = np.array(u_ineq)

        # 3.3) Stack equalities + inequalities together
        # The final constraint A matrix
        A = sp.vstack([Aeq, Aineq], format='csc')
        # The final constraint lower bound and upper bound
        lb = np.concatenate([beq, lineq])
        ub = np.concatenate([beq, uineq])

        # 3.4) Convert SciPy to CasADi matrix
        # helper function
        def scipy_to_casadi_csc(M):
            M = M.tocsc()
            spz = ca.Sparsity(M.shape[0], M.shape[1], M.indptr.tolist(), M.indices.tolist())
            return ca.DM(spz, M.data)

        # CasAdi constraint A matrix
        A_dm = scipy_to_casadi_csc(A)

        # CasAdi constraint upper bound and lower bound
        l_dm = ca.DM(lb)
        u_dm = ca.DM(ub)

        return g, A_dm, l_dm, u_dm


    def _compute_bounds(self, traj: ComTraj):

        # --------------------------------------------------------------------------------
        # This method returns decision variable bounds
        # --------------------------------------------------------------------------------

        fz_min = 10
        N = traj.N

        n_w = N*12 + N*12
        start_u = N*12

        # 1) Start with infinities once
        lbx_np = np.full((n_w, 1), -np.inf, dtype=float)
        ubx_np = np.full((n_w, 1),  np.inf, dtype=float)

        # 2) Precompute force indices for all (leg, axis, k)
        # Layout of forces per timestep: [FLx, FLy, FLz, FRx, FRy, FRz, RLx, RLy, RLz, RRx, RRy, RRz]
        force_block = (np.arange(12)[:, None] + 12*np.arange(N)[None, :])  # (12, N)
        force_idx   = start_u + force_block                                # (12, N)

        # 3) Contact mask
        contact = np.asarray(traj.contact_table, dtype=bool)  # (4, N); True=stance, False=swing

        # Indices for each leg's (x,y,z) rows within the 12 rows
        leg_rows = np.array([[0, 1, 2],
                            [3, 4, 5],
                            [6, 7, 8],
                            [9,10,11]])                                   # (4, 3)

        # 4) Swing legs → all three components = 0
        swing = ~contact                                                   # (4, N)
        # expand swing to the three axes:
        swing_xyz = np.repeat(swing[:, None, :], 3, axis=1)               # (4, 3, N)

        # map to (12, N) mask
        mask_12N = np.zeros((12, N), dtype=bool)
        mask_12N[leg_rows.reshape(-1), :] = swing_xyz.reshape(12, N)

        swing_idx = force_idx[mask_12N]                                   # (num_swing_vars,)
        lbx_np[swing_idx, 0] = 0.0
        ubx_np[swing_idx, 0] = 0.0

        # 5) Stance legs → fz >= fz_min (only the z row of each leg: rows 2,5,8,11)
        fz_rows = np.array([2, 5, 8, 11])                                 # (4,)
        stance_idx_2d = force_idx[fz_rows[:, None], np.arange(N)[None, :]]  # (4, N)
        stance_mask   = contact                                           # (4, N)
        stance_idx    = stance_idx_2d[stance_mask]                        # (num_stance,)

        # keep the tighter lower bound if any
        lbx_np[stance_idx, 0] = np.maximum(lbx_np[stance_idx, 0], fz_min)

        # 6) Convert once to CasADi
        lbx = ca.DM(lbx_np)
        ubx = ca.DM(ubx_np)

        return lbx, ubx
