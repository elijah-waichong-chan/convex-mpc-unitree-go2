import casadi as ca
import numpy as np
import scipy.sparse as sp
from dynamics import Dynamics
from reference_trajectory import Reference_Traj
import time

class Locomotion_MPC:
    def __init__(self, dynamics: Dynamics, traj: Reference_Traj):
        self._build_QP(dynamics, traj)

    def solve_QP(self, dynamics: Dynamics, traj: Reference_Traj):

        t0 = time.perf_counter()
        # Get the latesting QP Matrcies
        [H, g, A, lba, uba] = self._build_sparse_matrix(dynamics, traj)
        # Compute Bounds
        [lbx, ubx] = self._compute_bounds(traj)
        t1 = time.perf_counter()
        t_compute = t1 - t0

        # Solve the QP
        t0 = time.perf_counter()
        sol = self.solver(h=H, g=g, a=A, lba=lba, uba=uba, lbx=lbx, ubx=ubx)
        t1 = time.perf_counter()
        t_solve = t1 - t0
        st = self.solver.stats()
        print(f"[QP SOLVER] update matrix takes {t_compute*1e3:.3f} ms")
        print(f"[QP SOLVER] solver takes {t_solve*1e3:.3f} ms")
        print(f"[QP SOLVER] total time = {(t_compute + t_solve)*1e3:.3f} ms  ({1.0/(t_compute + t_solve):.1f} Hz)")
        print(f"[QP SOLVER] status: {st.get('return_status')}")

        return sol

    def _build_sparse_matrix(self, dynamics: Dynamics, traj: Reference_Traj):

        # 0) Start build time timer
        t0 = time.perf_counter()

        # 1) System Constants
        nx, nu = 12, 12 # State and Input size
        N = dynamics.N # Prediction horizon
        nvars = N*nx + N*nu # Total number of decision variables
        mu = 0.7 # Static Coefficient
        Q = np.diag([1, 1, 50,  1, 1, 1,  1, 1, 1,  1, 1, 1]) # State cost weight matrix
        R = np.diag([1e-6] * nu) # Input cost weight matrix
        contact = traj.contact_schedule  # (4,N) Contact schedule mask

        # 2) Extract Dynamics for the horizon
        Ad = np.asarray(dynamics.Ad)
        Bd = np.asarray(dynamics.Bd) 
        gd = np.asarray(dynamics.gd).reshape(nx,1)
        x0 = traj.initial_simplified_state.compute_x_vec().reshape(nx,1)

        # 3) Extract Reference Trajectory Vector (12xN)
        x_ref_traj = traj.compute_x_ref_vec()

        # 4) Build the Hessian H for the QP
        rows, cols, vals = [], [], []

        # Helper function to construct the matrix
        def add(i,j,v):
            rows.append(i)
            cols.append(j)
            vals.append(float(v))

        # X blocks: k=0..N-1
        for k in range(N):
            base = k*nx
            for i in range(nx):
                if Q[i,i] != 0:
                    add(base+i, base+i, 2*Q[i,i])

        # U blocks: k=0..N-1
        for k in range(N):
            base = N*nx + k*nu
            for i in range(nu):
                if R[i,i] != 0:
                    add(base+i, base+i, 2*R[i,i])

        H = ca.DM.triplet(rows, cols, ca.DM(vals), nvars, nvars)


        # 5) Build the linear term (reference cost offset) vector g for the QP
        gx_blocks = []

        # X block: k=0..N-1
        for k in range(N):
            gx_blocks.append(-2 * ca.DM(Q) @ x_ref_traj[:, k])
        gx = ca.vertcat(*gx_blocks)

        # U block: k=0..N-1
        gu = ca.DM.zeros(nu * N, 1)
        g  = ca.vertcat(gx, gu)

        # 6) Build the constraints matrix A for the QP
        # Using Scipy for efficient construction
        I_n = sp.eye(nx, format='csc') # Identity matrix (12 x 12)
        I_N = sp.eye(N,  format='csc') # Identity matrix (N x N)
        S   = sp.diags([np.ones(N-1)], [-1], shape=(N,N), format='csc') # Matrix with ones only in the lower diagonal half

        # 6.1) Dynamics Constraints
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

        # 6.2) Friction Constraints
        rows, cols, vals = [], [], []
        l_ineq, u_ineq = [], []

        # helper function to index contact forces for each leg
        def leg_cols(leg): return 3*leg+0, 3*leg+1, 3*leg+2

        baseU = N*nx
        r0 = 0
        # Building the inequality matrix
        for k in range(N):
            uk0 = baseU + k*nu
            for leg in range(4):

                if contact[leg, k] != 1: 
                    continue
                fx, fy, fz = leg_cols(leg)

                # fx - mu fz <= 0
                rows += [r0, r0]
                cols += [uk0+fx, uk0+fz]
                vals += [1.0, -mu]
                l_ineq += [-np.inf]
                u_ineq += [0.0]
                r0 += 1

                # -fx - mu fz <= 0
                rows += [r0, r0]
                cols += [uk0+fx, uk0+fz]
                vals += [-1.0, -mu]
                l_ineq += [-np.inf]
                u_ineq += [0.0]
                r0 += 1

                # fy - mu fz <= 0
                rows += [r0, r0]
                cols += [uk0+fy, uk0+fz]
                vals += [1.0, -mu]
                l_ineq += [-np.inf]
                u_ineq += [0.0]
                r0 += 1
                # -fy - mu fz <= 0
                rows += [r0, r0]
                cols += [uk0+fy, uk0+fz]
                vals += [-1.0, -mu]
                l_ineq += [-np.inf]
                u_ineq += [0.0]
                r0 += 1

        # The inequality matrix which multiply with decision variables
        Aineq = sp.csc_matrix((vals, (rows, cols)), shape=(r0, N*(nx+nu)))
        # Lower bound and Upper bound of the inequality constraints
        lineq = np.array(l_ineq)
        uineq = np.array(u_ineq)

        # 6.3) Stack equalities + inequalities together
        # The final constraint A matrix
        A = sp.vstack([Aeq, Aineq], format='csc')
        # The final constraint lower bound and upper bound
        lb = np.concatenate([beq, lineq])
        ub = np.concatenate([beq, uineq])

        # 6.4) Convert SciPy to CasADi matrix
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

        # 7) Stop the build timer
        t1 = time.perf_counter()
        t_build = t1 - t0

        # 8) Print the time used
        print(f"[MATRIX BUILDER] duration = {t_build*1e3:.3f} ms  ({1.0/t_build:.1f} Hz)")

        return H, g, A_dm, l_dm, u_dm


    def _build_QP(self, dynamics: Dynamics, traj: Reference_Traj):

        # This function builds a Quadratic Program solver with sparsity structure only.
        # Subsequent update of the matries are necessary to run the optimization.
        
        # 0) Start the build timer
        t0 = time.perf_counter()

        # 1) Build the Hessian H matrix, linear term g vector, constraint A matrix
        # and constraint bounds lba, uba vectors
        [H, g, A, lba, uba] = self._build_sparse_matrix(dynamics, traj)

        # 2) Create a sparsity QP solver with the H, A sparsity structure
        # Define the QP structure
        qp = {'h': H.sparsity(), 'a': A.sparsity()}
        # Build the solver with a desired solver option (ipqp used here)
        self.solver = ca.conic('S', 'ipqp', qp)

        # 3) Stop the build timer
        t1 = time.perf_counter()
        t_build = t1 - t0

        # 4) Print Summary
        print("[QP BUILDER] ✅ QP solver built successfully.")
        print(f"[QP BUILDER] duration = {t_build*1e3:.3f} ms  ({1.0/t_build:.1f} Hz)")
        print(f"[QP BUILDER] Dimensions: #Decision Variables = {H.size1()}, #Constraints = {A.size1()}")
        print(f"[QP BUILDER] Sparsity: nnz(H) = {H.nnz()}, nnz(A) = {A.nnz()}\n")


    def _compute_bounds(self, traj: Reference_Traj):

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
        contact = np.asarray(traj.contact_schedule, dtype=bool)  # (4, N); True=stance, False=swing

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
