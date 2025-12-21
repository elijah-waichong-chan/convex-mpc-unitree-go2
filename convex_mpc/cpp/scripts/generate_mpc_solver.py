#!/usr/bin/env python3
import numpy as np
import casadi as ca
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel


def build_friction_D(nu: int, mu: float) -> np.ndarray:
    """
    Build friction pyramid D so that h = D*u and we enforce:
        fx - mu fz <= 0
       -fx - mu fz <= 0
        fy - mu fz <= 0
       -fy - mu fz <= 0
    per leg (4 legs) => 16 rows total, nu=12 layout:
      [FLx, FLy, FLz, FRx, FRy, FRz, RLx, RLy, RLz, RRx, RRy, RRz]
    """
    assert nu == 12, "This helper assumes 4 legs * 3 forces = 12 inputs."
    D = np.zeros((16, nu))
    r = 0
    for leg in range(4):
        j = 3 * leg
        fx, fy, fz = j + 0, j + 1, j + 2

        # fx - mu fz <= 0
        D[r, fx] = 1.0
        D[r, fz] = -mu
        r += 1

        # -fx - mu fz <= 0
        D[r, fx] = -1.0
        D[r, fz] = -mu
        r += 1

        # fy - mu fz <= 0
        D[r, fy] = 1.0
        D[r, fz] = -mu
        r += 1

        # -fy - mu fz <= 0
        D[r, fy] = -1.0
        D[r, fz] = -mu
        r += 1

    return D


def make_ocp(nx: int, nu: int, N: int, dt: float, Q: np.ndarray, R: np.ndarray, mu: float) -> AcadosOcp:
    ocp = AcadosOcp()

    # -----------------------
    # Model: discrete linear, parametric A_k, B_k, g_d
    # x_{k+1} = A_k x_k + B_k u_k + g_d
    # -----------------------
    model = AcadosModel()
    model.name = "centroidal_mpc_hpipm"

    x = ca.SX.sym("x", nx)
    u = ca.SX.sym("u", nu)

    # Parameters p = [vec(A) ; vec(B) ; g_d]
    # IMPORTANT: CasADi reshape is column-major. In C++/Python runtime, flatten with order='F'.
    npA = nx * nx
    npB = nx * nu
    npg = nx
    np_stage = npA + npB + npg
    p = ca.SX.sym("p", np_stage)

    A = ca.reshape(p[0:npA], nx, nx)
    B = ca.reshape(p[npA:npA + npB], nx, nu)
    gd = p[npA + npB: npA + npB + npg]

    x_next = A @ x + B @ u + gd

    model.x = x
    model.u = u
    model.p = p
    model.disc_dyn_expr = x_next

    ocp.model = model
    ocp.parameter_values = np.zeros((np_stage,))
    ocp.solver_options.N_horizon = N
    ocp.dims.nx = nx
    ocp.dims.nu = nu
    ocp.dims.np = np_stage

    # -----------------------
    # Cost: LINEAR_LS
    # y = [x; u], yref_k = [x_ref_k; 0]
    # terminal: y_e = x, yref_e = x_ref_N (or your last ref)
    # -----------------------
    ny = nx + nu
    ocp.cost.cost_type = "LINEAR_LS"
    ocp.cost.cost_type_e = "LINEAR_LS"

    W = np.block([
        [Q, np.zeros((nx, nu))],
        [np.zeros((nu, nx)), R],
    ])
    ocp.cost.W = W
    ocp.cost.W_e = Q

    Vx = np.zeros((ny, nx))
    Vu = np.zeros((ny, nu))
    Vx[0:nx, 0:nx] = np.eye(nx)
    Vu[nx:nx + nu, 0:nu] = np.eye(nu)
    ocp.cost.Vx = Vx
    ocp.cost.Vu = Vu

    ocp.cost.Vx_e = np.eye(nx)

    # Placeholders; you will set these every MPC tick with cost_set(stage,"yref",...)
    ocp.cost.yref = np.zeros((ny,))
    ocp.cost.yref_e = np.zeros((nx,))

    # -----------------------
    # Constraints
    # 1) Fix x0 (update each tick by setting stage-0 lbx/ubx)
    # 2) Box bounds on u (swing legs u=0, stance fz>=fz_min) set at runtime
    # 3) Friction pyramid as general linear constraints D u <= ug (update ug per stage by contact)
    # -----------------------
    # (1) x0 bound at stage 0 only:
    ocp.constraints.idxbx_0 = np.arange(nx, dtype=np.int64)
    ocp.constraints.lbx_0 = np.zeros((nx,))
    ocp.constraints.ubx_0 = np.zeros((nx,))

    # (2) u bounds (enable bounds on all controls)
    ocp.constraints.idxbu = np.arange(nu, dtype=np.int64)  # lbu <= u[idxbu] <= ubu :contentReference[oaicite:2]{index=2}
    BIG = 1e8
    ocp.constraints.lbu = -BIG * np.ones((nu,))
    ocp.constraints.ubu =  BIG * np.ones((nu,))

    # (3) General linear constraints: lg <= D u <= ug
    Dmat = build_friction_D(nu=nu, mu=mu)
    ng = Dmat.shape[0]
    ocp.dims.ng = ng
    ocp.constraints.constr_type = "BGH"
    ocp.constraints.C = np.zeros((ng, nx))
    ocp.constraints.D = Dmat
    ocp.constraints.lg = -BIG * np.ones((ng,))
    ocp.constraints.ug = np.zeros((ng,))  # default: enforce friction everywhere; override per stage

    # no terminal general constraints
    ocp.dims.ng_e = 0
    ocp.constraints.C_e = np.zeros((0, nx))
    ocp.constraints.D_e = np.zeros((0, nu))
    ocp.constraints.lg_e = np.zeros((0,))
    ocp.constraints.ug_e = np.zeros((0,))

    # -----------------------
    # Solver options (HPIPM)
    # -----------------------
    ocp.solver_options.integrator_type = "DISCRETE"
    ocp.solver_options.tf = float(N * dt)

    # QP solver = HPIPM
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.qp_solver_cond_N = min(10, N)  # tune
    ocp.solver_options.qp_solver_warm_start = 1

    # RTI = one QP per MPC tick (good for linear/quadratic problems)
    ocp.solver_options.nlp_solver_type = "SQP_RTI"
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.print_level = 0

    return ocp


def main():
    import os
    import argparse
    import numpy as np

    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=20)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--json", type=str, default=None)
    args = parser.parse_args()

    NX = 12
    NU = 12
    MU = 0.8

    Q = np.diag([1, 1, 50, 10, 20, 1, 2, 2, 1, 1, 1, 1])
    R = np.diag([1e-5] * NU)

    ocp = make_ocp(nx=NX, nu=NU, N=args.N, dt=args.dt, Q=Q, R=R, mu=MU)

    # scripts/ is inside convex_mpc/cpp/scripts
    scripts_dir = os.path.dirname(os.path.abspath(__file__))
    cpp_dir     = os.path.abspath(os.path.join(scripts_dir, ".."))  # -> convex_mpc/cpp

    # Choose output directory
    if args.out is None:
        out_dir = os.path.join(cpp_dir, "c_generated_code", "centroidal_mpc_hpipm")
    else:
        out_dir = os.path.abspath(args.out)

    os.makedirs(out_dir, exist_ok=True)
    ocp.code_export_directory = out_dir

    # Choose json file path (keep it next to generated code by default)
    if args.json is None:
        json_path = os.path.join(out_dir, "centroidal_mpc_hpipm.json")
    else:
        json_path = os.path.abspath(args.json)

    # Generate solver C code into ocp.code_export_directory
    AcadosOcpSolver.generate(ocp, json_file=json_path, verbose=True)

    print(f"\n[acados] Generated code in: {out_dir}")
    print(f"[acados] JSON written to: {json_path}\n")

    print("Runtime updates you will do from C++ each MPC tick:")
    print("  - set stage params: solver.set(k, 'p', p_k)   (A,B,gd)")
    print("  - set yref:         solver.cost_set(k,'yref', yref_k)")
    print("  - set bounds:       solver.constraints_set(k,'lbu/ubu', ...)")
    print("  - set friction ug:  solver.constraints_set(k,'ug', ...)")
    print("  - fix x0:           solver.constraints_set(0,'lbx/ubx', x0)")

if __name__ == "__main__":
    main()
