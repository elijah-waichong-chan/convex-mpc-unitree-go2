#include <iostream>
#include <iomanip>
#include <chrono>

#include "centroidal_mpc_acados.hpp"

static std::array<double, 12*12> eye12_colmajor() {
  std::array<double, 144> A{};
  for (int c = 0; c < 12; ++c) A[c*12 + c] = 1.0; // col-major
  return A;
}

int main() {
  try {
    CentroidalMpcAcados mpc;

    // Dummy x0
    std::array<double, 12> x0{};
    x0[2] = 0.10;
    mpc.set_x0(x0);

    // Dummy dynamics: x_{k+1} = I x + 0 u
    auto A = eye12_colmajor();
    std::array<double, 12*12> B{};     // zeros
    std::array<double, 12> gd{};       // NOTE: in your Python model this last block is gd, not xref

    // yref = [xref; uref] (uref = 0)
    std::array<double, CentroidalMpcAcados::ny> yref{};

    const int N = mpc.horizon();
    for (int k = 0; k < N; ++k) {
      auto p = CentroidalMpcAcados::pack_p(A, B, gd);
      mpc.set_stage_params(k, p);
      mpc.set_stage_yref(k, yref);
    }
    mpc.set_terminal_yref(gd);

    // ---- time one solve() ----
    const auto t0 = std::chrono::steady_clock::now();
    const int status = mpc.solve();
    const auto t1 = std::chrono::steady_clock::now();

    const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << "solve() status = " << status << "\n";
    std::cout << "solve() time   = " << std::fixed << std::setprecision(3) << ms << " ms\n";

    std::array<double, 12> u0{};
    mpc.get_u0(u0);

    std::cout << "u0 = [";
    for (int i = 0; i < 12; ++i) {
      std::cout << std::fixed << std::setprecision(6) << u0[i] << (i+1<12 ? ", " : "");
    }
    std::cout << "]\n";

    // ---- optional: run multiple solves for average/min/max ----
    {
      constexpr int iters = 200;
      double sum_ms = 0.0, min_ms = 1e9, max_ms = 0.0;

      for (int i = 0; i < iters; ++i) {
        const auto a0 = std::chrono::steady_clock::now();
        (void)mpc.solve();
        const auto a1 = std::chrono::steady_clock::now();
        const double dt_ms = std::chrono::duration<double, std::milli>(a1 - a0).count();
        sum_ms += dt_ms;
        min_ms = std::min(min_ms, dt_ms);
        max_ms = std::max(max_ms, dt_ms);
      }

      std::cout << "solve() over " << iters << " iters: "
                << "avg " << (sum_ms / iters) << " ms, "
                << "min " << min_ms << " ms, "
                << "max " << max_ms << " ms\n";
    }

    return 0;
  } catch (const std::exception& e) {
    std::cerr << "ERROR: " << e.what() << "\n";
    return 1;
  }
}
