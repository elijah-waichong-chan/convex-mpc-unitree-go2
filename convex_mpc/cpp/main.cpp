#include <iostream>
#include <iomanip>
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
    x0[2] = 0.10; // z error just for fun
    mpc.set_x0(x0);

    // Dummy dynamics: x_{k+1} = I x + 0 u
    auto A = eye12_colmajor();
    std::array<double, 12*12> B{}; // zeros
    std::array<double, 12> xref{}; // zeros

    // yref = [xref; uref]  (uref = 0)
    std::array<double, CentroidalMpcAcados::ny> yref{};
    // first 12 entries already 0, last 12 entries already 0

    int N = mpc.horizon();
    for (int k = 0; k < N; ++k) {
      auto p = CentroidalMpcAcados::pack_p(A, B, xref);
      mpc.set_stage_params(k, p);
      mpc.set_stage_yref(k, yref);
    }
    mpc.set_terminal_yref(xref);

    int status = mpc.solve();
    std::cout << "solve() status = " << status << "\n";

    std::array<double, 12> u0{};
    mpc.get_u0(u0);

    std::cout << "u0 = [";
    for (int i = 0; i < 12; ++i) {
      std::cout << std::fixed << std::setprecision(6) << u0[i] << (i+1<12 ? ", " : "");
    }
    std::cout << "]\n";

    return 0;
  } catch (const std::exception& e) {
    std::cerr << "ERROR: " << e.what() << "\n";
    return 1;
  }
}
