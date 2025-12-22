#pragma once

#include <array>
#include <stdexcept>

// Generated acados solver header
extern "C" {
#include "acados_solver_centroidal_mpc_hpipm.h"

// These come from acados include/
#include "acados/ocp_nlp/ocp_nlp_common.h"
#include "acados_c/ocp_nlp_interface.h"
}

class CentroidalMpcAcados final {
public:
  static constexpr int nx = 12;
  static constexpr int nu = 12;
  static constexpr int np = 12 * 12 + 12 * 12 + 12; // A(144)+B(144)+x_ref(12)=300
  static constexpr int ny = nx + nu;

  // Constructor & Destructor
  explicit CentroidalMpcAcados();
  ~CentroidalMpcAcados();
  CentroidalMpcAcados(const CentroidalMpcAcados&) = delete;
  CentroidalMpcAcados& operator=(const CentroidalMpcAcados&) = delete;

  // ---- Vairables Assignment Methods ----
  void set_x0(const std::array<double, nx>& x0);

  // p_k = [vec(A); vec(B); x_ref] with column-major vec() consistent with CasADi reshape
  void set_stage_params(int stage, const std::array<double, np>& p_k);

  // yref_k length = ny for stages 0..N-1
  void set_stage_yref(int stage, const std::array<double, ny>& yref);

  // terminal yref_e length = nx at stage N
  void set_terminal_yref(const std::array<double, nx>& yref_e);

  // ----- solve / get -----
  int solve();
  void get_u0(std::array<double, nu>& u0) const;

  // Convenience: pack p from A,B,x_ref (all column-major)
  static std::array<double, np> pack_p(const std::array<double, nx*nx>& A_colmajor,
                                       const std::array<double, nx*nu>& B_colmajor,
                                       const std::array<double, nx>& x_ref);

private:
  centroidal_mpc_hpipm_solver_capsule* capsule_{nullptr};

  // pointers into acados
  ocp_nlp_config* nlp_config_{nullptr};
  ocp_nlp_dims*   nlp_dims_{nullptr};
  ocp_nlp_in*     nlp_in_{nullptr};
  ocp_nlp_out*    nlp_out_{nullptr};
};
