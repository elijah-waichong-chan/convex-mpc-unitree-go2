#include "centroidal_mpc_acados.hpp"
#include <cstring>

CentroidalMpcAcados::CentroidalMpcAcados() {
  capsule_ = centroidal_mpc_hpipm_acados_create_capsule();
  if (!capsule_) throw std::runtime_error("acados: create_capsule failed");

  int status = centroidal_mpc_hpipm_acados_create(capsule_);
  if (status) throw std::runtime_error("acados: acados_create failed with status " + std::to_string(status));

  nlp_config_ = centroidal_mpc_hpipm_acados_get_nlp_config(capsule_);
  nlp_dims_   = centroidal_mpc_hpipm_acados_get_nlp_dims(capsule_);
  nlp_in_     = centroidal_mpc_hpipm_acados_get_nlp_in(capsule_);
  nlp_out_    = centroidal_mpc_hpipm_acados_get_nlp_out(capsule_);

  if (!nlp_config_ || !nlp_dims_ || !nlp_in_ || !nlp_out_)
    throw std::runtime_error("acados: failed to get nlp pointers");
}

CentroidalMpcAcados::~CentroidalMpcAcados() {
  if (capsule_) {
    centroidal_mpc_hpipm_acados_free(capsule_);
    centroidal_mpc_hpipm_acados_free_capsule(capsule_);
    capsule_ = nullptr;
  }
}

void CentroidalMpcAcados::set_x0(const std::array<double, nx>& x0) {
  // fix x0 by setting lbx=ubx=x0 at stage 0
  ocp_nlp_constraints_model_set(nlp_config_, nlp_dims_, nlp_in_, nlp_out_, 0, "lbx", (void*)x0.data());
  ocp_nlp_constraints_model_set(nlp_config_, nlp_dims_, nlp_in_, nlp_out_, 0, "ubx", (void*)x0.data());
}
void CentroidalMpcAcados::set_stage_params(int stage, const std::array<double, np>& p_k) {
  // generated helper: update parameters at a stage
  // signature usually: <name>_acados_update_params(capsule, stage, p, np)
  int status = centroidal_mpc_hpipm_acados_update_params(capsule_, stage, const_cast<double*>(p_k.data()), np);
  if (status) {
    throw std::runtime_error("acados: update_params failed at stage " + std::to_string(stage)
                             + " with status " + std::to_string(status));
  }
}

void CentroidalMpcAcados::set_stage_yref(int stage, const std::array<double, ny>& yref) {
  ocp_nlp_cost_model_set(nlp_config_, nlp_dims_, nlp_in_, stage, "yref", (void*)yref.data());
}

void CentroidalMpcAcados::set_terminal_yref(const std::array<double, nx>& yref_e) {
  ocp_nlp_cost_model_set(nlp_config_, nlp_dims_, nlp_in_, N, "yref", (void*)yref_e.data());
}

int CentroidalMpcAcados::solve() {
  return centroidal_mpc_hpipm_acados_solve(capsule_);
}

void CentroidalMpcAcados::get_u0(std::array<double, nu>& u0) const {
  ocp_nlp_out_get(nlp_config_, nlp_dims_, nlp_out_, 0, "u", (void*)u0.data());
}

std::array<double, CentroidalMpcAcados::np>
CentroidalMpcAcados::pack_p(const std::array<double, nx*nx>& A_colmajor,
                            const std::array<double, nx*nu>& B_colmajor,
                            const std::array<double, nx>& x_ref) {
  std::array<double, np> p{};
  std::memcpy(p.data(), A_colmajor.data(), sizeof(double) * nx * nx);
  std::memcpy(p.data() + nx*nx, B_colmajor.data(), sizeof(double) * nx * nu);
  std::memcpy(p.data() + nx*nx + nx*nu, x_ref.data(), sizeof(double) * nx);
  return p;
}
