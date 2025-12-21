#include <casadi/casadi.hpp>
#include <iostream>

#include "centroidal_mpc.hpp"

int main() {
    casadi::DM A = casadi::DM::eye(3);
    std::cout << "CasADi DM eye(3):\n" << A << "\n";

    CentroidalMPC mpc;
    mpc.hello();

    return 0;
}