/*
 * Copyright (c) The acados authors.
 *
 * This file is part of acados.
 *
 * The 2-Clause BSD License
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.;
 */

// standard
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
// acados
// #include "acados/utils/print.h"
#include "acados_c/ocp_nlp_interface.h"
#include "acados_c/external_function_interface.h"

// example specific

#include "centroidal_mpc_hpipm_model/centroidal_mpc_hpipm_model.h"





#include "acados_solver_centroidal_mpc_hpipm.h"

#define NX     CENTROIDAL_MPC_HPIPM_NX
#define NZ     CENTROIDAL_MPC_HPIPM_NZ
#define NU     CENTROIDAL_MPC_HPIPM_NU
#define NP     CENTROIDAL_MPC_HPIPM_NP
#define NP_GLOBAL     CENTROIDAL_MPC_HPIPM_NP_GLOBAL
#define NY0    CENTROIDAL_MPC_HPIPM_NY0
#define NY     CENTROIDAL_MPC_HPIPM_NY
#define NYN    CENTROIDAL_MPC_HPIPM_NYN

#define NBX    CENTROIDAL_MPC_HPIPM_NBX
#define NBX0   CENTROIDAL_MPC_HPIPM_NBX0
#define NBU    CENTROIDAL_MPC_HPIPM_NBU
#define NG     CENTROIDAL_MPC_HPIPM_NG
#define NBXN   CENTROIDAL_MPC_HPIPM_NBXN
#define NGN    CENTROIDAL_MPC_HPIPM_NGN

#define NH     CENTROIDAL_MPC_HPIPM_NH
#define NHN    CENTROIDAL_MPC_HPIPM_NHN
#define NH0    CENTROIDAL_MPC_HPIPM_NH0
#define NPHI   CENTROIDAL_MPC_HPIPM_NPHI
#define NPHIN  CENTROIDAL_MPC_HPIPM_NPHIN
#define NPHI0  CENTROIDAL_MPC_HPIPM_NPHI0
#define NR     CENTROIDAL_MPC_HPIPM_NR

#define NS     CENTROIDAL_MPC_HPIPM_NS
#define NS0    CENTROIDAL_MPC_HPIPM_NS0
#define NSN    CENTROIDAL_MPC_HPIPM_NSN

#define NSBX   CENTROIDAL_MPC_HPIPM_NSBX
#define NSBU   CENTROIDAL_MPC_HPIPM_NSBU
#define NSH0   CENTROIDAL_MPC_HPIPM_NSH0
#define NSH    CENTROIDAL_MPC_HPIPM_NSH
#define NSHN   CENTROIDAL_MPC_HPIPM_NSHN
#define NSG    CENTROIDAL_MPC_HPIPM_NSG
#define NSPHI0 CENTROIDAL_MPC_HPIPM_NSPHI0
#define NSPHI  CENTROIDAL_MPC_HPIPM_NSPHI
#define NSPHIN CENTROIDAL_MPC_HPIPM_NSPHIN
#define NSGN   CENTROIDAL_MPC_HPIPM_NSGN
#define NSBXN  CENTROIDAL_MPC_HPIPM_NSBXN



// ** solver data **

centroidal_mpc_hpipm_solver_capsule * centroidal_mpc_hpipm_acados_create_capsule(void)
{
    void* capsule_mem = malloc(sizeof(centroidal_mpc_hpipm_solver_capsule));
    centroidal_mpc_hpipm_solver_capsule *capsule = (centroidal_mpc_hpipm_solver_capsule *) capsule_mem;

    return capsule;
}


int centroidal_mpc_hpipm_acados_free_capsule(centroidal_mpc_hpipm_solver_capsule *capsule)
{
    free(capsule);
    return 0;
}


int centroidal_mpc_hpipm_acados_create(centroidal_mpc_hpipm_solver_capsule* capsule)
{
    int N_shooting_intervals = CENTROIDAL_MPC_HPIPM_N;
    double* new_time_steps = NULL; // NULL -> don't alter the code generated time-steps
    return centroidal_mpc_hpipm_acados_create_with_discretization(capsule, N_shooting_intervals, new_time_steps);
}


int centroidal_mpc_hpipm_acados_update_time_steps(centroidal_mpc_hpipm_solver_capsule* capsule, int N, double* new_time_steps)
{

    if (N != capsule->nlp_solver_plan->N) {
        fprintf(stderr, "centroidal_mpc_hpipm_acados_update_time_steps: given number of time steps (= %d) " \
            "differs from the currently allocated number of " \
            "time steps (= %d)!\n" \
            "Please recreate with new discretization and provide a new vector of time_stamps!\n",
            N, capsule->nlp_solver_plan->N);
        return 1;
    }

    ocp_nlp_config * nlp_config = capsule->nlp_config;
    ocp_nlp_dims * nlp_dims = capsule->nlp_dims;
    ocp_nlp_in * nlp_in = capsule->nlp_in;

    for (int i = 0; i < N; i++)
    {
        ocp_nlp_in_set(nlp_config, nlp_dims, nlp_in, i, "Ts", &new_time_steps[i]);
        ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, i, "scaling", &new_time_steps[i]);
    }
    return 0;

}

/**
 * Internal function for centroidal_mpc_hpipm_acados_create: step 1
 */
void centroidal_mpc_hpipm_acados_create_set_plan(ocp_nlp_plan_t* nlp_solver_plan, const int N)
{
    assert(N == nlp_solver_plan->N);

    /************************************************
    *  plan
    ************************************************/

    nlp_solver_plan->nlp_solver = SQP_RTI;

    nlp_solver_plan->ocp_qp_solver_plan.qp_solver = PARTIAL_CONDENSING_HPIPM;
    nlp_solver_plan->relaxed_ocp_qp_solver_plan.qp_solver = PARTIAL_CONDENSING_HPIPM;
    nlp_solver_plan->nlp_cost[0] = LINEAR_LS;
    for (int i = 1; i < N; i++)
        nlp_solver_plan->nlp_cost[i] = LINEAR_LS;

    nlp_solver_plan->nlp_cost[N] = LINEAR_LS;

    for (int i = 0; i < N; i++)
    {
        nlp_solver_plan->nlp_dynamics[i] = DISCRETE_MODEL;
        // discrete dynamics does not need sim solver option, this field is ignored
        nlp_solver_plan->sim_solver_plan[i].sim_solver = INVALID_SIM_SOLVER;
    }

    nlp_solver_plan->nlp_constraints[0] = BGH;

    for (int i = 1; i < N; i++)
    {
        nlp_solver_plan->nlp_constraints[i] = BGH;
    }
    nlp_solver_plan->nlp_constraints[N] = BGH;

    nlp_solver_plan->regularization = NO_REGULARIZE;

    nlp_solver_plan->globalization = FIXED_STEP;
}


static ocp_nlp_dims* centroidal_mpc_hpipm_acados_create_setup_dimensions(centroidal_mpc_hpipm_solver_capsule* capsule)
{
    ocp_nlp_plan_t* nlp_solver_plan = capsule->nlp_solver_plan;
    const int N = nlp_solver_plan->N;
    ocp_nlp_config* nlp_config = capsule->nlp_config;

    /************************************************
    *  dimensions
    ************************************************/
    #define NINTNP1MEMS 18
    int* intNp1mem = (int*)malloc( (N+1)*sizeof(int)*NINTNP1MEMS );

    int* nx    = intNp1mem + (N+1)*0;
    int* nu    = intNp1mem + (N+1)*1;
    int* nbx   = intNp1mem + (N+1)*2;
    int* nbu   = intNp1mem + (N+1)*3;
    int* nsbx  = intNp1mem + (N+1)*4;
    int* nsbu  = intNp1mem + (N+1)*5;
    int* nsg   = intNp1mem + (N+1)*6;
    int* nsh   = intNp1mem + (N+1)*7;
    int* nsphi = intNp1mem + (N+1)*8;
    int* ns    = intNp1mem + (N+1)*9;
    int* ng    = intNp1mem + (N+1)*10;
    int* nh    = intNp1mem + (N+1)*11;
    int* nphi  = intNp1mem + (N+1)*12;
    int* nz    = intNp1mem + (N+1)*13;
    int* ny    = intNp1mem + (N+1)*14;
    int* nr    = intNp1mem + (N+1)*15;
    int* nbxe  = intNp1mem + (N+1)*16;
    int* np  = intNp1mem + (N+1)*17;

    for (int i = 0; i < N+1; i++)
    {
        // common
        nx[i]     = NX;
        nu[i]     = NU;
        nz[i]     = NZ;
        ns[i]     = NS;
        // cost
        ny[i]     = NY;
        // constraints
        nbx[i]    = NBX;
        nbu[i]    = NBU;
        nsbx[i]   = NSBX;
        nsbu[i]   = NSBU;
        nsg[i]    = NSG;
        nsh[i]    = NSH;
        nsphi[i]  = NSPHI;
        ng[i]     = NG;
        nh[i]     = NH;
        nphi[i]   = NPHI;
        nr[i]     = NR;
        nbxe[i]   = 0;
        np[i]     = NP;
    }

    // for initial state
    nbx[0] = NBX0;
    nsbx[0] = 0;
    ns[0] = NS0;
    
    nbxe[0] = 0;
    
    ny[0] = NY0;
    nh[0] = NH0;
    nsh[0] = NSH0;
    nsphi[0] = NSPHI0;
    nphi[0] = NPHI0;


    // terminal - common
    nu[N]   = 0;
    nz[N]   = 0;
    ns[N]   = NSN;
    // cost
    ny[N]   = NYN;
    // constraint
    nbx[N]   = NBXN;
    nbu[N]   = 0;
    ng[N]    = NGN;
    nh[N]    = NHN;
    nphi[N]  = NPHIN;
    nr[N]    = 0;

    nsbx[N]  = NSBXN;
    nsbu[N]  = 0;
    nsg[N]   = NSGN;
    nsh[N]   = NSHN;
    nsphi[N] = NSPHIN;

    /* create and set ocp_nlp_dims */
    ocp_nlp_dims * nlp_dims = ocp_nlp_dims_create(nlp_config);

    ocp_nlp_dims_set_opt_vars(nlp_config, nlp_dims, "nx", nx);
    ocp_nlp_dims_set_opt_vars(nlp_config, nlp_dims, "nu", nu);
    ocp_nlp_dims_set_opt_vars(nlp_config, nlp_dims, "nz", nz);
    ocp_nlp_dims_set_opt_vars(nlp_config, nlp_dims, "ns", ns);
    ocp_nlp_dims_set_opt_vars(nlp_config, nlp_dims, "np", np);

    ocp_nlp_dims_set_global(nlp_config, nlp_dims, "np_global", 0);
    ocp_nlp_dims_set_global(nlp_config, nlp_dims, "n_global_data", 0);

    for (int i = 0; i <= N; i++)
    {
        ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, i, "nbx", &nbx[i]);
        ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, i, "nbu", &nbu[i]);
        ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, i, "nsbx", &nsbx[i]);
        ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, i, "nsbu", &nsbu[i]);
        ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, i, "ng", &ng[i]);
        ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, i, "nsg", &nsg[i]);
        ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, i, "nbxe", &nbxe[i]);
    }
    ocp_nlp_dims_set_cost(nlp_config, nlp_dims, 0, "ny", &ny[0]);
    for (int i = 1; i < N; i++)
        ocp_nlp_dims_set_cost(nlp_config, nlp_dims, i, "ny", &ny[i]);
    ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, 0, "nh", &nh[0]);
    ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, 0, "nsh", &nsh[0]);

    for (int i = 1; i < N; i++)
    {
        ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, i, "nh", &nh[i]);
        ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, i, "nsh", &nsh[i]);
    }
    ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, N, "nh", &nh[N]);
    ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, N, "nsh", &nsh[N]);
    ocp_nlp_dims_set_cost(nlp_config, nlp_dims, N, "ny", &ny[N]);
    free(intNp1mem);

    return nlp_dims;
}


/**
 * Internal function for centroidal_mpc_hpipm_acados_create: step 3
 */
void centroidal_mpc_hpipm_acados_create_setup_functions(centroidal_mpc_hpipm_solver_capsule* capsule)
{
    const int N = capsule->nlp_solver_plan->N;

    /************************************************
    *  external functions
    ************************************************/

#define MAP_CASADI_FNC(__CAPSULE_FNC__, __MODEL_BASE_FNC__) do{ \
        capsule->__CAPSULE_FNC__.casadi_fun = & __MODEL_BASE_FNC__ ;\
        capsule->__CAPSULE_FNC__.casadi_n_in = & __MODEL_BASE_FNC__ ## _n_in; \
        capsule->__CAPSULE_FNC__.casadi_n_out = & __MODEL_BASE_FNC__ ## _n_out; \
        capsule->__CAPSULE_FNC__.casadi_sparsity_in = & __MODEL_BASE_FNC__ ## _sparsity_in; \
        capsule->__CAPSULE_FNC__.casadi_sparsity_out = & __MODEL_BASE_FNC__ ## _sparsity_out; \
        capsule->__CAPSULE_FNC__.casadi_work = & __MODEL_BASE_FNC__ ## _work; \
        external_function_external_param_casadi_create(&capsule->__CAPSULE_FNC__, &ext_fun_opts); \
    } while(false)

    external_function_opts ext_fun_opts;
    external_function_opts_set_to_default(&ext_fun_opts);


    ext_fun_opts.external_workspace = true;
    if (N > 0)
    {



    
        // discrete dynamics
        capsule->discr_dyn_phi_fun = (external_function_external_param_casadi *) malloc(sizeof(external_function_external_param_casadi)*N);
        for (int i = 0; i < N; i++)
        {
            MAP_CASADI_FNC(discr_dyn_phi_fun[i], centroidal_mpc_hpipm_dyn_disc_phi_fun);
        }

        capsule->discr_dyn_phi_fun_jac_ut_xt = (external_function_external_param_casadi *) malloc(sizeof(external_function_external_param_casadi)*N);
        for (int i = 0; i < N; i++)
        {
            MAP_CASADI_FNC(discr_dyn_phi_fun_jac_ut_xt[i], centroidal_mpc_hpipm_dyn_disc_phi_fun_jac);
        }

    

    
    } // N > 0

#undef MAP_CASADI_FNC
}


/**
 * Internal function for centroidal_mpc_hpipm_acados_create: step 5
 */
void centroidal_mpc_hpipm_acados_create_set_default_parameters(centroidal_mpc_hpipm_solver_capsule* capsule)
{

    const int N = capsule->nlp_solver_plan->N;
    // initialize parameters to nominal value
    double* p = calloc(NP, sizeof(double));

    for (int i = 0; i <= N; i++) {
        centroidal_mpc_hpipm_acados_update_params(capsule, i, p, NP);
    }
    free(p);


    // no global parameters defined
}


/**
 * Internal function for centroidal_mpc_hpipm_acados_create: step 5
 */
void centroidal_mpc_hpipm_acados_setup_nlp_in(centroidal_mpc_hpipm_solver_capsule* capsule, const int N, double* new_time_steps)
{
    assert(N == capsule->nlp_solver_plan->N);
    ocp_nlp_config* nlp_config = capsule->nlp_config;
    ocp_nlp_dims* nlp_dims = capsule->nlp_dims;

    int tmp_int = 0;

    /************************************************
    *  nlp_in
    ************************************************/
    ocp_nlp_in * nlp_in = capsule->nlp_in;
    /************************************************
    *  nlp_out
    ************************************************/
    ocp_nlp_out * nlp_out = capsule->nlp_out;

    // set up time_steps and cost_scaling

    if (new_time_steps)
    {
        // NOTE: this sets scaling and time_steps
        centroidal_mpc_hpipm_acados_update_time_steps(capsule, N, new_time_steps);
    }
    else
    {
        // set time_steps
    
        double time_step = 0.02;
        for (int i = 0; i < N; i++)
        {
            ocp_nlp_in_set(nlp_config, nlp_dims, nlp_in, i, "Ts", &time_step);
        }
        // set cost scaling
        double* cost_scaling = malloc((N+1)*sizeof(double));
        cost_scaling[0] = 0.02;
        cost_scaling[1] = 0.02;
        cost_scaling[2] = 0.02;
        cost_scaling[3] = 0.02;
        cost_scaling[4] = 0.02;
        cost_scaling[5] = 0.02;
        cost_scaling[6] = 0.02;
        cost_scaling[7] = 0.02;
        cost_scaling[8] = 0.02;
        cost_scaling[9] = 0.02;
        cost_scaling[10] = 0.02;
        cost_scaling[11] = 0.02;
        cost_scaling[12] = 0.02;
        cost_scaling[13] = 0.02;
        cost_scaling[14] = 0.02;
        cost_scaling[15] = 0.02;
        cost_scaling[16] = 1;
        for (int i = 0; i <= N; i++)
        {
            ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, i, "scaling", &cost_scaling[i]);
        }
        free(cost_scaling);
    }



    /**** Dynamics ****/
    for (int i = 0; i < N; i++)
    {
        ocp_nlp_dynamics_model_set_external_param_fun(nlp_config, nlp_dims, nlp_in, i, "disc_dyn_fun", &capsule->discr_dyn_phi_fun[i]);
        ocp_nlp_dynamics_model_set_external_param_fun(nlp_config, nlp_dims, nlp_in, i, "disc_dyn_fun_jac",
                                   &capsule->discr_dyn_phi_fun_jac_ut_xt[i]);
        
        
    }

    /**** Cost ****/
    double* yref_0 = calloc(NY0, sizeof(double));
    // change only the non-zero elements:
    ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, 0, "yref", yref_0);
    free(yref_0);

   double* W_0 = calloc(NY0*NY0, sizeof(double));
    // change only the non-zero elements:
    W_0[0+(NY0) * 0] = 1;
    W_0[1+(NY0) * 1] = 1;
    W_0[2+(NY0) * 2] = 50;
    W_0[3+(NY0) * 3] = 10;
    W_0[4+(NY0) * 4] = 20;
    W_0[5+(NY0) * 5] = 1;
    W_0[6+(NY0) * 6] = 2;
    W_0[7+(NY0) * 7] = 2;
    W_0[8+(NY0) * 8] = 1;
    W_0[9+(NY0) * 9] = 1;
    W_0[10+(NY0) * 10] = 1;
    W_0[11+(NY0) * 11] = 1;
    W_0[12+(NY0) * 12] = 0.00001;
    W_0[13+(NY0) * 13] = 0.00001;
    W_0[14+(NY0) * 14] = 0.00001;
    W_0[15+(NY0) * 15] = 0.00001;
    W_0[16+(NY0) * 16] = 0.00001;
    W_0[17+(NY0) * 17] = 0.00001;
    W_0[18+(NY0) * 18] = 0.00001;
    W_0[19+(NY0) * 19] = 0.00001;
    W_0[20+(NY0) * 20] = 0.00001;
    W_0[21+(NY0) * 21] = 0.00001;
    W_0[22+(NY0) * 22] = 0.00001;
    W_0[23+(NY0) * 23] = 0.00001;
    ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, 0, "W", W_0);
    free(W_0);
    double* Vx_0 = calloc(NY0*NX, sizeof(double));
    // change only the non-zero elements:
    Vx_0[0+(NY0) * 0] = 1;
    Vx_0[1+(NY0) * 1] = 1;
    Vx_0[2+(NY0) * 2] = 1;
    Vx_0[3+(NY0) * 3] = 1;
    Vx_0[4+(NY0) * 4] = 1;
    Vx_0[5+(NY0) * 5] = 1;
    Vx_0[6+(NY0) * 6] = 1;
    Vx_0[7+(NY0) * 7] = 1;
    Vx_0[8+(NY0) * 8] = 1;
    Vx_0[9+(NY0) * 9] = 1;
    Vx_0[10+(NY0) * 10] = 1;
    Vx_0[11+(NY0) * 11] = 1;
    ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, 0, "Vx", Vx_0);
    free(Vx_0);
    double* Vu_0 = calloc(NY0*NU, sizeof(double));
    // change only the non-zero elements:
    Vu_0[12+(NY0) * 0] = 1;
    Vu_0[13+(NY0) * 1] = 1;
    Vu_0[14+(NY0) * 2] = 1;
    Vu_0[15+(NY0) * 3] = 1;
    Vu_0[16+(NY0) * 4] = 1;
    Vu_0[17+(NY0) * 5] = 1;
    Vu_0[18+(NY0) * 6] = 1;
    Vu_0[19+(NY0) * 7] = 1;
    Vu_0[20+(NY0) * 8] = 1;
    Vu_0[21+(NY0) * 9] = 1;
    Vu_0[22+(NY0) * 10] = 1;
    Vu_0[23+(NY0) * 11] = 1;
    ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, 0, "Vu", Vu_0);
    free(Vu_0);
    double* yref = calloc(NY, sizeof(double));
    // change only the non-zero elements:

    for (int i = 1; i < N; i++)
    {
        ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, i, "yref", yref);
    }
    free(yref);
    double* W = calloc(NY*NY, sizeof(double));
    // change only the non-zero elements:
    W[0+(NY) * 0] = 1;
    W[1+(NY) * 1] = 1;
    W[2+(NY) * 2] = 50;
    W[3+(NY) * 3] = 10;
    W[4+(NY) * 4] = 20;
    W[5+(NY) * 5] = 1;
    W[6+(NY) * 6] = 2;
    W[7+(NY) * 7] = 2;
    W[8+(NY) * 8] = 1;
    W[9+(NY) * 9] = 1;
    W[10+(NY) * 10] = 1;
    W[11+(NY) * 11] = 1;
    W[12+(NY) * 12] = 0.00001;
    W[13+(NY) * 13] = 0.00001;
    W[14+(NY) * 14] = 0.00001;
    W[15+(NY) * 15] = 0.00001;
    W[16+(NY) * 16] = 0.00001;
    W[17+(NY) * 17] = 0.00001;
    W[18+(NY) * 18] = 0.00001;
    W[19+(NY) * 19] = 0.00001;
    W[20+(NY) * 20] = 0.00001;
    W[21+(NY) * 21] = 0.00001;
    W[22+(NY) * 22] = 0.00001;
    W[23+(NY) * 23] = 0.00001;

    for (int i = 1; i < N; i++)
    {
        ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, i, "W", W);
    }
    free(W);
    double* Vx = calloc(NY*NX, sizeof(double));
    // change only the non-zero elements:
    Vx[0+(NY) * 0] = 1;
    Vx[1+(NY) * 1] = 1;
    Vx[2+(NY) * 2] = 1;
    Vx[3+(NY) * 3] = 1;
    Vx[4+(NY) * 4] = 1;
    Vx[5+(NY) * 5] = 1;
    Vx[6+(NY) * 6] = 1;
    Vx[7+(NY) * 7] = 1;
    Vx[8+(NY) * 8] = 1;
    Vx[9+(NY) * 9] = 1;
    Vx[10+(NY) * 10] = 1;
    Vx[11+(NY) * 11] = 1;
    for (int i = 1; i < N; i++)
    {
        ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, i, "Vx", Vx);
    }
    free(Vx);

    
    double* Vu = calloc(NY*NU, sizeof(double));
    // change only the non-zero elements:
    Vu[12+(NY) * 0] = 1;
    Vu[13+(NY) * 1] = 1;
    Vu[14+(NY) * 2] = 1;
    Vu[15+(NY) * 3] = 1;
    Vu[16+(NY) * 4] = 1;
    Vu[17+(NY) * 5] = 1;
    Vu[18+(NY) * 6] = 1;
    Vu[19+(NY) * 7] = 1;
    Vu[20+(NY) * 8] = 1;
    Vu[21+(NY) * 9] = 1;
    Vu[22+(NY) * 10] = 1;
    Vu[23+(NY) * 11] = 1;

    for (int i = 1; i < N; i++)
    {
        ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, i, "Vu", Vu);
    }
    free(Vu);
    double* yref_e = calloc(NYN, sizeof(double));
    // change only the non-zero elements:
    ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, N, "yref", yref_e);
    free(yref_e);

    double* W_e = calloc(NYN*NYN, sizeof(double));
    // change only the non-zero elements:
    W_e[0+(NYN) * 0] = 1;
    W_e[1+(NYN) * 1] = 1;
    W_e[2+(NYN) * 2] = 50;
    W_e[3+(NYN) * 3] = 10;
    W_e[4+(NYN) * 4] = 20;
    W_e[5+(NYN) * 5] = 1;
    W_e[6+(NYN) * 6] = 2;
    W_e[7+(NYN) * 7] = 2;
    W_e[8+(NYN) * 8] = 1;
    W_e[9+(NYN) * 9] = 1;
    W_e[10+(NYN) * 10] = 1;
    W_e[11+(NYN) * 11] = 1;
    ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, N, "W", W_e);
    free(W_e);
    double* Vx_e = calloc(NYN*NX, sizeof(double));
    // change only the non-zero elements:
    Vx_e[0+(NYN) * 0] = 1;
    Vx_e[1+(NYN) * 1] = 1;
    Vx_e[2+(NYN) * 2] = 1;
    Vx_e[3+(NYN) * 3] = 1;
    Vx_e[4+(NYN) * 4] = 1;
    Vx_e[5+(NYN) * 5] = 1;
    Vx_e[6+(NYN) * 6] = 1;
    Vx_e[7+(NYN) * 7] = 1;
    Vx_e[8+(NYN) * 8] = 1;
    Vx_e[9+(NYN) * 9] = 1;
    Vx_e[10+(NYN) * 10] = 1;
    Vx_e[11+(NYN) * 11] = 1;
    ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, N, "Vx", Vx_e);
    free(Vx_e);







    /**** Constraints ****/

    // bounds for initial stage
    // x0
    int* idxbx0 = malloc(NBX0 * sizeof(int));
    idxbx0[0] = 0;
    idxbx0[1] = 1;
    idxbx0[2] = 2;
    idxbx0[3] = 3;
    idxbx0[4] = 4;
    idxbx0[5] = 5;
    idxbx0[6] = 6;
    idxbx0[7] = 7;
    idxbx0[8] = 8;
    idxbx0[9] = 9;
    idxbx0[10] = 10;
    idxbx0[11] = 11;

    double* lubx0 = calloc(2*NBX0, sizeof(double));
    double* lbx0 = lubx0;
    double* ubx0 = lubx0 + NBX0;
    // change only the non-zero elements:

    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, nlp_out, 0, "idxbx", idxbx0);
    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, nlp_out, 0, "lbx", lbx0);
    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, nlp_out, 0, "ubx", ubx0);
    free(idxbx0);
    free(lubx0);












    /* constraints that are the same for initial and intermediate */
    // u
    int* idxbu = malloc(NBU * sizeof(int));
    idxbu[0] = 0;
    idxbu[1] = 1;
    idxbu[2] = 2;
    idxbu[3] = 3;
    idxbu[4] = 4;
    idxbu[5] = 5;
    idxbu[6] = 6;
    idxbu[7] = 7;
    idxbu[8] = 8;
    idxbu[9] = 9;
    idxbu[10] = 10;
    idxbu[11] = 11;
    double* lubu = calloc(2*NBU, sizeof(double));
    double* lbu = lubu;
    double* ubu = lubu + NBU;
    lbu[0] = -100000000;
    ubu[0] = 100000000;
    lbu[1] = -100000000;
    ubu[1] = 100000000;
    lbu[2] = -100000000;
    ubu[2] = 100000000;
    lbu[3] = -100000000;
    ubu[3] = 100000000;
    lbu[4] = -100000000;
    ubu[4] = 100000000;
    lbu[5] = -100000000;
    ubu[5] = 100000000;
    lbu[6] = -100000000;
    ubu[6] = 100000000;
    lbu[7] = -100000000;
    ubu[7] = 100000000;
    lbu[8] = -100000000;
    ubu[8] = 100000000;
    lbu[9] = -100000000;
    ubu[9] = 100000000;
    lbu[10] = -100000000;
    ubu[10] = 100000000;
    lbu[11] = -100000000;
    ubu[11] = 100000000;

    for (int i = 0; i < N; i++)
    {
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, nlp_out, i, "idxbu", idxbu);
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, nlp_out, i, "lbu", lbu);
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, nlp_out, i, "ubu", ubu);
    }
    free(idxbu);
    free(lubu);


    // set up general constraints for stage 0 to N-1
    double* D = calloc(NG*NU, sizeof(double));
    double* C = calloc(NG*NX, sizeof(double));
    double* lug = calloc(2*NG, sizeof(double));
    double* lg = lug;
    double* ug = lug + NG;
    D[0+NG * 0] = 1;
    D[0+NG * 2] = -0.8;
    D[1+NG * 0] = -1;
    D[1+NG * 2] = -0.8;
    D[2+NG * 1] = 1;
    D[2+NG * 2] = -0.8;
    D[3+NG * 1] = -1;
    D[3+NG * 2] = -0.8;
    D[4+NG * 3] = 1;
    D[4+NG * 5] = -0.8;
    D[5+NG * 3] = -1;
    D[5+NG * 5] = -0.8;
    D[6+NG * 4] = 1;
    D[6+NG * 5] = -0.8;
    D[7+NG * 4] = -1;
    D[7+NG * 5] = -0.8;
    D[8+NG * 6] = 1;
    D[8+NG * 8] = -0.8;
    D[9+NG * 6] = -1;
    D[9+NG * 8] = -0.8;
    D[10+NG * 7] = 1;
    D[10+NG * 8] = -0.8;
    D[11+NG * 7] = -1;
    D[11+NG * 8] = -0.8;
    D[12+NG * 9] = 1;
    D[12+NG * 11] = -0.8;
    D[13+NG * 9] = -1;
    D[13+NG * 11] = -0.8;
    D[14+NG * 10] = 1;
    D[14+NG * 11] = -0.8;
    D[15+NG * 10] = -1;
    D[15+NG * 11] = -0.8;
    lg[0] = -100000000;
    lg[1] = -100000000;
    lg[2] = -100000000;
    lg[3] = -100000000;
    lg[4] = -100000000;
    lg[5] = -100000000;
    lg[6] = -100000000;
    lg[7] = -100000000;
    lg[8] = -100000000;
    lg[9] = -100000000;
    lg[10] = -100000000;
    lg[11] = -100000000;
    lg[12] = -100000000;
    lg[13] = -100000000;
    lg[14] = -100000000;
    lg[15] = -100000000;

    for (int i = 0; i < N; i++)
    {
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, nlp_out, i, "D", D);
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, nlp_out, i, "C", C);
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, nlp_out, i, "lg", lg);
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, nlp_out, i, "ug", ug);
    }
    free(D);
    free(C);
    free(lug);




    /* Path constraints */














    /* terminal constraints */




















}


static void centroidal_mpc_hpipm_acados_create_set_opts(centroidal_mpc_hpipm_solver_capsule* capsule)
{
    const int N = capsule->nlp_solver_plan->N;
    ocp_nlp_config* nlp_config = capsule->nlp_config;
    void *nlp_opts = capsule->nlp_opts;

    /************************************************
    *  opts
    ************************************************/



    int fixed_hess = 0;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "fixed_hess", &fixed_hess);

    double globalization_fixed_step_length = 1;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "globalization_fixed_step_length", &globalization_fixed_step_length);




    int with_solution_sens_wrt_params = false;
    ocp_nlp_solver_opts_set(nlp_config, capsule->nlp_opts, "with_solution_sens_wrt_params", &with_solution_sens_wrt_params);

    int with_value_sens_wrt_params = false;
    ocp_nlp_solver_opts_set(nlp_config, capsule->nlp_opts, "with_value_sens_wrt_params", &with_value_sens_wrt_params);

    double solution_sens_qp_t_lam_min = 0.000000001;
    ocp_nlp_solver_opts_set(nlp_config, capsule->nlp_opts, "solution_sens_qp_t_lam_min", &solution_sens_qp_t_lam_min);

    int globalization_full_step_dual = 0;
    ocp_nlp_solver_opts_set(nlp_config, capsule->nlp_opts, "globalization_full_step_dual", &globalization_full_step_dual);

    double levenberg_marquardt = 0;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "levenberg_marquardt", &levenberg_marquardt);

    /* options QP solver */
    int qp_solver_cond_N;const int qp_solver_cond_N_ori = 10;
    qp_solver_cond_N = N < qp_solver_cond_N_ori ? N : qp_solver_cond_N_ori; // use the minimum value here
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "qp_cond_N", &qp_solver_cond_N);

    int nlp_solver_ext_qp_res = 0;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "ext_qp_res", &nlp_solver_ext_qp_res);

    bool store_iterates = false;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "store_iterates", &store_iterates);
    // set HPIPM mode: should be done before setting other QP solver options
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "qp_hpipm_mode", "BALANCE");



    int qp_solver_t0_init = 2;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "qp_t0_init", &qp_solver_t0_init);




    int as_rti_iter = 1;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "as_rti_iter", &as_rti_iter);

    int as_rti_level = 4;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "as_rti_level", &as_rti_level);

    int rti_log_residuals = 0;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "rti_log_residuals", &rti_log_residuals);

    int rti_log_only_available_residuals = 0;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "rti_log_only_available_residuals", &rti_log_only_available_residuals);

    bool with_anderson_acceleration = false;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "with_anderson_acceleration", &with_anderson_acceleration);

    double anderson_activation_threshold = 10;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "anderson_activation_threshold", &anderson_activation_threshold);

    int qp_solver_iter_max = 50;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "qp_iter_max", &qp_solver_iter_max);


    int qp_solver_warm_start = 1;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "qp_warm_start", &qp_solver_warm_start);

    int print_level = 0;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "print_level", &print_level);
    int qp_solver_cond_ric_alg = 1;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "qp_cond_ric_alg", &qp_solver_cond_ric_alg);

    int qp_solver_ric_alg = 1;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "qp_ric_alg", &qp_solver_ric_alg);


    int ext_cost_num_hess = 0;
}


/**
 * Internal function for centroidal_mpc_hpipm_acados_create: step 7
 */
void centroidal_mpc_hpipm_acados_set_nlp_out(centroidal_mpc_hpipm_solver_capsule* capsule)
{
    const int N = capsule->nlp_solver_plan->N;
    ocp_nlp_config* nlp_config = capsule->nlp_config;
    ocp_nlp_dims* nlp_dims = capsule->nlp_dims;
    ocp_nlp_out* nlp_out = capsule->nlp_out;
    ocp_nlp_in* nlp_in = capsule->nlp_in;

    // initialize primal solution
    double* xu0 = calloc(NX+NU, sizeof(double));
    double* x0 = xu0;

    // initialize with x0


    double* u0 = xu0 + NX;

    for (int i = 0; i < N; i++)
    {
        // x0
        ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, nlp_in, i, "x", x0);
        // u0
        ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, nlp_in, i, "u", u0);
    }
    ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, nlp_in, N, "x", x0);
    free(xu0);
}


/**
 * Internal function for centroidal_mpc_hpipm_acados_create: step 9
 */
int centroidal_mpc_hpipm_acados_create_precompute(centroidal_mpc_hpipm_solver_capsule* capsule) {
    int status = ocp_nlp_precompute(capsule->nlp_solver, capsule->nlp_in, capsule->nlp_out);

    if (status != ACADOS_SUCCESS) {
        printf("\nocp_nlp_precompute failed!\n\n");
        exit(1);
    }

    return status;
}


int centroidal_mpc_hpipm_acados_create_with_discretization(centroidal_mpc_hpipm_solver_capsule* capsule, int N, double* new_time_steps)
{
    // If N does not match the number of shooting intervals used for code generation, new_time_steps must be given.
    if (N != CENTROIDAL_MPC_HPIPM_N && !new_time_steps) {
        fprintf(stderr, "centroidal_mpc_hpipm_acados_create_with_discretization: new_time_steps is NULL " \
            "but the number of shooting intervals (= %d) differs from the number of " \
            "shooting intervals (= %d) during code generation! Please provide a new vector of time_stamps!\n", \
             N, CENTROIDAL_MPC_HPIPM_N);
        return 1;
    }

    // number of expected runtime parameters
    capsule->nlp_np = NP;

    // 1) create and set nlp_solver_plan; create nlp_config
    capsule->nlp_solver_plan = ocp_nlp_plan_create(N);
    centroidal_mpc_hpipm_acados_create_set_plan(capsule->nlp_solver_plan, N);
    capsule->nlp_config = ocp_nlp_config_create(*capsule->nlp_solver_plan);

    // 2) create and set dimensions
    capsule->nlp_dims = centroidal_mpc_hpipm_acados_create_setup_dimensions(capsule);

    // 3) create and set nlp_opts
    capsule->nlp_opts = ocp_nlp_solver_opts_create(capsule->nlp_config, capsule->nlp_dims);
    centroidal_mpc_hpipm_acados_create_set_opts(capsule);

    // 4) create and set nlp_out
    // 4.1) nlp_out
    capsule->nlp_out = ocp_nlp_out_create(capsule->nlp_config, capsule->nlp_dims);
    // 4.2) sens_out
    capsule->sens_out = ocp_nlp_out_create(capsule->nlp_config, capsule->nlp_dims);
    centroidal_mpc_hpipm_acados_set_nlp_out(capsule);

    // 5) create nlp_in
    capsule->nlp_in = ocp_nlp_in_create(capsule->nlp_config, capsule->nlp_dims);

    // 6) setup functions, nlp_in and default parameters
    centroidal_mpc_hpipm_acados_create_setup_functions(capsule);
    centroidal_mpc_hpipm_acados_setup_nlp_in(capsule, N, new_time_steps);
    centroidal_mpc_hpipm_acados_create_set_default_parameters(capsule);

    // 7) create solver
    capsule->nlp_solver = ocp_nlp_solver_create(capsule->nlp_config, capsule->nlp_dims, capsule->nlp_opts, capsule->nlp_in);


    // 8) do precomputations
    int status = centroidal_mpc_hpipm_acados_create_precompute(capsule);

    return status;
}

/**
 * This function is for updating an already initialized solver with a different number of qp_cond_N. It is useful for code reuse after code export.
 */
int centroidal_mpc_hpipm_acados_update_qp_solver_cond_N(centroidal_mpc_hpipm_solver_capsule* capsule, int qp_solver_cond_N)
{
    // 1) destroy solver
    ocp_nlp_solver_destroy(capsule->nlp_solver);

    // 2) set new value for "qp_cond_N"
    const int N = capsule->nlp_solver_plan->N;
    if(qp_solver_cond_N > N)
        printf("Warning: qp_solver_cond_N = %d > N = %d\n", qp_solver_cond_N, N);
    ocp_nlp_solver_opts_set(capsule->nlp_config, capsule->nlp_opts, "qp_cond_N", &qp_solver_cond_N);

    // 3) continue with the remaining steps from centroidal_mpc_hpipm_acados_create_with_discretization(...):
    // -> 8) create solver
    capsule->nlp_solver = ocp_nlp_solver_create(capsule->nlp_config, capsule->nlp_dims, capsule->nlp_opts, capsule->nlp_in);

    // -> 9) do precomputations
    int status = centroidal_mpc_hpipm_acados_create_precompute(capsule);
    return status;
}


int centroidal_mpc_hpipm_acados_reset(centroidal_mpc_hpipm_solver_capsule* capsule, int reset_qp_solver_mem)
{

    // set initialization to all zeros

    const int N = capsule->nlp_solver_plan->N;
    ocp_nlp_config* nlp_config = capsule->nlp_config;
    ocp_nlp_dims* nlp_dims = capsule->nlp_dims;
    ocp_nlp_out* nlp_out = capsule->nlp_out;
    ocp_nlp_in* nlp_in = capsule->nlp_in;
    ocp_nlp_solver* nlp_solver = capsule->nlp_solver;

    double* buffer = calloc(NX+NU+NZ+2*NS+2*NSN+2*NS0+NBX+NBU+NG+NH+NPHI+NBX0+NBXN+NHN+NH0+NPHIN+NGN, sizeof(double));

    for(int i=0; i<N+1; i++)
    {
        ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, nlp_in, i, "x", buffer);
        ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, nlp_in, i, "u", buffer);
        ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, nlp_in, i, "sl", buffer);
        ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, nlp_in, i, "su", buffer);
        ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, nlp_in, i, "lam", buffer);
        ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, nlp_in, i, "z", buffer);
        if (i<N)
        {
            ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, nlp_in, i, "pi", buffer);
        }
    }
    // get qp_status: if NaN -> reset memory
    int qp_status;
    ocp_nlp_get(capsule->nlp_solver, "qp_status", &qp_status);
    if (reset_qp_solver_mem || (qp_status == 3))
    {
        // printf("\nin reset qp_status %d -> resetting QP memory\n", qp_status);
        ocp_nlp_solver_reset_qp_memory(nlp_solver, nlp_in, nlp_out);
    }

    free(buffer);
    return 0;
}




int centroidal_mpc_hpipm_acados_update_params(centroidal_mpc_hpipm_solver_capsule* capsule, int stage, double *p, int np)
{
    int solver_status = 0;

    int casadi_np = 300;
    if (casadi_np != np) {
        printf("acados_update_params: trying to set %i parameters for external functions."
            " External function has %i parameters. Exiting.\n", np, casadi_np);
        exit(1);
    }
    ocp_nlp_in_set(capsule->nlp_config, capsule->nlp_dims, capsule->nlp_in, stage, "parameter_values", p);

    return solver_status;
}


int centroidal_mpc_hpipm_acados_update_params_sparse(centroidal_mpc_hpipm_solver_capsule * capsule, int stage, int *idx, double *p, int n_update)
{
    ocp_nlp_in_set_params_sparse(capsule->nlp_config, capsule->nlp_dims, capsule->nlp_in, stage, idx, p, n_update);

    return 0;
}


int centroidal_mpc_hpipm_acados_set_p_global_and_precompute_dependencies(centroidal_mpc_hpipm_solver_capsule* capsule, double* data, int data_len)
{

    // printf("No global_data, centroidal_mpc_hpipm_acados_set_p_global_and_precompute_dependencies does nothing.\n");
    return 0;
}




int centroidal_mpc_hpipm_acados_solve(centroidal_mpc_hpipm_solver_capsule* capsule)
{
    // solve NLP
    int solver_status = ocp_nlp_solve(capsule->nlp_solver, capsule->nlp_in, capsule->nlp_out);

    return solver_status;
}



int centroidal_mpc_hpipm_acados_setup_qp_matrices_and_factorize(centroidal_mpc_hpipm_solver_capsule* capsule)
{
    int solver_status = ocp_nlp_setup_qp_matrices_and_factorize(capsule->nlp_solver, capsule->nlp_in, capsule->nlp_out);

    return solver_status;
}






int centroidal_mpc_hpipm_acados_free(centroidal_mpc_hpipm_solver_capsule* capsule)
{
    // before destroying, keep some info
    const int N = capsule->nlp_solver_plan->N;
    // free memory
    ocp_nlp_solver_opts_destroy(capsule->nlp_opts);
    ocp_nlp_in_destroy(capsule->nlp_in);
    ocp_nlp_out_destroy(capsule->nlp_out);
    ocp_nlp_out_destroy(capsule->sens_out);
    ocp_nlp_solver_destroy(capsule->nlp_solver);
    ocp_nlp_dims_destroy(capsule->nlp_dims);
    ocp_nlp_config_destroy(capsule->nlp_config);
    ocp_nlp_plan_destroy(capsule->nlp_solver_plan);

    /* free external function */
    // dynamics
    for (int i = 0; i < N; i++)
    {
        external_function_external_param_casadi_free(&capsule->discr_dyn_phi_fun[i]);
        external_function_external_param_casadi_free(&capsule->discr_dyn_phi_fun_jac_ut_xt[i]);
        
        
    }
    free(capsule->discr_dyn_phi_fun);
    free(capsule->discr_dyn_phi_fun_jac_ut_xt);
  
  

    // cost

    // constraints



    return 0;
}


void centroidal_mpc_hpipm_acados_print_stats(centroidal_mpc_hpipm_solver_capsule* capsule)
{
    int nlp_iter, stat_m, stat_n, tmp_int;
    ocp_nlp_get(capsule->nlp_solver, "nlp_iter", &nlp_iter);
    ocp_nlp_get(capsule->nlp_solver, "stat_n", &stat_n);
    ocp_nlp_get(capsule->nlp_solver, "stat_m", &stat_m);


    int stat_n_max = 16;
    if (stat_n > stat_n_max)
    {
        printf("stat_n_max = %d is too small, increase it in the template!\n", stat_n_max);
        exit(1);
    }
    double stat[1616];
    ocp_nlp_get(capsule->nlp_solver, "statistics", stat);

    int nrow = nlp_iter+1 < stat_m ? nlp_iter+1 : stat_m;


    printf("iter\tqp_stat\tqp_iter\n");
    for (int i = 0; i < nrow; i++)
    {
        for (int j = 0; j < stat_n + 1; j++)
        {
            tmp_int = (int) stat[i + j * nrow];
            printf("%d\t", tmp_int);
        }
        printf("\n");
    }
}

int centroidal_mpc_hpipm_acados_custom_update(centroidal_mpc_hpipm_solver_capsule* capsule, double* data, int data_len)
{
    (void)capsule;
    (void)data;
    (void)data_len;
    printf("\ndummy function that can be called in between solver calls to update parameters or numerical data efficiently in C.\n");
    printf("nothing set yet..\n");
    return 1;

}



ocp_nlp_in *centroidal_mpc_hpipm_acados_get_nlp_in(centroidal_mpc_hpipm_solver_capsule* capsule) { return capsule->nlp_in; }
ocp_nlp_out *centroidal_mpc_hpipm_acados_get_nlp_out(centroidal_mpc_hpipm_solver_capsule* capsule) { return capsule->nlp_out; }
ocp_nlp_out *centroidal_mpc_hpipm_acados_get_sens_out(centroidal_mpc_hpipm_solver_capsule* capsule) { return capsule->sens_out; }
ocp_nlp_solver *centroidal_mpc_hpipm_acados_get_nlp_solver(centroidal_mpc_hpipm_solver_capsule* capsule) { return capsule->nlp_solver; }
ocp_nlp_config *centroidal_mpc_hpipm_acados_get_nlp_config(centroidal_mpc_hpipm_solver_capsule* capsule) { return capsule->nlp_config; }
void *centroidal_mpc_hpipm_acados_get_nlp_opts(centroidal_mpc_hpipm_solver_capsule* capsule) { return capsule->nlp_opts; }
ocp_nlp_dims *centroidal_mpc_hpipm_acados_get_nlp_dims(centroidal_mpc_hpipm_solver_capsule* capsule) { return capsule->nlp_dims; }
ocp_nlp_plan_t *centroidal_mpc_hpipm_acados_get_nlp_plan(centroidal_mpc_hpipm_solver_capsule* capsule) { return capsule->nlp_solver_plan; }
