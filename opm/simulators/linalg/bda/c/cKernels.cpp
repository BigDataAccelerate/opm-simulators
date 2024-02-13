/*
  Copyright 2024 Equinor ASA

  This file is part of the Open Porous Media project (OPM).

  OPM is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  OPM is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with OPM.  If not, see <http://www.gnu.org/licenses/>.
*/

#include <config.h>
#include <cmath>
#include <sstream>

#include <opm/common/OpmLog/OpmLog.hpp>
#include <opm/common/ErrorMacros.hpp>
#include <dune/common/timer.hh>

#include <opm/simulators/linalg/bda/c/cKernels.hpp>
// #include <opm/simulators/linalg/bda/opencl/ChowPatelIlu.hpp>  // defines CHOW_PATEL

#include <iostream>

namespace Opm
{
namespace Accelerator
{

using Opm::OpmLog;
using Dune::Timer;

// define static variables and kernels
// int cKernels::verbosity;
// std::vector<double> cKernels::tmp;
// bool cKernels::initialized = false;

// divide A by B, and round up: return (int)ceil(A/B)
unsigned int ceilDivision(const unsigned int A, const unsigned int B)
{
    return A / B + (A % B > 0);
}

// template <unsigned int block_size>
// cKernels<block_size>::cKernels(int verbosity_)
cKernels::cKernels(int verbosity_)
{
    verbosity = verbosity_;
    initialized = true;
} 

// double cKernels::dot(cl::Buffer& in1, cl::Buffer& in2, cl::Buffer& out, int N)
// {
//     const unsigned int work_group_size = 256;
//     const unsigned int num_work_groups = ceilDivision(N, work_group_size);
//     const unsigned int total_work_items = num_work_groups * work_group_size;
//     const unsigned int lmem_per_work_group = sizeof(double) * work_group_size;
//     Timer t_dot;
//     tmp.resize(num_work_groups);
// 
//     cl::Event event = (*dot_k)(cl::EnqueueArgs(*queue, cl::NDRange(total_work_items), cl::NDRange(work_group_size)), in1, in2, out, N, cl::Local(lmem_per_work_group));
// 
//     queue->enqueueReadBuffer(out, CL_TRUE, 0, sizeof(double) * num_work_groups, tmp.data());
// 
//     double gpu_sum = 0.0;
//     for (unsigned int i = 0; i < num_work_groups; ++i) {
//         gpu_sum += tmp[i];
//     }
// 
//     if (verbosity >= 4) {
//         event.wait();
//         std::ostringstream oss;
//         oss << std::scientific << "cKernels dot() time: " << t_dot.stop() << " s";
//         OpmLog::info(oss.str());
//     }
// 
//     return gpu_sum;
// }

double cKernels::norm(double* in, double* out, int N)
{
//     const unsigned int work_group_size = 256;
//     const unsigned int num_work_groups = ceilDivision(N, work_group_size);
//     const unsigned int total_work_items = num_work_groups * work_group_size;
//     const unsigned int lmem_per_work_group = sizeof(double) * work_group_size;
//     Timer t_norm;
//     tmp.resize(num_work_groups);
// 
//     cl::Event event = (*norm_k)(cl::EnqueueArgs(*queue, cl::NDRange(total_work_items), cl::NDRange(work_group_size)), in, out, N, cl::Local(lmem_per_work_group));
// 
//     queue->enqueueReadBuffer(out, CL_TRUE, 0, sizeof(double) * num_work_groups, tmp.data());
    std::cout << "N = " << N << std::endl;
    for(int i=0; i<100; i++) {
        std::cout << std::scientific << "     in[" << i << "] = " << in[i];
        std::cout << std::scientific << "\t\tout[" << i << "] = " << out[i] << std::endl;
    }
    double gpu_norm = 10.0;
    exit(0);
//     for (unsigned int i = 0; i < num_work_groups; ++i) {
//         gpu_norm += tmp[i];
//     }
//     gpu_norm = sqrt(gpu_norm);
// 
//     if (verbosity >= 4) {
//         event.wait();
//         std::ostringstream oss;
//         oss << std::scientific << "cKernels norm() time: " << t_norm.stop() << " s";
//         OpmLog::info(oss.str());
//     }

    return gpu_norm;
}

// void cKernels::axpy(cl::Buffer& in, const double a, cl::Buffer& out, int N)
// {
//     const unsigned int work_group_size = 32;
//     const unsigned int num_work_groups = ceilDivision(N, work_group_size);
//     const unsigned int total_work_items = num_work_groups * work_group_size;
//     Timer t_axpy;
// 
//     cl::Event event = (*axpy_k)(cl::EnqueueArgs(*queue, cl::NDRange(total_work_items), cl::NDRange(work_group_size)), in, a, out, N);
// 
//     if (verbosity >= 4) {
//         event.wait();
//         std::ostringstream oss;
//         oss << std::scientific << "cKernels axpy() time: " << t_axpy.stop() << " s";
//         OpmLog::info(oss.str());
//     }
// }
// 
// void cKernels::scale(cl::Buffer& in, const double a, int N)
// {
//     const unsigned int work_group_size = 32;
//     const unsigned int num_work_groups = ceilDivision(N, work_group_size);
//     const unsigned int total_work_items = num_work_groups * work_group_size;
//     Timer t_scale;
// 
//     cl::Event event = (*scale_k)(cl::EnqueueArgs(*queue, cl::NDRange(total_work_items), cl::NDRange(work_group_size)), in, a, N);
// 
//     if (verbosity >= 4) {
//         event.wait();
//         std::ostringstream oss;
//         oss << std::scientific << "cKernels scale() time: " << t_scale.stop() << " s";
//         OpmLog::info(oss.str());
//     }
// }
// 
// void cKernels::vmul(const double alpha, cl::Buffer& in1, cl::Buffer& in2, cl::Buffer& out, int N)
// {
//     const unsigned int work_group_size = 32;
//     const unsigned int num_work_groups = ceilDivision(N, work_group_size);
//     const unsigned int total_work_items = num_work_groups * work_group_size;
//     Timer t_vmul;
// 
//     cl::Event event = (*vmul_k)(cl::EnqueueArgs(*queue, cl::NDRange(total_work_items), cl::NDRange(work_group_size)), alpha, in1, in2, out, N);
// 
//     if (verbosity >= 4) {
//         event.wait();
//         std::ostringstream oss;
//         oss << std::scientific << "cKernels vmul() time: " << t_vmul.stop() << " s";
//         OpmLog::info(oss.str());
//     }
// }
// 
// void cKernels::custom(cl::Buffer& p, cl::Buffer& v, cl::Buffer& r,
//                            const double omega, const double beta, int N)
// {
//     const unsigned int work_group_size = 32;
//     const unsigned int num_work_groups = ceilDivision(N, work_group_size);
//     const unsigned int total_work_items = num_work_groups * work_group_size;
//     Timer t_custom;
// 
//     cl::Event event = (*custom_k)(cl::EnqueueArgs(*queue, cl::NDRange(total_work_items), cl::NDRange(work_group_size)), p, v, r, omega, beta, N);
// 
//     if (verbosity >= 4) {
//         event.wait();
//         std::ostringstream oss;
//         oss << std::scientific << "cKernels custom() time: " << t_custom.stop() << " s";
//         OpmLog::info(oss.str());
//     }
// }
// 
// void cKernels::full_to_pressure_restriction(const cl::Buffer& fine_y, cl::Buffer& weights, cl::Buffer& coarse_y, int Nb)
// {
//     const unsigned int work_group_size = 32;
//     const unsigned int num_work_groups = ceilDivision(Nb, work_group_size);
//     const unsigned int total_work_items = num_work_groups * work_group_size;
//     Timer t;
// 
//     cl::Event event = (*full_to_pressure_restriction_k)(cl::EnqueueArgs(*queue, cl::NDRange(total_work_items), cl::NDRange(work_group_size)), fine_y, weights, coarse_y, Nb);
// 
//     if (verbosity >= 4) {
//         event.wait();
//         std::ostringstream oss;
//         oss << std::scientific << "cKernels full_to_pressure_restriction() time: " << t.stop() << " s";
//         OpmLog::info(oss.str());
//     }
// }
// 
// void cKernels::add_coarse_pressure_correction(cl::Buffer& coarse_x, cl::Buffer& fine_x, int pressure_idx, int Nb)
// {
//     const unsigned int work_group_size = 32;
//     const unsigned int num_work_groups = ceilDivision(Nb, work_group_size);
//     const unsigned int total_work_items = num_work_groups * work_group_size;
//     Timer t;
// 
//     cl::Event event = (*add_coarse_pressure_correction_k)(cl::EnqueueArgs(*queue, cl::NDRange(total_work_items), cl::NDRange(work_group_size)), coarse_x, fine_x, pressure_idx, Nb);
// 
//     if (verbosity >= 4) {
//         event.wait();
//         std::ostringstream oss;
//         oss << std::scientific << "cKernels add_coarse_pressure_correction() time: " << t.stop() << " s";
//         OpmLog::info(oss.str());
//     }
// }
// 
// void cKernels::prolongate_vector(const cl::Buffer& in, cl::Buffer& out, const cl::Buffer& cols, int N)
// {
//     const unsigned int work_group_size = 32;
//     const unsigned int num_work_groups = ceilDivision(N, work_group_size);
//     const unsigned int total_work_items = num_work_groups * work_group_size;
//     Timer t;
// 
//     cl::Event event = (*prolongate_vector_k)(cl::EnqueueArgs(*queue, cl::NDRange(total_work_items), cl::NDRange(work_group_size)), in, out, cols, N);
// 
//     if (verbosity >= 4) {
//         event.wait();
//         std::ostringstream oss;
//         oss << std::scientific << "cKernels prolongate_vector() time: " << t.stop() << " s";
//         OpmLog::info(oss.str());
//     }
// }
// 
// void cKernels::spmv(cl::Buffer& vals, cl::Buffer& cols, cl::Buffer& rows,
//                          const cl::Buffer& x, cl::Buffer& b, int Nb,
//                          unsigned int block_size, bool reset, bool add)
// {
//     const unsigned int work_group_size = 32;
//     const unsigned int num_work_groups = ceilDivision(Nb, work_group_size);
//     const unsigned int total_work_items = num_work_groups * work_group_size;
//     const unsigned int lmem_per_work_group = sizeof(double) * work_group_size;
//     Timer t_spmv;
//     cl::Event event;
// 
//     if (block_size > 1) {
//         if (add) {
//             event = (*spmv_blocked_add_k)(cl::EnqueueArgs(*queue, cl::NDRange(total_work_items), cl::NDRange(work_group_size)),
//                         vals, cols, rows, Nb, x, b, block_size, cl::Local(lmem_per_work_group));
//         } else {
//             event = (*spmv_blocked_k)(cl::EnqueueArgs(*queue, cl::NDRange(total_work_items), cl::NDRange(work_group_size)),
//                         vals, cols, rows, Nb, x, b, block_size, cl::Local(lmem_per_work_group));
//         }
//     } else {
//         if (reset) {
//             event = (*spmv_k)(cl::EnqueueArgs(*queue, cl::NDRange(total_work_items), cl::NDRange(work_group_size)),
//                         vals, cols, rows, Nb, x, b, cl::Local(lmem_per_work_group));
//         } else {
//             event = (*spmv_noreset_k)(cl::EnqueueArgs(*queue, cl::NDRange(total_work_items), cl::NDRange(work_group_size)),
//                         vals, cols, rows, Nb, x, b, cl::Local(lmem_per_work_group));
//         }
//     }
// 
//     if (verbosity >= 4) {
//         event.wait();
//         std::ostringstream oss;
//         oss << std::scientific << "cKernels spmv_blocked() time: " << t_spmv.stop() << " s";
//         OpmLog::info(oss.str());
//     }
// }
// 
// void cKernels::residual(cl::Buffer& vals, cl::Buffer& cols, cl::Buffer& rows,
//                             cl::Buffer& x, const cl::Buffer& rhs,
//                             cl::Buffer& out, int Nb, unsigned int block_size)
// {
//     const unsigned int work_group_size = 32;
//     const unsigned int num_work_groups = ceilDivision(Nb, work_group_size);
//     const unsigned int total_work_items = num_work_groups * work_group_size;
//     const unsigned int lmem_per_work_group = sizeof(double) * work_group_size;
//     Timer t_residual;
//     cl::Event event;
// 
//     if (block_size > 1) {
//         event = (*residual_blocked_k)(cl::EnqueueArgs(*queue, cl::NDRange(total_work_items), cl::NDRange(work_group_size)),
//                     vals, cols, rows, Nb, x, rhs, out, block_size, cl::Local(lmem_per_work_group));
//     } else {
//         event = (*residual_k)(cl::EnqueueArgs(*queue, cl::NDRange(total_work_items), cl::NDRange(work_group_size)),
//                     vals, cols, rows, Nb, x, rhs, out, cl::Local(lmem_per_work_group));
//     }
// 
//     if (verbosity >= 4) {
//         event.wait();
//         std::ostringstream oss;
//         oss << std::scientific << "cKernels residual_blocked() time: " << t_residual.stop() << " s";
//         OpmLog::info(oss.str());
//     }
// }
// 
// void cKernels::ILU_apply1(cl::Buffer& rowIndices, cl::Buffer& vals, cl::Buffer& cols,
//                                cl::Buffer& rows, cl::Buffer& diagIndex,
//                                const cl::Buffer& y, cl::Buffer& x,
//                                cl::Buffer& rowsPerColor, int color,
//                                int rowsThisColor, unsigned int block_size)
// {
//     const unsigned int work_group_size = preferred_workgroup_size_multiple;
//     const unsigned int num_work_groups = rowsThisColor;
//     const unsigned int total_work_items = num_work_groups * work_group_size;
//     const unsigned int lmem_per_work_group = sizeof(double) * work_group_size;
//     Timer t_ilu_apply1;
// 
//     cl::Event event = (*ILU_apply1_k)(cl::EnqueueArgs(*queue, cl::NDRange(total_work_items), cl::NDRange(work_group_size)),
//                           rowIndices, vals, cols, rows, diagIndex,
//                           y, x, rowsPerColor, color, block_size,
//                           cl::Local(lmem_per_work_group));
// 
//     if (verbosity >= 5) {
//         event.wait();
//         std::ostringstream oss;
//         oss << std::scientific << "cKernels ILU_apply1() time: " << t_ilu_apply1.stop() << " s";
//         OpmLog::info(oss.str());
//     }
// }
// 
// void cKernels::ILU_apply2(cl::Buffer& rowIndices, cl::Buffer& vals, cl::Buffer& cols,
//                                cl::Buffer& rows, cl::Buffer& diagIndex,
//                                cl::Buffer& invDiagVals, cl::Buffer& x,
//                                cl::Buffer& rowsPerColor, int color,
//                                int rowsThisColor, unsigned int block_size)
// {
//     const unsigned int work_group_size = preferred_workgroup_size_multiple;
//     const unsigned int num_work_groups = rowsThisColor;
//     const unsigned int total_work_items = num_work_groups * work_group_size;
//     const unsigned int lmem_per_work_group = sizeof(double) * work_group_size;
//     Timer t_ilu_apply2;
// 
//     cl::Event event = (*ILU_apply2_k)(cl::EnqueueArgs(*queue, cl::NDRange(total_work_items), cl::NDRange(work_group_size)),
//                           rowIndices, vals, cols, rows, diagIndex,
//                           invDiagVals, x, rowsPerColor, color, block_size,
//                           cl::Local(lmem_per_work_group));
// 
//     if (verbosity >= 5) {
//         event.wait();
//         std::ostringstream oss;
//         oss << std::scientific << "cKernels ILU_apply2() time: " << t_ilu_apply2.stop() << " s";
//         OpmLog::info(oss.str());
//     }
// }
// 
// void cKernels::ILU_decomp(int firstRow, int lastRow, cl::Buffer& rowIndices,
//                                cl::Buffer& vals, cl::Buffer& cols, cl::Buffer& rows,
//                                cl::Buffer& diagIndex, cl::Buffer& invDiagVals,
//                                int rowsThisColor, unsigned int block_size)
// {
//     const unsigned int work_group_size = 128;
//     const unsigned int num_work_groups = rowsThisColor;
//     const unsigned int total_work_items = num_work_groups * work_group_size;
//     const unsigned int num_hwarps_per_group = work_group_size / 16;
//     const unsigned int lmem_per_work_group = num_hwarps_per_group * block_size * block_size * sizeof(double);           // each block needs a pivot
//     Timer t_ilu_decomp;
// 
//     cl::Event event = (*ilu_decomp_k)(cl::EnqueueArgs(*queue, cl::NDRange(total_work_items), cl::NDRange(work_group_size)),
//                           firstRow, lastRow, rowIndices,
//                           vals, cols, rows,
//                           invDiagVals, diagIndex, rowsThisColor,
//                           cl::Local(lmem_per_work_group));
// 
//     if (verbosity >= 4) {
//         event.wait();
//         std::ostringstream oss;
//         oss << std::scientific << "cKernels ILU_decomp() time: " << t_ilu_decomp.stop() << " s";
//         OpmLog::info(oss.str());
//     }
// }
// 
// void cKernels::apply_stdwells(cl::Buffer& d_Cnnzs_ocl, cl::Buffer &d_Dnnzs_ocl, cl::Buffer &d_Bnnzs_ocl,
//     cl::Buffer &d_Ccols_ocl, cl::Buffer &d_Bcols_ocl, cl::Buffer &d_x, cl::Buffer &d_y,
//     int dim, int dim_wells, cl::Buffer &d_val_pointers_ocl, int num_std_wells)
// {
//     const unsigned int work_group_size = 32;
//     const unsigned int total_work_items = num_std_wells * work_group_size;
//     const unsigned int lmem1 = sizeof(double) * work_group_size;
//     const unsigned int lmem2 = sizeof(double) * dim_wells;
//     Timer t_apply_stdwells;
// 
//     cl::Event event = (*stdwell_apply_k)(cl::EnqueueArgs(*queue, cl::NDRange(total_work_items), cl::NDRange(work_group_size)),
//                           d_Cnnzs_ocl, d_Dnnzs_ocl, d_Bnnzs_ocl, d_Ccols_ocl, d_Bcols_ocl, d_x, d_y, dim, dim_wells, d_val_pointers_ocl,
//                           cl::Local(lmem1), cl::Local(lmem2), cl::Local(lmem2));
// 
//     if (verbosity >= 4) {
//         event.wait();
//         std::ostringstream oss;
//         oss << std::scientific << "cKernels apply_stdwells() time: " << t_apply_stdwells.stop() << " s";
//         OpmLog::info(oss.str());
//     }
// }
// 
// void cKernels::isaiL(cl::Buffer& diagIndex, cl::Buffer& colPointers, cl::Buffer& mapping, cl::Buffer& nvc,
//     cl::Buffer& luIdxs, cl::Buffer& xxIdxs, cl::Buffer& dxIdxs, cl::Buffer& LUvals, cl::Buffer& invLvals, unsigned int Nb)
// {
//     const unsigned int work_group_size = 256;
//     const unsigned int num_work_groups = ceilDivision(Nb, work_group_size);
//     const unsigned int total_work_items = num_work_groups * work_group_size;
// 
//     Timer t_isaiL;
//     cl::Event event = (*isaiL_k)(cl::EnqueueArgs(*queue, cl::NDRange(total_work_items), cl::NDRange(work_group_size)),
//             diagIndex, colPointers, mapping, nvc, luIdxs, xxIdxs, dxIdxs, LUvals, invLvals, Nb);
// 
//     if (verbosity >= 4) {
//         event.wait();
//         std::ostringstream oss;
//         oss << std::scientific << "cKernels isaiL() time: " << t_isaiL.stop() << " s";
//         OpmLog::info(oss.str());
//     }
// }
// 
// void cKernels::isaiU(cl::Buffer& diagIndex, cl::Buffer& colPointers, cl::Buffer& rowIndices, cl::Buffer& mapping,
//         cl::Buffer& nvc, cl::Buffer& luIdxs, cl::Buffer& xxIdxs, cl::Buffer& dxIdxs, cl::Buffer& LUvals,
//         cl::Buffer& invDiagVals, cl::Buffer& invUvals, unsigned int Nb)
// {
//     const unsigned int work_group_size = 256;
//     const unsigned int num_work_groups = ceilDivision(Nb, work_group_size);
//     const unsigned int total_work_items = num_work_groups * work_group_size;
// 
//     Timer t_isaiU;
//     cl::Event event = (*isaiU_k)(cl::EnqueueArgs(*queue, cl::NDRange(total_work_items), cl::NDRange(work_group_size)),
//             diagIndex, colPointers, rowIndices, mapping, nvc, luIdxs, xxIdxs, dxIdxs, LUvals, invDiagVals, invUvals, Nb);
// 
//     if (verbosity >= 4) {
//         event.wait();
//         std::ostringstream oss;
//         oss << std::scientific << "cKernels isaiU() time: " << t_isaiU.stop() << " s";
//         OpmLog::info(oss.str());
//     }
// }
// #define INSTANTIATE_BDA_FUNCTIONS(n)  \
// template class cKernels<n>;
// 
// INSTANTIATE_BDA_FUNCTIONS(1);
// INSTANTIATE_BDA_FUNCTIONS(2);
// INSTANTIATE_BDA_FUNCTIONS(3);
// INSTANTIATE_BDA_FUNCTIONS(4);
// INSTANTIATE_BDA_FUNCTIONS(5);
// INSTANTIATE_BDA_FUNCTIONS(6);
// 
// #undef INSTANTIATE_BDA_FUNCTIONS

} // namespace Accelerator
} // namespace Opm
