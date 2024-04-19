/*
  Copyright 2019 Equinor ASA

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
#include <algorithm>

#include <opm/common/OpmLog/OpmLog.hpp>
#include <opm/common/ErrorMacros.hpp>
#include <dune/common/timer.hh>

#include <opm/simulators/linalg/bda/rocm/rocsparseBILU0.hpp> 
#include <opm/simulators/linalg/bda/Reorder.hpp>

#include <sstream>

#include <iostream> //Razvan
#include <hip/hip_runtime_api.h>

#define HIP_CHECK(STAT)                                  \
    do {                                                 \
        const hipError_t stat = (STAT);                  \
        if(stat != hipSuccess)                           \
        {                                                \
            std::ostringstream oss;                      \
            oss << "rocsparseBILU0::hip ";       \
            oss << "error: " << hipGetErrorString(stat); \
            OPM_THROW(std::logic_error, oss.str());      \
        }                                                \
    } while(0)

#define ROCSPARSE_CHECK(STAT)                            \
    do {                                                 \
        const rocsparse_status stat = (STAT);            \
        if(stat != rocsparse_status_success)             \
        {                                                \
            std::ostringstream oss;                      \
            oss << "rocsparseBILU0::rocsparse "; \
            oss << "error: " << stat;                    \
            OPM_THROW(std::logic_error, oss.str());      \
        }                                                \
    } while(0)

#define ROCBLAS_CHECK(STAT)                              \
    do {                                                 \
        const rocblas_status stat = (STAT);              \
        if(stat != rocblas_status_success)               \
        {                                                \
            std::ostringstream oss;                      \
            oss << "rocsparseBILU0::rocblas ";   \
            oss << "error: " << stat;                    \
            OPM_THROW(std::logic_error, oss.str());      \
        }                                                \
    } while(0)

namespace Opm
{
namespace Accelerator
{

using Opm::OpmLog;
using Dune::Timer;

template <unsigned int block_size>
rocsparseBILU0<block_size>::rocsparseBILU0(int verbosity_) : 
    rocsparsePreconditioner<block_size>(verbosity_)
{
}

template <unsigned int block_size>
bool rocsparseBILU0<block_size>::initialize(std::shared_ptr<BlockedMatrix> matrix, std::shared_ptr<BlockedMatrix> jacMatrix, rocsparse_int *d_Arows, rocsparse_int *d_Acols) { 
    this->Nb = matrix->Nb;
    this->N = Nb * block_size;
    this->nnzb = matrix->nnzbs;
    this->nnz = nnzb * block_size * block_size;
    //nnzbs_prec = this->nnzb;

//     if (jacMatrix) {
//         useJacMatrix = true;
// //         nnzbs_prec = jacMatrix->nnzbs;
//     }

//     mat = matrix;
    this->jacMat = jacMatrix;

//     HIP_CHECK(hipMalloc((void**)&d_r, sizeof(double) * N));
//     HIP_CHECK(hipMalloc((void**)&d_rw, sizeof(double) * N));
//     HIP_CHECK(hipMalloc((void**)&d_p, sizeof(double) * N));
//     HIP_CHECK(hipMalloc((void**)&d_pw, sizeof(double) * N));
//     HIP_CHECK(hipMalloc((void**)&d_s, sizeof(double) * N));
//     HIP_CHECK(hipMalloc((void**)&d_t, sizeof(double) * N));
//     HIP_CHECK(hipMalloc((void**)&d_v, sizeof(double) * N));

//     HIP_CHECK(hipMalloc((void**)&d_Arows, sizeof(rocsparse_int) * (Nb + 1)));
//     HIP_CHECK(hipMalloc((void**)&d_Acols, sizeof(rocsparse_int) * nnzb));
//     HIP_CHECK(hipMalloc((void**)&d_Avals, sizeof(double) * nnz));
//     HIP_CHECK(hipMalloc((void**)&d_x, sizeof(double) * N));
//     HIP_CHECK(hipMalloc((void**)&d_b, sizeof(double) * N));

    if (this->useJacMatrix) {
        HIP_CHECK(hipMalloc((void**)&d_Mrows, sizeof(rocsparse_int) * (Nb + 1)));
        HIP_CHECK(hipMalloc((void**)&d_Mcols, sizeof(rocsparse_int) * this->nnzbs_prec));
        HIP_CHECK(hipMalloc((void**)&d_Mvals, sizeof(double) * this->nnzbs_prec * block_size * block_size));
    } else { // preconditioner matrix is same
        HIP_CHECK(hipMalloc((void**)&d_Mvals, sizeof(double) * this->nnzbs_prec * block_size * block_size));
        d_Mcols = d_Acols;
        d_Mrows = d_Arows;
    }
    
    return true;
} // end initialize()

template <unsigned int block_size>
bool rocsparseBILU0<block_size>::analyze_matrix(BlockedMatrix *mat)
{
// std::cout << "-----in : cBILU0<block_size>::analyze_matrix(BlockedMatrix *mat_) --> call analyze_matrix(mat, nullptr) \n";
    
    return analyze_matrix(mat, &(*this->jacMat));
}


template <unsigned int block_size>
bool rocsparseBILU0<block_size>::analyze_matrix(BlockedMatrix *mat, BlockedMatrix *jacMat)
{
std::cout << "-----in : cBILU0<block_size>::analyze_matrix(mat_, jacMat) \n";
    std::size_t d_bufferSize_M, d_bufferSize_L, d_bufferSize_U, d_bufferSize;
    Timer t;
    
     auto *matToDecompose = jacMat ? jacMat : mat; // decompose jacMat if valid, otherwise decompose mat

//     ROCSPARSE_CHECK(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));

    ROCSPARSE_CHECK(rocsparse_create_mat_info(&ilu_info));
// #if HIP_VERSION >= 50400000
//     ROCSPARSE_CHECK(rocsparse_create_mat_info(&spmv_info));
// #endif

//     ROCSPARSE_CHECK(rocsparse_create_mat_descr(&descr_A));
    ROCSPARSE_CHECK(rocsparse_create_mat_descr(&descr_M));

    ROCSPARSE_CHECK(rocsparse_create_mat_descr(&descr_L));
    ROCSPARSE_CHECK(rocsparse_set_mat_fill_mode(descr_L, rocsparse_fill_mode_lower));
    ROCSPARSE_CHECK(rocsparse_set_mat_diag_type(descr_L, rocsparse_diag_type_unit));

    ROCSPARSE_CHECK(rocsparse_create_mat_descr(&descr_U));
    ROCSPARSE_CHECK(rocsparse_set_mat_fill_mode(descr_U, rocsparse_fill_mode_upper));
    ROCSPARSE_CHECK(rocsparse_set_mat_diag_type(descr_U, rocsparse_diag_type_non_unit));

    ROCSPARSE_CHECK(rocsparse_dbsrilu0_buffer_size(handle, dir, Nb, this->nnzbs_prec,
                                 descr_M, d_Mvals, d_Mrows, d_Mcols, block_size, ilu_info, &d_bufferSize_M));
    ROCSPARSE_CHECK(rocsparse_dbsrsv_buffer_size(handle, dir, operation, Nb, this->nnzbs_prec,
                               descr_L, d_Mvals, d_Mrows, d_Mcols, block_size, ilu_info, &d_bufferSize_L));
    ROCSPARSE_CHECK(rocsparse_dbsrsv_buffer_size(handle, dir, operation, Nb, this->nnzbs_prec,
                               descr_U, d_Mvals, d_Mrows, d_Mcols, block_size, ilu_info, &d_bufferSize_U));

    d_bufferSize = std::max(d_bufferSize_M, std::max(d_bufferSize_L, d_bufferSize_U));

    HIP_CHECK(hipMalloc((void**)&d_buffer, d_bufferSize));

    // analysis of ilu LU decomposition
    ROCSPARSE_CHECK(rocsparse_dbsrilu0_analysis(handle, dir, \
                               Nb, this->nnzbs_prec, descr_M, d_Mvals, d_Mrows, d_Mcols, \
                               block_size, ilu_info, rocsparse_analysis_policy_reuse, rocsparse_solve_policy_auto, d_buffer));

    int zero_position = 0;
    rocsparse_status status = rocsparse_bsrilu0_zero_pivot(handle, ilu_info, &zero_position);
    if (rocsparse_status_success != status) {
        printf("L has structural and/or numerical zero at L(%d,%d)\n", zero_position, zero_position);
        return false;
    }

    // analysis of ilu apply
    ROCSPARSE_CHECK(rocsparse_dbsrsv_analysis(handle, dir, operation, \
                             Nb, this->nnzbs_prec, descr_L, d_Mvals, d_Mrows, d_Mcols, \
                             block_size, ilu_info, rocsparse_analysis_policy_reuse, rocsparse_solve_policy_auto, d_buffer));
    ROCSPARSE_CHECK(rocsparse_dbsrsv_analysis(handle, dir, operation, \
                             Nb, this->nnzbs_prec, descr_U, d_Mvals, d_Mrows, d_Mcols, \
                             block_size, ilu_info, rocsparse_analysis_policy_reuse, rocsparse_solve_policy_auto, d_buffer));

// #if HIP_VERSION >= 50400000 //TODO-R: when is descr_A actually used???
//     ROCSPARSE_CHECK(rocsparse_dbsrmv_ex_analysis(handle, dir, operation,
//         Nb, Nb, nnzb,
//         descr_A, d_Avals, d_Arows, d_Acols,
//         block_size, spmv_info));
// #endif

    if (verbosity >= 3) {
        HIP_CHECK(hipStreamSynchronize(stream));
        std::ostringstream out;
        out << "rocsparseSolver::analyze_matrix(): " << t.stop() << " s";
        OpmLog::info(out.str());
    }

std::cout << "-----out: cBILU0<block_size>::analyze_matrix(mat_, jacMat) \n";

    return true;
}



template <unsigned int block_size>
bool rocsparseBILU0<block_size>::create_preconditioner(BlockedMatrix *mat)
{
    return create_preconditioner(mat, &*this->jacMat);
}

//TODO: this is opencl code , should be changed into rocsparse code!!!!
template <unsigned int block_size>
bool rocsparseBILU0<block_size>::create_preconditioner(BlockedMatrix *mat, BlockedMatrix *jacMat)
{
    Timer t;

    bool result = true;
    ROCSPARSE_CHECK(rocsparse_dbsrilu0(handle, dir, Nb, this->nnzbs_prec, descr_M,
                    d_Mvals, d_Mrows, d_Mcols, block_size, ilu_info, rocsparse_solve_policy_auto, d_buffer));

    // Check for zero pivot
    int zero_position = 0;
    rocsparse_status status = rocsparse_bsrilu0_zero_pivot(handle, ilu_info, &zero_position);
    if(rocsparse_status_success != status)
    {
        printf("L has structural and/or numerical zero at L(%d,%d)\n", zero_position, zero_position);
        return false;
    }

    if (verbosity >= 3) {
        HIP_CHECK(hipStreamSynchronize(stream));
        std::ostringstream out;
        out << "rocsparseSolver::create_preconditioner(): " << t.stop() << " s";
        OpmLog::info(out.str());
    }
    return result;
} // end create_preconditioner()

template <unsigned int block_size>
void rocsparseBILU0<block_size>::copy_system_to_gpu(double *d_Avals) {
    Timer t;

//     HIP_CHECK(hipMemcpyAsync(d_Arows, mat->rowPointers, sizeof(rocsparse_int) * (Nb + 1), hipMemcpyHostToDevice, stream));
//     HIP_CHECK(hipMemcpyAsync(d_Acols, mat->colIndices, sizeof(rocsparse_int) * nnzb, hipMemcpyHostToDevice, stream));
//     HIP_CHECK(hipMemcpyAsync(d_Avals, mat->nnzValues, sizeof(double) * nnz, hipMemcpyHostToDevice, stream));
    if (this->useJacMatrix) {
        HIP_CHECK(hipMemcpyAsync(d_Mrows, this->jacMat->rowPointers, sizeof(rocsparse_int) * (Nb + 1), hipMemcpyHostToDevice, stream));
        HIP_CHECK(hipMemcpyAsync(d_Mcols, this->jacMat->colIndices, sizeof(rocsparse_int) * this->nnzbs_prec, hipMemcpyHostToDevice, stream));
        HIP_CHECK(hipMemcpyAsync(d_Mvals, this->jacMat->nnzValues, sizeof(double) * this->nnzbs_prec * block_size * block_size, hipMemcpyHostToDevice, stream));
    } else {
        HIP_CHECK(hipMemcpyAsync(d_Mvals, d_Avals, sizeof(double) * nnz, hipMemcpyDeviceToDevice, stream));
    }
//     HIP_CHECK(hipMemsetAsync(d_x, 0, sizeof(double) * N, stream));
//     HIP_CHECK(hipMemcpyAsync(d_b, b, sizeof(double) * N, hipMemcpyHostToDevice, stream));

    if (verbosity >= 3) {
        HIP_CHECK(hipStreamSynchronize(stream));
        std::ostringstream out;
        out << "rocsparseSolver::copy_system_to_gpu(): " << t.stop() << " s";
        OpmLog::info(out.str());
    }
} // end copy_system_to_gpu()

// don't copy rowpointers and colindices, they stay the same
template <unsigned int block_size>
void rocsparseBILU0<block_size>::update_system_on_gpu(double *d_Avals) {
    Timer t;

//     HIP_CHECK(hipMemcpyAsync(d_Avals, mat->nnzValues, sizeof(double) * nnz, hipMemcpyHostToDevice, stream));
    if (this->useJacMatrix) {
        HIP_CHECK(hipMemcpyAsync(d_Mvals, this->jacMat->nnzValues, sizeof(double) * this->nnzbs_prec * block_size * block_size, hipMemcpyHostToDevice, stream));
    } else {
        HIP_CHECK(hipMemcpyAsync(d_Mvals, d_Avals, sizeof(double) * nnz, hipMemcpyDeviceToDevice, stream));
    }
//     HIP_CHECK(hipMemsetAsync(d_x, 0, sizeof(double) * N, stream));
//     HIP_CHECK(hipMemcpyAsync(d_b, b, sizeof(double) * N, hipMemcpyHostToDevice, stream));

    if (verbosity >= 3) {
        HIP_CHECK(hipStreamSynchronize(stream));
        std::ostringstream out;
        out << "rocsparseSolver::update_system_on_gpu(): " << t.stop() << " s";
        OpmLog::info(out.str());
    }
} // end update_system_on_gpu()

// kernels are blocking on an NVIDIA GPU, so waiting for events is not needed
// however, if individual kernel calls are timed, waiting for events is needed
// behavior on other GPUs is untested
template <unsigned int block_size>
void rocsparseBILU0<block_size>::apply( double& y, double& x)
{
    const double relaxation = 0.9;
    double zero = 0.0;
    double one  = 1.0;

    Timer t_apply;

        // apply ilu0
//         prec->apply(*d_p, *d_pw);
        rocsparse_dbsrsv_solve(this->handle, this->dir, \
                              this->operation, Nb, this->nnzbs_prec, &one, \
                              this->descr_L, d_Mvals, d_Mrows, d_Mcols, block_size, ilu_info, &y, d_t, rocsparse_solve_policy_auto, d_buffer);
//         ROCSPARSE_CHECK(
        rocsparse_dbsrsv_solve(handle, this->dir, \
                              this->operation, Nb, this->nnzbs_prec, &one, \
                              descr_U, d_Mvals, d_Mrows, d_Mcols, block_size, ilu_info, d_t, &x, rocsparse_solve_policy_auto, d_buffer);
        //);
        
    for (int color = 0; color < numColors; ++color) {
// #if CHOW_PATEL
//         OpenclKernels::ILU_apply1(s.rowIndices, s.Lvals, s.Lcols, s.Lrows,
//                                   s.diagIndex, y, x, s.rowsPerColor,
//                                   color, rowsPerColor[color], block_size);
// #else
//         OpenclKernels::ILU_apply1(s.rowIndices, s.LUvals, s.LUcols, s.LUrows,
//                                   s.diagIndex, y, x, s.rowsPerColor,
//                                   color, rowsPerColor[color], block_size);
//         std::cout << "ILU_apply1 for color: " << color << std::endl;
// #endif
    }

    for (int color = numColors - 1; color >= 0; --color) {
// #if CHOW_PATEL
//         OpenclKernels::ILU_apply2(s.rowIndices, s.Uvals, s.Ucols, s.Urows,
//                                   s.diagIndex, s.invDiagVals, x, s.rowsPerColor,
//                                   color, rowsPerColor[color], block_size);
// #else
//         OpenclKernels::ILU_apply2(s.rowIndices, s.LUvals, s.LUcols, s.LUrows,
//                                   s.diagIndex, s.invDiagVals, x, s.rowsPerColor,
//                                   color, rowsPerColor[color], block_size);
//         std::cout << "ILU_apply2 for color: " << color << std::endl;
// #endif
    }

    // apply relaxation
//     OpenclKernels::scale(x, relaxation, N);
       std::cout << "scale kernel" << std::endl;

    if (verbosity >= 4) {
        std::ostringstream out;
        out << "cBILU0 apply: " << t_apply.stop() << " s";
        OpmLog::info(out.str());
    }
}

#define INSTANTIATE_BDA_FUNCTIONS(n) \
template class rocsparseBILU0<n>;

INSTANTIATE_BDA_FUNCTIONS(1);
INSTANTIATE_BDA_FUNCTIONS(2);
INSTANTIATE_BDA_FUNCTIONS(3);
INSTANTIATE_BDA_FUNCTIONS(4);
INSTANTIATE_BDA_FUNCTIONS(5);
INSTANTIATE_BDA_FUNCTIONS(6);

#undef INSTANTIATE_BDA_FUNCTIONS

} // namespace Accelerator
} // namespace Opm
