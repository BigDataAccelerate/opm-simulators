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

#ifndef OPM_ROCSPARSECPR_HPP
#define OPM_ROCSPARSECPR_HPP

#include <mutex>

#include <opm/simulators/linalg/bda/rocm/rocsparseBILU0.hpp>
#include <opm/simulators/linalg/bda/Matrix.hpp>
#include <opm/simulators/linalg/bda/CprCreation.hpp>
#include <opm/simulators/linalg/bda/c/cMatrix.hpp>
#include <opm/simulators/linalg/bda/rocm/rocsparsePreconditioner.hpp>

#include <opm/simulators/linalg/bda/rocm/rocsparseSolverBackend.hpp>

namespace Opm
{
namespace Accelerator
{

class BlockedMatrix;

/// This class implements a Constrained Pressure Residual (CPR) preconditioner
template <unsigned int block_size>
class rocsparseCPR : public rocsparsePreconditioner<block_size>, public CprCreation<block_size>
{
    typedef rocsparsePreconditioner<block_size> Base;

    using Base::N;
    using Base::Nb;
    using Base::nnz;
    using Base::nnzb;
    using Base::verbosity;

private:
    std::vector<cMatrix> d_Amatrices, d_Rmatrices; // scalar matrices that represent the AMG hierarchy
    
    std::vector<std::vector<int> > d_PcolIndices; // prolongation does not need a full matrix, only store colIndices
    std::vector<std::vector<double> > d_invDiags; // inverse of diagonal of Amatrices
    std::vector<double> d_t, d_f, d_u; // intermediate vectors used during amg cycle
    std::unique_ptr<double> d_rs;      // use before extracting the pressure
    std::unique_ptr<double> d_weights; // the quasiimpes weights, used to extract pressure
    std::unique_ptr<cMatrix> d_mat;   // stores blocked matrix
    std::unique_ptr<double> d_coarse_y, d_coarse_x; // stores the scalar vectors
    std::once_flag rocm_buffers_allocated;  // only allocate OpenCL Buffers once

    std::unique_ptr<rocsparseBILU0<block_size> > bilu0;                    // Blocked ILU0 preconditioner

    std::unique_ptr<rocsparseSolverBackend<1> > coarse_solver; // coarse solver is scalar
//     bool opencl_ilu_parallel;                   // whether ILU0 operation should be parallelized

    // Initialize and allocate matrices and vectors
    void init_rocm_buffers();//TODO: rename to rocm/c and update in code

    // Copy matrices and vectors to GPU
    void rocm_upload();//TODO: rename to rocm/c and update in code

    // apply pressure correction to vector
    void apply_amg(const double& y, double& x);

    void amg_cycle_gpu(const int level, double &y, double &x);

public:

    rocsparseCPR(int verbosity);

    bool initialize(std::shared_ptr<BlockedMatrix> matrix, std::shared_ptr<BlockedMatrix> jacMatrix, rocsparse_int *d_Arows, rocsparse_int *d_Acols);
    
    void copy_system_to_gpu(double *b);

    /// Reassign pointers, in case the addresses of the Dune variables have changed
    /// \param[in] vals           array of nonzeroes, each block is stored row-wise and contiguous, contains nnz values
    /// \param[in] b              input vector b, contains N values
//     void update_system(double *vals, double *b);

    /// Update linear system to GPU
    /// \param[in] b              input vector, contains N values
    void update_system_on_gpu(double *b);


    bool analyze_matrix(BlockedMatrix *mat);
    bool analyze_matrix(BlockedMatrix *mat, BlockedMatrix *jacMat);

    bool create_preconditioner(BlockedMatrix *mat);
    bool create_preconditioner(BlockedMatrix *mat, BlockedMatrix *jacMat);
    
#if HAVE_OPENCL
    // apply preconditioner, x = prec(y)
    void apply(const cl::Buffer& y, cl::Buffer& x) {}
#endif
    // applies blocked ilu0
    // also applies amg for pressure component
    void apply(double& y, double& x);
};

} // namespace Accelerator
} // namespace Opm

#endif

