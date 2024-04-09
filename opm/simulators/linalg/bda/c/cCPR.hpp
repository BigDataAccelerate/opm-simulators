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

#ifndef OPM_CCPR_HPP
#define OPM_CCPR_HPP

#include <mutex>

#include <opm/simulators/linalg/bda/c/cBILU0.hpp>
#include <opm/simulators/linalg/bda/Matrix.hpp>
#include <opm/simulators/linalg/bda/CprCreation.hpp>
#include <opm/simulators/linalg/bda/c/cMatrix.hpp>
#include <opm/simulators/linalg/bda/c/cPreconditioner.hpp>

#include <opm/simulators/linalg/bda/c/cSolverBackend.hpp>

namespace Opm
{
namespace Accelerator
{

class BlockedMatrix;

/// This class implements a Constrained Pressure Residual (CPR) preconditioner
template <unsigned int block_size>
class cCPR : public cPreconditioner<block_size>, public CprCreation<block_size>
{
    typedef cPreconditioner<block_size> Base;

    using Base::N;
    using Base::Nb;
    using Base::nnz;
    using Base::nnzb;
    using Base::verbosity;

private:
    std::vector<cMatrix> d_Amatrices, d_Rmatrices; // scalar matrices that represent the AMG hierarchy
    
    std::vector<std::vector<int> > d_PcolIndices; // prolongation does not need a full matrix, only store colIndices --> TODO: this can be a pointer to the previously defined CPU data structure
    std::vector<std::vector<double> > d_invDiags; // inverse of diagonal of Amatrices
    std::vector<double> d_t, d_f, d_u; // intermediate vectors used during amg cycle
    std::unique_ptr<double> d_rs;      // use before extracting the pressure
    std::unique_ptr<double> d_weights; // the quasiimpes weights, used to extract pressure
    std::unique_ptr<cMatrix> d_mat;   // stores blocked matrix
    std::unique_ptr<double> d_coarse_y, d_coarse_x; // stores the scalar vectors
    std::once_flag c_buffers_allocated;  // only allocate OpenCL Buffers once

    std::unique_ptr<cBILU0<block_size> > bilu0;                    // Blocked ILU0 preconditioner

    std::unique_ptr<cSolverBackend<1> > coarse_solver; // coarse solver is scalar
    bool opencl_ilu_parallel;                   // whether ILU0 operation should be parallelized

    // Initialize and allocate matrices and vectors
    void init_c_buffers();//TODO: rename to rocm/c and update in code

    // Copy matrices and vectors to GPU
    void c_upload();//TODO: rename to rocm/c and update in code

    // apply pressure correction to vector
    void apply_amg(const double& y, double& x);

    void amg_cycle_gpu(const int level, double &y, double &x);

public:

     cCPR(int verbosity, bool opencl_ilu_parallel);

    bool analyze_matrix(BlockedMatrix *mat);//TODO: factor out to CPRCreation preconditioner!!!
    bool analyze_matrix(BlockedMatrix *mat, BlockedMatrix *jacMat);//TODO: factor out to CPRCreation preconditioner!!!

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

