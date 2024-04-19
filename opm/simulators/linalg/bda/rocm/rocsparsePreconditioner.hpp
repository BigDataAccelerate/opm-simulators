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

#ifndef OPM_ROCSPARSEPRECONDITIONER_HEADER_INCLUDED
#define OPM_ROCSPARSEPRECONDITIONER_HEADER_INCLUDED

#include <opm/simulators/linalg/bda/Preconditioner.hpp>

#include <rocsparse/rocsparse.h>

namespace Opm
{
namespace Accelerator
{


class BlockedMatrix;

template <unsigned int block_size>
class rocsparsePreconditioner : public Preconditioner<block_size>
{

protected:
    rocsparsePreconditioner(int verbosity_) :
    Preconditioner<block_size>(verbosity_)
    {};
    
    rocsparse_handle handle;
    rocsparse_direction dir = rocsparse_direction_row;
    rocsparse_operation operation = rocsparse_operation_none;
    rocsparse_mat_descr descr_L, descr_U;
    
    hipStream_t stream;
    
public:
    
    int nnzbs_prec = 0;    // number of nnz blocks in preconditioner matrix M
    bool useJacMatrix = false;
    std::shared_ptr<BlockedMatrix> jacMat = nullptr;              // matrix for preconditioner
    
    virtual ~rocsparsePreconditioner() = default;

    static std::unique_ptr<rocsparsePreconditioner<block_size>> create(PreconditionerType type, int verbosity);

    // apply preconditioner, x = prec(y)
    virtual void apply(double& y, double& x) = 0;
 
    // analyze matrix, e.g. the sparsity pattern
    // probably only called once
    // the version with two params can be overloaded, if not, it will default to using the one param version
//     virtual bool analyze_matrix(BlockedMatrix *mat) override;
//     virtual bool analyze_matrix(BlockedMatrix *mat, BlockedMatrix *jacMat) override;

    // create/update preconditioner, probably used every linear solve
    // the version with two params can be overloaded, if not, it will default to using the one param version
    virtual bool create_preconditioner(BlockedMatrix *mat) = 0;
    virtual bool create_preconditioner(BlockedMatrix *mat, BlockedMatrix *jacMat) = 0;
    
    virtual bool initialize(std::shared_ptr<BlockedMatrix> matrix, std::shared_ptr<BlockedMatrix> jacMatrix, rocsparse_int *d_Arows, rocsparse_int *d_Acols) = 0;
    virtual void copy_system_to_gpu(double *b)=0;

    /// Reassign pointers, in case the addresses of the Dune variables have changed
    /// \param[in] vals           array of nonzeroes, each block is stored row-wise and contiguous, contains nnz values
    /// \param[in] b              input vector b, contains N values
//     virtual void update_system(double *vals, double *b)=0;

    /// Update linear system to GPU
    /// \param[in] b              input vector, contains N values
    virtual void update_system_on_gpu(double *b)=0;

    void set_matrix_analysis(rocsparse_mat_descr descr_L, rocsparse_mat_descr descr_U);
    void set_context(rocsparse_handle handle, rocsparse_direction dir, rocsparse_operation operation, hipStream_t stream);
};

} //namespace Accelerator
} //namespace Opm

#endif
