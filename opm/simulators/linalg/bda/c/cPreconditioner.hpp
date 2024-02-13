/*
  Copyright 2021 Equinor ASA

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

#ifndef OPM_CPRECONDITIONER_HEADER_INCLUDED
#define OPM_CPRECONDITIONER_HEADER_INCLUDED

#include <opm/simulators/linalg/bda/Preconditioner.hpp>

namespace Opm
{
namespace Accelerator
{


class BlockedMatrix;

template <unsigned int block_size>
class cPreconditioner : public Preconditioner<block_size>
{

protected:
    cPreconditioner(int verbosity_) :
    Preconditioner<block_size>(verbosity_)
    {};

public:
    virtual ~cPreconditioner() = default;
    
    static std::unique_ptr<cPreconditioner> create(PreconditionerType type, int verbosity, bool opencl_ilu_parallel);

    // apply preconditioner, x = prec(y)
    virtual void apply(double& y, double& x) = 0;

    // analyze matrix, e.g. the sparsity pattern
    // probably only called once
    // the version with two params can be overloaded, if not, it will default to using the one param version
//     virtual bool analyze_matrix(BlockedMatrix *mat) = 0;
//     virtual bool analyze_matrix(BlockedMatrix *mat, BlockedMatrix *jacMat);

    // create/update preconditioner, probably used every linear solve
    // the version with two params can be overloaded, if not, it will default to using the one param version
    virtual bool create_preconditioner(BlockedMatrix *mat) = 0;
    virtual bool create_preconditioner(BlockedMatrix *mat, BlockedMatrix *jacMat) = 0;
};

} //namespace Accelerator
} //namespace Opm

#endif
