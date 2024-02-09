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

#include <config.h>
#include <memory>
#include <opm/common/TimingMacros.hpp>
#include <opm/common/ErrorMacros.hpp>

// #include <opm/simulators/linalg/bda/opencl/BILU0.hpp>
// #include <opm/simulators/linalg/bda/opencl/BISAI.hpp>
#include <opm/simulators/linalg/bda/c/cCPR.hpp>
#include <opm/simulators/linalg/bda/c/cPreconditioner.hpp>

namespace Opm
{
namespace Accelerator
{

template <unsigned int block_size>
std::unique_ptr<cPreconditioner<block_size> > cPreconditioner<block_size>::create(PreconditionerType type, int verbosity, bool opencl_ilu_parallel) {
//     if (type == PreconditionerType::BILU0) {
//         return std::make_unique<Opm::Accelerator::BILU0<block_size> >(opencl_ilu_parallel, verbosity);
/*/     } else*/ if (type == PreconditionerType::CPR) {
        return std::make_unique<Opm::Accelerator::cCPR<block_size> >(verbosity, opencl_ilu_parallel);
    } /*else if (type == PreconditionerType::BISAI) {*/
// //         return std::make_unique<Opm::Accelerator::BISAI<block_size> >(opencl_ilu_parallel, verbosity);
//     } else {
//         OPM_THROW(std::logic_error, "Invalid PreconditionerType");
//     }
}

template <unsigned int block_size>
bool cPreconditioner<block_size>::analyze_matrix(BlockedMatrix *mat, [[maybe_unused]] BlockedMatrix *jacMat) {
    return analyze_matrix(mat);
}

template <unsigned int block_size>
bool cPreconditioner<block_size>::create_preconditioner(BlockedMatrix *mat, [[maybe_unused]] BlockedMatrix *jacMat) {
    return create_preconditioner(mat);
}

//TODO: define macros if we define opencl or c based buffers or whatever
#define INSTANTIATE_BDA_FUNCTIONS(n)  \
template std::unique_ptr<cPreconditioner<n> > cPreconditioner<n>::create(PreconditionerType, int, bool);         \
template bool cPreconditioner<n>::analyze_matrix(BlockedMatrix *, BlockedMatrix *);                              \
template bool cPreconditioner<n>::create_preconditioner(BlockedMatrix *, BlockedMatrix *);

INSTANTIATE_BDA_FUNCTIONS(1);
INSTANTIATE_BDA_FUNCTIONS(2);
INSTANTIATE_BDA_FUNCTIONS(3);
INSTANTIATE_BDA_FUNCTIONS(4);
INSTANTIATE_BDA_FUNCTIONS(5);
INSTANTIATE_BDA_FUNCTIONS(6);

#undef INSTANTIATE_BDA_FUNCTIONS

} //namespace Accelerator
} //namespace Opm

