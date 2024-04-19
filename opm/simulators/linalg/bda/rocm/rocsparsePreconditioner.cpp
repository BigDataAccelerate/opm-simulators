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
#include <memory>
#include <opm/common/TimingMacros.hpp>
#include <opm/common/ErrorMacros.hpp>

#include <opm/simulators/linalg/bda/rocm/rocsparseBILU0.hpp>
#include <opm/simulators/linalg/bda/rocm/rocsparseCPR.hpp>
#include <opm/simulators/linalg/bda/rocm/rocsparsePreconditioner.hpp>

namespace Opm
{
namespace Accelerator
{

template <unsigned int block_size>
std::unique_ptr<rocsparsePreconditioner<block_size> > rocsparsePreconditioner<block_size>::create(PreconditionerType type, int verbosity) {
    if (type == PreconditionerType::BILU0) {
        return std::make_unique<Opm::Accelerator::rocsparseBILU0<block_size> >(verbosity);
    } else if (type == PreconditionerType::CPR) {
        return std::make_unique<Opm::Accelerator::rocsparseCPR<block_size> >(verbosity);
    } else {
        OPM_THROW(std::logic_error, "Invalid PreconditionerType");
    }
}

template <unsigned int block_size>
void rocsparsePreconditioner<block_size>::set_matrix_analysis(rocsparse_mat_descr descr_L, rocsparse_mat_descr descr_U)
{
    descr_L = descr_L;
    descr_U = descr_U;
}

template <unsigned int block_size>
void rocsparsePreconditioner<block_size>::set_context(rocsparse_handle handle, rocsparse_direction dir, rocsparse_operation operation, hipStream_t stream)
{
    handle = handle;
    dir = dir;
    operation = operation;
    stream = stream;
}

// template <unsigned int block_size>
// bool rocsparsePreconditioner<block_size>::analyze_matrix(BlockedMatrix *mat, BlockedMatrix *jacMat)
// {
//     const unsigned int bs = block_size;
// 
//     this->N = mat->Nb * block_size;
//     this->Nb = mat->Nb;
//     this->nnz = mat->nnzbs * block_size * block_size;
//     this->nnzb = mat->nnzbs;
// 
//     std::vector<int> CSCRowIndices;
//     std::vector<int> CSCColPointers;
// 
//     auto *matToDecompose = jacMat ? jacMat : mat; // decompose jacMat if valid, otherwise decompose mat
// 
//     if (opencl_ilu_parallel) {
//         toOrder.resize(Nb);
//         fromOrder.resize(Nb);
//         CSCRowIndices.resize(matToDecompose->nnzbs);
//         CSCColPointers.resize(Nb + 1);
// 
//         LUmat = std::make_unique<BlockedMatrix>(*matToDecompose);
// 
//         Timer t_convert;
//         csrPatternToCsc(matToDecompose->colIndices, matToDecompose->rowPointers, CSCRowIndices.data(), CSCColPointers.data(), Nb);
//         if(verbosity >= 3){
//             std::ostringstream out;
//             out << "openclBILU0 convert CSR to CSC: " << t_convert.stop() << " s";
//             OpmLog::info(out.str());
//         }
//     } else {
//         LUmat = std::make_unique<BlockedMatrix>(*matToDecompose);
//     }
// 
//     Timer t_analysis;
//     std::ostringstream out;
//     if (opencl_ilu_parallel) {
//         out << "opencl_ilu_parallel: true (level_scheduling)\n";
//         findLevelScheduling(matToDecompose->colIndices, matToDecompose->rowPointers, CSCRowIndices.data(), CSCColPointers.data(), Nb, &numColors, toOrder.data(), fromOrder.data(), rowsPerColor);
//     } else {
//         out << "opencl_ilu_parallel: false\n";
//         // numColors = 1;
//         // rowsPerColor.emplace_back(Nb);
//         numColors = Nb;
//         for(int i = 0; i < Nb; ++i){
//             rowsPerColor.emplace_back(1);
//         }
//     }
// 
//     if (verbosity >= 1) {
//         out << "openclBILU0 analysis took: " << t_analysis.stop() << " s, " << numColors << " colors\n";
//     }
// 
// #if CHOW_PATEL
//     out << "openclBILU0 CHOW_PATEL: " << CHOW_PATEL << ", CHOW_PATEL_GPU: " << CHOW_PATEL_GPU;
// #endif
//     OpmLog::info(out.str());
// 
//     diagIndex.resize(mat->Nb);
//     invDiagVals.resize(mat->Nb * bs * bs);
// 
// #if CHOW_PATEL
//     Lmat = std::make_unique<BlockedMatrix>(mat->Nb, (mat->nnzbs - mat->Nb) / 2, block_size);
//     Umat = std::make_unique<BlockedMatrix>(mat->Nb, (mat->nnzbs - mat->Nb) / 2, block_size);
// #endif
// 
//     s.invDiagVals = cl::Buffer(*context, CL_MEM_READ_WRITE, sizeof(double) * bs * bs * mat->Nb);
//     s.rowsPerColor = cl::Buffer(*context, CL_MEM_READ_WRITE, sizeof(int) * (numColors + 1));
//     s.diagIndex = cl::Buffer(*context, CL_MEM_READ_WRITE, sizeof(int) * LUmat->Nb);
//     s.rowIndices = cl::Buffer(*context, CL_MEM_READ_WRITE, sizeof(unsigned) * LUmat->Nb);
// #if CHOW_PATEL
//     s.Lvals = cl::Buffer(*context, CL_MEM_READ_WRITE, sizeof(double) * bs * bs * Lmat->nnzbs);
//     s.Lcols = cl::Buffer(*context, CL_MEM_READ_WRITE, sizeof(int) * Lmat->nnzbs);
//     s.Lrows = cl::Buffer(*context, CL_MEM_READ_WRITE, sizeof(int) * (Lmat->Nb + 1));
//     s.Uvals = cl::Buffer(*context, CL_MEM_READ_WRITE, sizeof(double) * bs * bs * Lmat->nnzbs);
//     s.Ucols = cl::Buffer(*context, CL_MEM_READ_WRITE, sizeof(int) * Lmat->nnzbs);
//     s.Urows = cl::Buffer(*context, CL_MEM_READ_WRITE, sizeof(int) * (Lmat->Nb + 1));
// #else
//     s.LUvals = cl::Buffer(*context, CL_MEM_READ_WRITE, sizeof(double) * bs * bs * LUmat->nnzbs);
//     s.LUcols = cl::Buffer(*context, CL_MEM_READ_WRITE, sizeof(int) * LUmat->nnzbs);
//     s.LUrows = cl::Buffer(*context, CL_MEM_READ_WRITE, sizeof(int) * (LUmat->Nb + 1));
// #endif
// 
//     events.resize(3);
//     err = queue->enqueueWriteBuffer(s.invDiagVals, CL_FALSE, 0, mat->Nb * sizeof(double) * bs * bs, invDiagVals.data(), nullptr, &events[0]);
// 
//     rowsPerColorPrefix.resize(numColors + 1); // resize initializes value 0.0
//     for (int i = 0; i < numColors; ++i) {
//         rowsPerColorPrefix[i + 1] = rowsPerColorPrefix[i] + rowsPerColor[i];
//     }
// 
//     err |= queue->enqueueWriteBuffer(s.rowsPerColor, CL_FALSE, 0, (numColors + 1) * sizeof(int), rowsPerColorPrefix.data(), nullptr, &events[1]);
// 
//     if (opencl_ilu_parallel) {
//         err |= queue->enqueueWriteBuffer(s.rowIndices, CL_FALSE, 0, Nb * sizeof(unsigned), fromOrder.data(), nullptr, &events[2]);
//     } else {
//         // fromOrder is not initialized, so use something else to fill s.rowIndices
//         // s.rowIndices[i] == i must hold, since every rowidx is mapped to itself (i.e. no actual mapping)
//         // rowsPerColorPrefix is misused here, it contains an increasing sequence (0, 1, 2, ...)
//         err |= queue->enqueueWriteBuffer(s.rowIndices, CL_FALSE, 0, Nb * sizeof(unsigned), rowsPerColorPrefix.data(), nullptr, &events[2]);
//     }
// 
//     cl::WaitForEvents(events);
//     events.clear();
//     if (err != CL_SUCCESS) {
//         // enqueueWriteBuffer is C and does not throw exceptions like C++ OpenCL
//         OPM_THROW(std::logic_error, "openclBILU0 OpenCL enqueueWriteBuffer error");
//     }
// 
//     return true;
// }

// template bool rocsparsePreconditioner<n>::analyze_matrix(BlockedMatrix *);  \
// template bool rocsparsePreconditioner<n>::analyze_matrix(BlockedMatrix *, BlockedMatrix *);                             

#define INSTANTIATE_BDA_FUNCTIONS(n)  \
template std::unique_ptr<rocsparsePreconditioner<n> > rocsparsePreconditioner<n>::create(PreconditionerType, int);         \
template void rocsparsePreconditioner<n>::set_matrix_analysis(rocsparse_mat_descr, rocsparse_mat_descr ); \
template void rocsparsePreconditioner<n>::set_context(rocsparse_handle, rocsparse_direction, rocsparse_operation, hipStream_t);


INSTANTIATE_BDA_FUNCTIONS(1);
INSTANTIATE_BDA_FUNCTIONS(2);
INSTANTIATE_BDA_FUNCTIONS(3);
INSTANTIATE_BDA_FUNCTIONS(4);
INSTANTIATE_BDA_FUNCTIONS(5);
INSTANTIATE_BDA_FUNCTIONS(6);

#undef INSTANTIATE_BDA_FUNCTIONS

} //namespace Accelerator
} //namespace Opm

