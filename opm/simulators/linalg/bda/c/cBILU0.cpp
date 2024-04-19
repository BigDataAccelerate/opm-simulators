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

#include <opm/simulators/linalg/bda/c/cBILU0.hpp> 
#include <opm/simulators/linalg/bda/Reorder.hpp>

#include <sstream>

#include <iostream> //Razvan

namespace Opm
{
namespace Accelerator
{

using Opm::OpmLog;
using Dune::Timer;

template <unsigned int block_size>
cBILU0<block_size>::cBILU0(bool opencl_ilu_parallel_, int verbosity_) : 
    cPreconditioner<block_size>(verbosity_)
{
    opencl_ilu_parallel = opencl_ilu_parallel_;
}


template <unsigned int block_size>
bool cBILU0<block_size>::analyze_matrix(BlockedMatrix *mat)
{
std::cout << "-----in : cBILU0<block_size>::analyze_matrix(BlockedMatrix *mat_) --> call analyze_matrix(mat, nullptr) \n";
    return analyze_matrix(mat, nullptr);
}


template <unsigned int block_size>
bool cBILU0<block_size>::analyze_matrix(BlockedMatrix *mat, BlockedMatrix *jacMat)
{
std::cout << "-----in : cBILU0<block_size>::analyze_matrix(mat_, jacMat) \n";
    const unsigned int bs = block_size;

    this->N = mat->Nb * block_size;
    this->Nb = mat->Nb;
    this->nnz = mat->nnzbs * block_size * block_size;
    this->nnzb = mat->nnzbs;

    std::vector<int> CSCRowIndices;
    std::vector<int> CSCColPointers;

    auto *matToDecompose = jacMat ? jacMat : mat; // decompose jacMat if valid, otherwise decompose mat

    if (opencl_ilu_parallel) {
        toOrder.resize(Nb);
        fromOrder.resize(Nb);
        CSCRowIndices.resize(matToDecompose->nnzbs);
        CSCColPointers.resize(Nb + 1);

        LUmat = std::make_unique<BlockedMatrix>(*matToDecompose);

        Timer t_convert;
        csrPatternToCsc(matToDecompose->colIndices, matToDecompose->rowPointers, CSCRowIndices.data(), CSCColPointers.data(), Nb);
        if(verbosity >= 3){
            std::ostringstream out;
            out << "cBILU0 convert CSR to CSC: " << t_convert.stop() << " s";
            OpmLog::info(out.str());
        }
    } else {
        LUmat = std::make_unique<BlockedMatrix>(*matToDecompose);
    }

    Timer t_analysis;
    std::ostringstream out;
    if (opencl_ilu_parallel) {
        out << "opencl_ilu_parallel: true (level_scheduling)\n";
        findLevelScheduling(matToDecompose->colIndices, matToDecompose->rowPointers, CSCRowIndices.data(), CSCColPointers.data(), Nb, &numColors, toOrder.data(), fromOrder.data(), rowsPerColor);
    } else {
        out << "opencl_ilu_parallel: false\n";
        // numColors = 1;
        // rowsPerColor.emplace_back(Nb);
        numColors = Nb;
        for(int i = 0; i < Nb; ++i){
            rowsPerColor.emplace_back(1);
        }
    }

    if (verbosity >= 1) {
        out << "cBILU0 analysis took: " << t_analysis.stop() << " s, " << numColors << " colors\n";
    }

    OpmLog::info(out.str());

    diagIndex.resize(mat->Nb);
    invDiagVals.resize(mat->Nb * bs * bs);

    s.invDiagVals = (double*)malloc(sizeof(double) * bs * bs * mat->Nb);
    s.rowsPerColor = (double*)malloc(sizeof(int) * (numColors + 1));
    s.diagIndex = (double*)malloc(sizeof(int) * LUmat->Nb);
    s.rowIndices = (double*)malloc(sizeof(unsigned) * LUmat->Nb);

    s.LUvals = (double*)malloc(sizeof(double) * bs * bs * LUmat->nnzbs);
    s.LUcols = (double*)malloc(sizeof(int) * LUmat->nnzbs);
    s.LUrows = (double*)malloc(sizeof(int) * (LUmat->Nb + 1));

    rowsPerColorPrefix.resize(numColors + 1); // resize initializes value 0.0
    for (int i = 0; i < numColors; ++i) {
        rowsPerColorPrefix[i + 1] = rowsPerColorPrefix[i] + rowsPerColor[i];
    }
    
std::cout << "-----out: cBILU0<block_size>::analyze_matrix(mat_, jacMat) \n";

    return true;
}



template <unsigned int block_size>
bool cBILU0<block_size>::create_preconditioner(BlockedMatrix *mat)
{
    return create_preconditioner(mat, nullptr);
}


template <unsigned int block_size>
bool cBILU0<block_size>::create_preconditioner(BlockedMatrix *mat, BlockedMatrix *jacMat)
{
    const unsigned int bs = block_size;

    auto *matToDecompose = jacMat ? jacMat : mat;

    // TODO: remove this copy by replacing inplace ilu decomp by out-of-place ilu decomp
    Timer t_copy;
    memcpy(LUmat->nnzValues, matToDecompose->nnzValues, sizeof(double) * bs * bs * matToDecompose->nnzbs);

    if (verbosity >= 3){
        std::ostringstream out;
        out << "cBILU0 memcpy: " << t_copy.stop() << " s";
        OpmLog::info(out.str());
    }
std::cout << " N = " << N << std::endl;
std::cout << " Nb = " << Nb << std::endl;
std::cout << " numColors = " << numColors << std::endl;

    Timer t_copyToGpu;

    std::call_once(pattern_uploaded, [&](){
        // find the positions of each diagonal block
        for (int row = 0; row < Nb; ++row) {
            int rowStart = LUmat->rowPointers[row];
            int rowEnd = LUmat->rowPointers[row+1];
            
            auto candidate = std::find(LUmat->colIndices + rowStart, LUmat->colIndices + rowEnd, row);
            assert(candidate != LUmat->colIndices + rowEnd);
            diagIndex[row] = candidate - LUmat->colIndices;
        }
    });

    if (verbosity >= 3) {
        std::ostringstream out;
        out << "cBILU0 copy to GPU: " << t_copyToGpu.stop() << " s";
        OpmLog::info(out.str());
    }

    Timer t_decomposition;
    std::ostringstream out;
    for (int color = 0; color < numColors; ++color) {
        const unsigned int firstRow = rowsPerColorPrefix[color];
        const unsigned int lastRow = rowsPerColorPrefix[color + 1];
        if (verbosity >= 5) {
            out << "color " << color << ": " << firstRow << " - " << lastRow << " = " << lastRow - firstRow << "\n";
        }
//         OpenclKernels::ILU_decomp(firstRow, lastRow, s.rowIndices,
//                                   s.LUvals, s.LUcols, s.LUrows, s.diagIndex,
//                                   s.invDiagVals, rowsPerColor[color], block_size);
    }

    if (verbosity >= 3) {
        out << "cBILU0 decomposition: " << t_decomposition.stop() << " s";
        OpmLog::info(out.str());
    }

    return true;
} // end create_preconditioner()


// kernels are blocking on an NVIDIA GPU, so waiting for events is not needed
// however, if individual kernel calls are timed, waiting for events is needed
// behavior on other GPUs is untested
template <unsigned int block_size>
void cBILU0<block_size>::apply( double& y, double& x)
{
    const double relaxation = 0.9;
    Timer t_apply;
    
std::cout << "############# TODO: cBILU0 apply" << std::endl;
std::cout << "   input:   block_size = " << block_size << std::endl;
std::cout << "   input:   numColors = " << numColors << std::endl;

//     for (int color = 0; color < numColors; ++color) {
// std::cout << "     input:   nodesPerColorPrefix[" <<color<<"] = "<< rowsPerColor[color] << std::endl;
//     }
std::cout << "   before:   x[3] = " << (&x)[3] << std::endl;
std::cout << "   before:   y[3] = " << (&y)[3] << std::endl;

//     for (int color = 0; color < numColors; ++color) {
// // std::cout << "      LT: color " << color << std::endl;
// //         OpenclKernels::ILU_apply1(s.rowIndices, s.LUvals, s.LUcols, s.LUrows,
// //                                   s.diagIndex, y, x, s.rowsPerColor,
// //                                   color, rowsPerColor[color], block_size);
// //         std::cout << "ILU_apply1 for color: " << color << std::endl;
//     }

std::cout << "   after L:   x[3] = " << (&x)[3] << std::endl;
//     for (int color = numColors - 1; color >= 0; --color) {
// // std::cout << "    UP:   color " << color << std::endl;
// //         OpenclKernels::ILU_apply2(s.rowIndices, s.LUvals, s.LUcols, s.LUrows,
// //                                   s.diagIndex, s.invDiagVals, x, s.rowsPerColor,
// //                                   color, rowsPerColor[color], block_size);
// //         std::cout << "ILU_apply2 for color: " << color << std::endl;
//     }

    // apply relaxation
//     OpenclKernels::scale(x, relaxation, N);
      std::cout << "----> TODO: OpenclKernels::scale(x, relaxation,N);\n"; 
std::cout << "---> after: x[3] = " << (&x)[3] << std::endl;
std::cout << "exiting in prec->apply in cBILU0\n";exit(0);

    if (verbosity >= 4) {
        std::ostringstream out;
        out << "cBILU0 apply: " << t_apply.stop() << " s";
        OpmLog::info(out.str());
    }
}

#define INSTANTIATE_BDA_FUNCTIONS(n) \
template class cBILU0<n>;

INSTANTIATE_BDA_FUNCTIONS(1);
INSTANTIATE_BDA_FUNCTIONS(2);
INSTANTIATE_BDA_FUNCTIONS(3);
INSTANTIATE_BDA_FUNCTIONS(4);
INSTANTIATE_BDA_FUNCTIONS(5);
INSTANTIATE_BDA_FUNCTIONS(6);

#undef INSTANTIATE_BDA_FUNCTIONS

} // namespace Accelerator
} // namespace Opm
