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

#ifndef OPM_OPENCLMATRIX_HEADER_INCLUDED
#define OPM_OPENCLMATRIX_HEADER_INCLUDED

#include <vector>

namespace Opm
{
namespace Accelerator
{

class Matrix;
class BlockedMatrix;

/// This struct resembles a csr matrix, only doubles are supported
/// The matrix data is stored in OpenCL Buffers
class cMatrix {
public:

    cMatrix(int Nb_, int Mb_, int nnzbs_, unsigned int block_size_)
    : Nb(Nb_),
      Mb(Mb_),
      nnzbs(nnzbs_),
      block_size(block_size_)
    {
        nnzValues = (double*) malloc(sizeof(double) * block_size * block_size * nnzbs);
        colIndices = (int*) malloc(sizeof(int) * nnzbs);
        rowPointers = (int*) malloc(sizeof(int) * (Nb + 1));
    }

    double* nnzValues;
    int* colIndices;
    int* rowPointers;
    int Nb, Mb;
    int nnzbs;
    unsigned int block_size;
};

} // namespace Accelerator
} // namespace Opm

#endif // OPM_OPENCLMATRIX_HEADER_INCLUDED
