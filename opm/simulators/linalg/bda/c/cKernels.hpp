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

#ifndef OPM_CKERNELS_HEADER_INCLUDED
#define OPM_CKERNELS_HEADER_INCLUDED

#include <vector>

namespace Opm
{
namespace Accelerator
{

// template <unsigned int block_size>
class cKernels {
private:
     int verbosity;
     std::vector<double> tmp; // used as tmp CPU buffer for dot() and norm()
     bool initialized;
    
public:
     cKernels(int verbosity_);

     double dot(double* in1, double* in2, double* out, int N);
     double norm(double* in, double* out, int N);
//      void axpy(double* in, const double a, double* out, int N);
//      void scale(double* in, const double a, int N);
//      void vmul(const double alpha, double* in1, double* in2, double* out, int N);
     void custom(double*  p, double* v, double* r, const double omega, const double beta, int N);
//      void full_to_pressure_restriction(const double* fine_y, double* weights, double* coarse_y, int Nb);
//      void add_coarse_pressure_correction(double* coarse_x, double* fine_x, int pressure_idx, int Nb);
//      void prolongate_vector(const double* in, double* out, const double* cols, int N);
//      void spmv(double* vals, double* cols, double* rows, const double* x, double* b, int Nb, unsigned int block_size, bool reset = true, bool add = false);
//      void residual(double* vals, double* cols, double* rows, double* x, const double* rhs, double* out, int Nb, unsigned int block_size);
// 
//      void ILU_apply1(double* rowIndices, double* vals, double* cols, double* rows, double* diagIndex,
//         const double* y, double* x, double* rowsPerColor, int color, int Nb, unsigned int block_size);
// 
//      void ILU_apply2(double* rowIndices, double* vals, double* cols, double* rows, double* diagIndex,
//         double* invDiagVals, double* x, double* rowsPerColor, int color, int Nb, unsigned int block_size);
// 
//      void ILU_decomp(int firstRow, int lastRow, double* rowIndices, double* vals, double* cols, double* rows,
//         double* diagIndex, double* invDiagVals, int Nb, unsigned int block_size);
// 
//      void apply_stdwells(double* d_Cnnzs_ocl, cl::Buffer &d_Dnnzs_ocl, cl::Buffer &d_Bnnzs_ocl,
//         cl::Buffer &d_Ccols_ocl, cl::Buffer &d_Bcols_ocl, cl::Buffer &d_x, cl::Buffer &d_y,
//         int dim, int dim_wells, cl::Buffer &d_val_pointers_ocl, int num_std_wells);
// 
//      void isaiL(double* diagIndex, double* colPointers, double* mapping, double* nvc,
//             double* luIdxs, double* xxIdxs, double* dxIdxs, double* LUvals, double* invLvals, unsigned int Nb);
// 
//      void isaiU(double* diagIndex, double* colPointers, double* rowIndices, double* mapping,
//             double* nvc, double* luIdxs, double* xxIdxs, double* dxIdxs, double* LUvals,
//             double* invDiagVals, double* invUvals, unsigned int Nb);

};

} // namespace Accelerator
} // namespace Opm

#endif // OPM_CMATRIX_HEADER_INCLUDED
