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

//      double dot(cl::Buffer& in1, cl::Buffer& in2, cl::Buffer& out, int N);
     double norm(double* in, double* out, int N);
//      void axpy(cl::Buffer& in, const double a, cl::Buffer& out, int N);
//      void scale(cl::Buffer& in, const double a, int N);
//      void vmul(const double alpha, cl::Buffer& in1, cl::Buffer& in2, cl::Buffer& out, int N);
//      void custom(cl::Buffer& p, cl::Buffer& v, cl::Buffer& r, const double omega, const double beta, int N);
//      void full_to_pressure_restriction(const cl::Buffer& fine_y, cl::Buffer& weights, cl::Buffer& coarse_y, int Nb);
//      void add_coarse_pressure_correction(cl::Buffer& coarse_x, cl::Buffer& fine_x, int pressure_idx, int Nb);
//      void prolongate_vector(const cl::Buffer& in, cl::Buffer& out, const cl::Buffer& cols, int N);
//      void spmv(cl::Buffer& vals, cl::Buffer& cols, cl::Buffer& rows, const cl::Buffer& x, cl::Buffer& b, int Nb, unsigned int block_size, bool reset = true, bool add = false);
//      void residual(cl::Buffer& vals, cl::Buffer& cols, cl::Buffer& rows, cl::Buffer& x, const cl::Buffer& rhs, cl::Buffer& out, int Nb, unsigned int block_size);
// 
//      void ILU_apply1(cl::Buffer& rowIndices, cl::Buffer& vals, cl::Buffer& cols, cl::Buffer& rows, cl::Buffer& diagIndex,
//         const cl::Buffer& y, cl::Buffer& x, cl::Buffer& rowsPerColor, int color, int Nb, unsigned int block_size);
// 
//      void ILU_apply2(cl::Buffer& rowIndices, cl::Buffer& vals, cl::Buffer& cols, cl::Buffer& rows, cl::Buffer& diagIndex,
//         cl::Buffer& invDiagVals, cl::Buffer& x, cl::Buffer& rowsPerColor, int color, int Nb, unsigned int block_size);
// 
//      void ILU_decomp(int firstRow, int lastRow, cl::Buffer& rowIndices, cl::Buffer& vals, cl::Buffer& cols, cl::Buffer& rows,
//         cl::Buffer& diagIndex, cl::Buffer& invDiagVals, int Nb, unsigned int block_size);
// 
//      void apply_stdwells(cl::Buffer& d_Cnnzs_ocl, cl::Buffer &d_Dnnzs_ocl, cl::Buffer &d_Bnnzs_ocl,
//         cl::Buffer &d_Ccols_ocl, cl::Buffer &d_Bcols_ocl, cl::Buffer &d_x, cl::Buffer &d_y,
//         int dim, int dim_wells, cl::Buffer &d_val_pointers_ocl, int num_std_wells);
// 
//      void isaiL(cl::Buffer& diagIndex, cl::Buffer& colPointers, cl::Buffer& mapping, cl::Buffer& nvc,
//             cl::Buffer& luIdxs, cl::Buffer& xxIdxs, cl::Buffer& dxIdxs, cl::Buffer& LUvals, cl::Buffer& invLvals, unsigned int Nb);
// 
//      void isaiU(cl::Buffer& diagIndex, cl::Buffer& colPointers, cl::Buffer& rowIndices, cl::Buffer& mapping,
//             cl::Buffer& nvc, cl::Buffer& luIdxs, cl::Buffer& xxIdxs, cl::Buffer& dxIdxs, cl::Buffer& LUvals,
//             cl::Buffer& invDiagVals, cl::Buffer& invUvals, unsigned int Nb);

};

} // namespace Accelerator
} // namespace Opm

#endif // OPM_CMATRIX_HEADER_INCLUDED
