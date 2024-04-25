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
#include <cmath>
#include <sstream>

#include <opm/common/OpmLog/OpmLog.hpp>
#include <opm/common/ErrorMacros.hpp>
#include <dune/common/timer.hh>

#include <opm/simulators/linalg/bda/rocm/hipKernels.hpp>
// #include <opm/simulators/linalg/bda/opencl/ChowPatelIlu.hpp>  // defines CHOW_PATEL

#include <opm/simulators/linalg/bda/Misc.hpp>

#include <iostream> //Razvan to delete

namespace Opm
{
namespace Accelerator
{

using Opm::OpmLog;
using Dune::Timer;

// define static variables and kernels
int HipKernels::verbosity;
double* HipKernels::tmp;
bool HipKernels::initialized = false;
std::size_t HipKernels::preferred_workgroup_size_multiple = 0;

#ifdef __HIP__
/// HIP kernel to apply the standard wellcontributions
__global__ void vmul_k()
{
    // from vmul.cl <-- KERNEL vmul_k
// // multiply vector with another vector and a scalar, element-wise
// // add result to a third vector
// __kernel void vmul(
//     const double alpha,
//     __global double const *in1,
//     __global double const *in2,
//     __global double *out,
//     const int N)
// {
//     unsigned int NUM_THREADS = get_global_size(0);
//     int idx = get_global_id(0);
// 
//     while(idx < N){
//         out[idx] += alpha * in1[idx] * in2[idx];
//         idx += NUM_THREADS;
//     }
// }

}

/// HIP kernel to apply the standard wellcontributions
__global__ void full_to_pressure_restriction_k()
{
// from full_to_pressure_restriction.cl --> KERNEL full_to_pressure_restriction_k
// transform blocked vector to scalar vector using pressure-weights
// every workitem handles one blockrow
// __kernel void full_to_pressure_restriction(
//     __global const double *fine_y,
//     __global const double *weights,
//     __global double *coarse_y,
//     const unsigned int Nb)
// {
//     const unsigned int NUM_THREADS = get_global_size(0);
//     const unsigned int block_size = 3;
//     unsigned int target_block_row = get_global_id(0);
// 
//     while(target_block_row < Nb){
//         double sum = 0.0;
//         unsigned int idx = block_size * target_block_row;
//         for (unsigned int i = 0; i < block_size; ++i) {
//             sum += fine_y[idx + i] * weights[idx + i];
//         }
//         coarse_y[target_block_row] = sum;
//         target_block_row += NUM_THREADS;
//     }
// }    
}

/// HIP kernel to apply the standard wellcontributions
__global__ void add_coarse_pressure_correction_k()
{
// from add_coarse_pressure_correction.cl --> KERNEL add_coarse_pressure_correction_k
// // add the coarse pressure solution back to the finer, complete solution
// // every workitem handles one blockrow
// __kernel void add_coarse_pressure_correction(
//     __global const double *coarse_x,
//     __global double *fine_x,
//     const unsigned int pressure_idx,
//     const unsigned int Nb)
// {
//     const unsigned int NUM_THREADS = get_global_size(0);
//     const unsigned int block_size = 3;
//     unsigned int target_block_row = get_global_id(0);
// 
//     while(target_block_row < Nb){
//         fine_x[target_block_row * block_size + pressure_idx] += coarse_x[target_block_row];
//         target_block_row += NUM_THREADS;
//     }
// }
}

/// HIP kernel to apply the standard wellcontributions
__global__ void prolongate_vector_k(
            const double *Cnnzs,
            const double *Dnnzs,
            const double *Bnnzs,
            const unsigned *Ccols,
            const unsigned *Bcols,
            const double *x,
            double *y,
            const unsigned dim,
            const unsigned dim_wells,
            const unsigned *val_pointers)
{
    // KERNEL prolongate_vector_k
// prolongate vector during amg cycle
// every workitem handles one row
// __kernel void prolongate_vector(
//     __global const double *in,
//     __global double *out,
//     __global const int *cols,
//     const unsigned int N)
// {
//     const unsigned int NUM_THREADS = get_global_size(0);
//     unsigned int row = get_global_id(0);
// 
//     while(row < N){
//         out[row] += in[cols[row]];
//         row += NUM_THREADS;
//     }
// }

// CODE FROM std_wells
//     unsigned wgId = blockIdx.x;
//     unsigned wiId = threadIdx.x;
//     unsigned valSize = val_pointers[wgId + 1] - val_pointers[wgId];
//     unsigned valsPerBlock = dim*dim_wells;
//     unsigned numActiveWorkItems = (blockDim.x/valsPerBlock)*valsPerBlock;
//     unsigned numBlocksPerWarp = blockDim.x/valsPerBlock;
//     unsigned c = wiId % dim;
//     unsigned r = (wiId/dim) % dim_wells;
//     double temp;
// 
//     extern __shared__ double localSum[];
//     double *z1 = localSum + gridDim.x;
//     double *z2 = z1 + dim_wells;
// 
//     localSum[wiId] = 0;
//     if(wiId < numActiveWorkItems){
//         unsigned b = wiId/valsPerBlock + val_pointers[wgId];
//         while(b < valSize + val_pointers[wgId]){
//             int colIdx = Bcols[b];
//             localSum[wiId] += Bnnzs[b*dim*dim_wells + r*dim + c]*x[colIdx*dim + c];
//             b += numBlocksPerWarp;
//         }
// 
//         // merge all blocks in this workgroup into 1 block
//         // if numBlocksPerWarp >= 3, should use loop
//         // block 1:     block 2:
//         //  0  1  2     12 13 14
//         //  3  4  5     15 16 17
//         //  6  7  8     18 19 20
//         //  9 10 11     21 22 23
//         // workitem i will hold the sum of workitems i and i + valsPerBlock
//         if(wiId < valsPerBlock){
//             for (unsigned i = 1; i < numBlocksPerWarp; ++i) {
//                 localSum[wiId] += localSum[wiId + i*valsPerBlock];
// 	    }
//         }
// 
//         if(c == 0 && wiId < valsPerBlock){
//             for(unsigned i = dim - 1; i > 0; --i){
//                 localSum[wiId] += localSum[wiId + i];
//             }
//             z1[r] = localSum[wiId];
//         }
//     }
// 
//     __syncthreads();
// 
//     if(wiId < dim_wells){
//         temp = 0.0;
//         for(unsigned i = 0; i < dim_wells; ++i){
//             temp += Dnnzs[wgId*dim_wells*dim_wells + wiId*dim_wells + i]*z1[i];
//         }
//         z2[wiId] = temp;
//     }
// 
//     __syncthreads();
// 
//     if(wiId < dim*valSize){
//         temp = 0.0;
//         unsigned bb = wiId/dim + val_pointers[wgId];
//         for (unsigned j = 0; j < dim_wells; ++j){
//             temp += Cnnzs[bb*dim*dim_wells + j*dim + c]*z2[j];
//         }
// 
//         int colIdx = Ccols[bb];
//         y[colIdx*dim + c] -= temp;
//     }
}

/// HIP kernel to apply the standard wellcontributions
__global__ void residual_blocked_k()
{
    // KERNEL residual_blocked_k    
// res = rhs - mat * x
// algorithm based on:
// Optimization of Block Sparse Matrix-Vector Multiplication on Shared-MemoryParallel Architectures,
// Ryan Eberhardt, Mark Hoemmen, 2016, https://doi.org/10.1109/IPDPSW.2016.42
// __kernel void residual_blocked(
//     __global const double *vals,
//     __global const int *cols,
//     __global const int *rows,
//     const int Nb,
//     __global const double *x,
//     __global const double *rhs,
//     __global double *out,
//     const unsigned int block_size,
//     __local double *tmp)
// {
//     const unsigned int warpsize = 32;
//     const unsigned int bsize = get_local_size(0);
//     const unsigned int idx_b = get_global_id(0) / bsize;
//     const unsigned int idx_t = get_local_id(0);
//     unsigned int idx = idx_b * bsize + idx_t;
//     const unsigned int bs = block_size;
//     const unsigned int num_active_threads = (warpsize/bs/bs)*bs*bs;
//     const unsigned int num_blocks_per_warp = warpsize/bs/bs;
//     const unsigned int NUM_THREADS = get_global_size(0);
//     const unsigned int num_warps_in_grid = NUM_THREADS / warpsize;
//     unsigned int target_block_row = idx / warpsize;
//     const unsigned int lane = idx_t % warpsize;
//     const unsigned int c = (lane / bs) % bs;
//     const unsigned int r = lane % bs;
// 
//     // for 3x3 blocks:
//     // num_active_threads: 27
//     // num_blocks_per_warp: 3
// 
//     while(target_block_row < Nb){
//         unsigned int first_block = rows[target_block_row];
//         unsigned int last_block = rows[target_block_row+1];
//         unsigned int block = first_block + lane / (bs*bs);
//         double local_out = 0.0;
// 
//         if(lane < num_active_threads){
//             for(; block < last_block; block += num_blocks_per_warp){
//                 double x_elem = x[cols[block]*bs + c];
//                 double A_elem = vals[block*bs*bs + c + r*bs];
//                 local_out += x_elem * A_elem;
//             }
//         }
// 
//         // do reduction in shared mem
//         tmp[lane] = local_out;
//         barrier(CLK_LOCAL_MEM_FENCE);
// 
//         for(unsigned int offset = 3; offset <= 24; offset <<= 1)
//         {
//             if (lane + offset < warpsize)
//             {
//                 tmp[lane] += tmp[lane + offset];
//             }
//             barrier(CLK_LOCAL_MEM_FENCE);
//         }
// 
//         if(lane < bs){
//             unsigned int row = target_block_row*bs + lane;
//             out[row] = rhs[row] - tmp[lane];
//         }
//         target_block_row += num_warps_in_grid;
//     }
// }
}

/// HIP kernel to apply the standard wellcontributions
__global__ void residual_k()
{
// KERNEL residual_k    
// res = rhs - mat * x
// algorithm based on:
// Optimization of Block Sparse Matrix-Vector Multiplication on Shared-MemoryParallel Architectures,
// Ryan Eberhardt, Mark Hoemmen, 2016, https://doi.org/10.1109/IPDPSW.2016.42
// __kernel void residual(
//     __global const double *vals,
//     __global const int *cols,
//     __global const int *rows,
//     const int N,
//     __global const double *x,
//     __global const double *rhs,
//     __global double *out,
//     __local double *tmp)
// {
//     const unsigned int bsize = get_local_size(0);
//     const unsigned int idx_b = get_global_id(0) / bsize;
//     const unsigned int idx_t = get_local_id(0);
//     const unsigned int num_workgroups = get_num_groups(0);
// 
//     int row = idx_b;
// 
//     while (row < N) {
//         int rowStart = rows[row];
//         int rowEnd = rows[row+1];
//         int rowLength = rowEnd - rowStart;
//         double local_sum = 0.0;
//         for (int j = rowStart + idx_t; j < rowEnd; j += bsize) {
//             int col = cols[j];
//             local_sum += vals[j] * x[col];
//         }
// 
//         tmp[idx_t] = local_sum;
//         barrier(CLK_LOCAL_MEM_FENCE);
// 
//         int offset = bsize / 2;
//         while(offset > 0) {
//             if (idx_t < offset) {
//                 tmp[idx_t] += tmp[idx_t + offset];
//             }
//             barrier(CLK_LOCAL_MEM_FENCE);
//             offset = offset / 2;
//         }
// 
//         if (idx_t == 0) {
//             out[row] = rhs[row] - tmp[idx_t];
//         }
// 
//         row += num_workgroups;
//     }
// }
}
#endif

void HipKernels::init(int verbosity_)
{
    if (initialized) {
        OpmLog::debug("Warning HipKernels is already initialized");
        return;
    }

//     queue = queue_;
//     verbosity = verbosity_;

    initialized = true;
} // end get_opencl_kernels()

void HipKernels::vmul(const double alpha, double* in1, double* in2, double* out, int N)
{
std::cout << "TODO: HipKernels::vmul(jacobi_damping, invDiags[level], t, x, Ncur);\n";
#ifdef __HIP__
    unsigned gridDim = num_std_wells;
    unsigned blockDim = 64;
    unsigned shared_mem_size = (blockDim + 2 * dim_wells) * sizeof(double); // shared memory for localSum, z1 and z2
    // dim3(N) will create a vector {N, 1, 1}
    vmul_k<<<dim3(gridDim), dim3(blockDim), shared_mem_size, stream>>>(
        d_Cnnzs_hip, d_Dnnzs_hip, d_Bnnzs_hip, d_Ccols_hip, d_Bcols_hip,
        d_x, d_y, dim, dim_wells, d_val_pointers_hip
    );
    HIP_CHECK(hipStreamSynchronize(stream));
#else
    OPM_THROW(std::logic_error, "Error prolongate_vector for rocsparse only supported when compiling with hipcc");
#endif
    
// //     const unsigned int work_group_size = 32;
// //     const unsigned int num_work_groups = ceilDivision(N, work_group_size);
// //     const unsigned int total_work_items = num_work_groups * work_group_size;
// //     Timer t_vmul;
// // 
// //     cl::Event event = (*vmul_k)(cl::EnqueueArgs(*queue, cl::NDRange(total_work_items), cl::NDRange(work_group_size)), alpha, in1, in2, out, N);
// // 
// //     if (verbosity >= 4) {
// //         event.wait();
// //         std::ostringstream oss;
// //         oss << std::scientific << "HipKernels vmul() time: " << t_vmul.stop() << " s";
// //         OpmLog::info(oss.str());
// //     }

}

void HipKernels::full_to_pressure_restriction(const double* fine_y, double* weights, double* coarse_y, int Nb)
{
    std::cout << " TODO: HipKernels::full_to_pressure_restriction(*rs, *weights, *coarse_y, Nb);\n";
// //     const unsigned int work_group_size = 32;
// //     const unsigned int num_work_groups = ceilDivision(Nb, work_group_size);
// //     const unsigned int total_work_items = num_work_groups * work_group_size;
// //     Timer t;
// // 
// //     cl::Event event = (*full_to_pressure_restriction_k)(cl::EnqueueArgs(*queue, cl::NDRange(total_work_items), cl::NDRange(work_group_size)), fine_y, weights, coarse_y, Nb);
// // 
// //     if (verbosity >= 4) {
// //         event.wait();
// //         std::ostringstream oss;
// //         oss << std::scientific << "HipKernels full_to_pressure_restriction() time: " << t.stop() << " s";
// //         OpmLog::info(oss.str());
// //     }
       
}

void HipKernels::add_coarse_pressure_correction(double* coarse_x, double* fine_x, int pressure_idx, int Nb)
{
std::cout << " TODO: HipKernels::add_coarse_pressure_correction(*d_coarse_x, x, pressure_idx, Nb);\n";
// //     const unsigned int work_group_size = 32;
// //     const unsigned int num_work_groups = ceilDivision(Nb, work_group_size);
// //     const unsigned int total_work_items = num_work_groups * work_group_size;
// //     Timer t;
// // 
// //     cl::Event event = (*add_coarse_pressure_correction_k)(cl::EnqueueArgs(*queue, cl::NDRange(total_work_items), cl::NDRange(work_group_size)), coarse_x, fine_x, pressure_idx, Nb);
// // 
// //     if (verbosity >= 4) {
// //         event.wait();
// //         std::ostringstream oss;
// //         oss << std::scientific << "HipKernels add_coarse_pressure_correction() time: " << t.stop() << " s";
// //         OpmLog::info(oss.str());
// //     }

}

void HipKernels::prolongate_vector(const double* in, double* out, const int* cols, int N)
{
    std::cout << "TO TEST: HipKernels::prolongate_vector(u, x, PcolIndices[level], Ncur);\n";
    
#ifdef __HIP__
    unsigned gridDim = num_std_wells;
    unsigned blockDim = 64;
    unsigned shared_mem_size = (blockDim + 2 * dim_wells) * sizeof(double); // shared memory for localSum, z1 and z2
    // dim3(N) will create a vector {N, 1, 1}
    prolongate_vector_k<<<dim3(gridDim), dim3(blockDim), shared_mem_size, stream>>>(
        d_Cnnzs_hip, d_Dnnzs_hip, d_Bnnzs_hip, d_Ccols_hip, d_Bcols_hip,
        d_x, d_y, dim, dim_wells, d_val_pointers_hip
    );
    HIP_CHECK(hipStreamSynchronize(stream));
#else
    OPM_THROW(std::logic_error, "Error prolongate_vector for rocsparse only supported when compiling with hipcc");
#endif


//     const unsigned int work_group_size = 32;
//     const unsigned int num_work_groups = ceilDivision(N, work_group_size);
//     const unsigned int total_work_items = num_work_groups * work_group_size;
//     Timer t;
// 
//     cl::Event event = (*prolongate_vector_k)(cl::EnqueueArgs(*queue, cl::NDRange(total_work_items), cl::NDRange(work_group_size)), in, out, cols, N);
// 
//     if (verbosity >= 4) {
//         event.wait();
//         std::ostringstream oss;
//         oss << std::scientific << "HipKernels prolongate_vector() time: " << t.stop() << " s";
//         OpmLog::info(oss.str());
//     }
}

void HipKernels::residual(double* vals, int* cols, int* rows, double* x, const double* rhs, double* out, int Nb, unsigned int block_size)
{
    std::cout << " TODO: HipKernels::residual(mat->nnzValues, mat->colIndices, mat->rowPointers, x, y, *rs, Nb, block_size);\n";
    
#ifdef __HIP__
    unsigned gridDim = num_std_wells;
    unsigned blockDim = 64;
    unsigned shared_mem_size = (blockDim + 2 * dim_wells) * sizeof(double); // shared memory for localSum, z1 and z2
    // dim3(N) will create a vector {N, 1, 1}
    residual_blocked_k<<<dim3(gridDim), dim3(blockDim), shared_mem_size, stream>>>(
        d_Cnnzs_hip, d_Dnnzs_hip, d_Bnnzs_hip, d_Ccols_hip, d_Bcols_hip,
        d_x, d_y, dim, dim_wells, d_val_pointers_hip
    );
    HIP_CHECK(hipStreamSynchronize(stream));
#else
    OPM_THROW(std::logic_error, "Error prolongate_vector for rocsparse only supported when compiling with hipcc");
#endif
    
    
//     const unsigned int work_group_size = 32;
//     const unsigned int num_work_groups = ceilDivision(Nb, work_group_size);
//     const unsigned int total_work_items = num_work_groups * work_group_size;
//     const unsigned int lmem_per_work_group = sizeof(double) * work_group_size;
//     Timer t_residual;
//     cl::Event event;
// 
//     if (block_size > 1) {
//         event = (*residual_blocked_k)(cl::EnqueueArgs(*queue, cl::NDRange(total_work_items), cl::NDRange(work_group_size)),
//                     vals, cols, rows, Nb, x, rhs, out, block_size, cl::Local(lmem_per_work_group));
//     } else {
//         event = (*residual_k)(cl::EnqueueArgs(*queue, cl::NDRange(total_work_items), cl::NDRange(work_group_size)),
//                     vals, cols, rows, Nb, x, rhs, out, cl::Local(lmem_per_work_group));
//     }
// 
//     if (verbosity >= 4) {
//         event.wait();
//         std::ostringstream oss;
//         oss << std::scientific << "HipKernels residual_blocked() time: " << t_residual.stop() << " s";
//         OpmLog::info(oss.str());
//     }
}

} // namespace Accelerator
} // namespace Opm
