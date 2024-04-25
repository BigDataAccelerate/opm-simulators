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
#include <opm/common/TimingMacros.hpp>
#include <opm/common/OpmLog/OpmLog.hpp>
#include <opm/common/ErrorMacros.hpp>
#include <dune/common/timer.hh>

#include <dune/common/shared_ptr.hh>

#include <opm/simulators/linalg/PreconditionerFactory.hpp>
#include <opm/simulators/linalg/PropertyTree.hpp>

#include <opm/simulators/linalg/bda/BdaBridge.hpp>
#include <opm/simulators/linalg/bda/BlockedMatrix.hpp>
#include <opm/simulators/linalg/bda/rocm/rocsparseCPR.hpp>
#include <opm/simulators/linalg/bda/rocm/hipKernels.hpp>

#include <opm/simulators/linalg/bda/Misc.hpp>//Razvan
// #include </home/rnane/Work/bigdataccelerate/src/opm-project/builds/dune/dune-2.8/dune-istl/dune/istl/paamg/matrixhierarchy.hh>

#include <hip/hip_runtime_api.h>
#include <hip/hip_version.h>

#define HIP_CHECK(STAT)                                  \
    do {                                                 \
        const hipError_t stat = (STAT);                  \
        if(stat != hipSuccess)                           \
        {                                                \
            std::ostringstream oss;                      \
            oss << "rocsparseSolverBackend::hip ";       \
            oss << "error: " << hipGetErrorString(stat); \
            OPM_THROW(std::logic_error, oss.str());      \
        }                                                \
    } while(0)

#define ROCSPARSE_CHECK(STAT)                            \
    do {                                                 \
        const rocsparse_status stat = (STAT);            \
        if(stat != rocsparse_status_success)             \
        {                                                \
            std::ostringstream oss;                      \
            oss << "rocsparseSolverBackend::rocsparse "; \
            oss << "error: " << stat;                    \
            OPM_THROW(std::logic_error, oss.str());      \
        }                                                \
    } while(0)

#define ROCBLAS_CHECK(STAT)                              \
    do {                                                 \
        const rocblas_status stat = (STAT);              \
        if(stat != rocblas_status_success)               \
        {                                                \
            std::ostringstream oss;                      \
            oss << "rocsparseSolverBackend::rocblas ";   \
            oss << "error: " << stat;                    \
            OPM_THROW(std::logic_error, oss.str());      \
        }                                                \
    } while(0)

namespace Opm
{
namespace Accelerator
{

using Opm::OpmLog;
using Dune::Timer;

template <unsigned int block_size>
rocsparseCPR<block_size>::rocsparseCPR(int verbosity_) :
    rocsparsePreconditioner<block_size>(verbosity_)
{
    bilu0 = std::make_unique<rocsparseBILU0<block_size> >(verbosity_);
}

template <unsigned int block_size>
bool rocsparseCPR<block_size>::initialize(std::shared_ptr<BlockedMatrix> matrix, std::shared_ptr<BlockedMatrix> jacMatrix, rocsparse_int *d_Arows, rocsparse_int *d_Acols) {
//openclCPR init_opencl_buffers() function ---> NOTE: implemented in the below function init_rocm_buffers!!!
//     d_Amatrices.reserve(this->num_levels);
//     d_Rmatrices.reserve(this->num_levels - 1);
//     d_invDiags.reserve(this->num_levels - 1);
//     for (Matrix& m : this->Amatrices) {
//         d_Amatrices.emplace_back(context.get(), m.N, m.M, m.nnzs, 1);
//     }
//     for (Matrix& m : this->Rmatrices) {
//         d_Rmatrices.emplace_back(context.get(), m.N, m.M, m.nnzs, 1);
//         d_f.emplace_back(*context, CL_MEM_READ_WRITE, sizeof(double) * m.N);
//         d_u.emplace_back(*context, CL_MEM_READ_WRITE, sizeof(double) * m.N);
// 
//         d_PcolIndices.emplace_back(*context, CL_MEM_READ_WRITE, sizeof(int) * m.M);
//         d_invDiags.emplace_back(*context, CL_MEM_READ_WRITE, sizeof(double) * m.M); // create a cl::Buffer
//         d_t.emplace_back(*context, CL_MEM_READ_WRITE, sizeof(double) * m.M);
//     }
//     d_weights = std::make_unique<cl::Buffer>(*context, CL_MEM_READ_WRITE, sizeof(double) * this->N);
//     d_rs = std::make_unique<cl::Buffer>(*context, CL_MEM_READ_WRITE, sizeof(double) * this->N);
//     d_mat = std::make_unique<OpenclMatrix>(context.get(), this->Nb, this->Nb, this->nnzb, block_size);
//     d_coarse_y = std::make_unique<cl::Buffer>(*context, CL_MEM_READ_WRITE, sizeof(double) * this->Nb);
//     d_coarse_x = std::make_unique<cl::Buffer>(*context, CL_MEM_READ_WRITE, sizeof(double) * this->Nb);
}

template <unsigned int block_size>
void rocsparseCPR<block_size>::copy_system_to_gpu(double *b) {

//openclCPR opencl_upload() code: --> NOTE: implemented in rocm_upload!
//     d_mat->upload(queue.get(), this->mat);
// 
//     err = CL_SUCCESS;
//     events.resize(2 * this->Rmatrices.size() + 1);
//     err |= queue->enqueueWriteBuffer(*d_weights, CL_FALSE, 0, sizeof(double) * this->N, this->weights.data(), nullptr, &events[0]);
//     for (unsigned int i = 0; i < this->Rmatrices.size(); ++i) {
//         d_Amatrices[i].upload(queue.get(), &this->Amatrices[i]);
// 
//         err |= queue->enqueueWriteBuffer(d_invDiags[i], CL_FALSE, 0, sizeof(double) * this->Amatrices[i].N, this->invDiags[i].data(), nullptr, &events[2*i+1]);
//         err |= queue->enqueueWriteBuffer(d_PcolIndices[i], CL_FALSE, 0, sizeof(int) * this->Amatrices[i].N, this->PcolIndices[i].data(), nullptr, &events[2*i+2]);
//     }
//     cl::WaitForEvents(events);
//     events.clear();
//     if (err != CL_SUCCESS) {
//         // enqueueWriteBuffer is C and does not throw exceptions like C++ OpenCL
//         OPM_THROW(std::logic_error, "openclCPR OpenCL enqueueWriteBuffer error");
//     }
//     for (unsigned int i = 0; i < this->Rmatrices.size(); ++i) {
//         d_Rmatrices[i].upload(queue.get(), &this->Rmatrices[i]);
//     }

}

template <unsigned int block_size>
void rocsparseCPR<block_size>::update_system_on_gpu(double *b) {
    
}

template <unsigned int block_size>
bool rocsparseCPR<block_size>::analyze_matrix(BlockedMatrix *mat_) {
std::cout << "-----in : rocsparseCPR<block_size>::analyze_matrix(mat_)\n";
    this->Nb = mat_->Nb;
    this->nnzb = mat_->nnzbs;
    this->N = Nb * block_size;
    this->nnz = nnzb * block_size * block_size;
    
    bool success = bilu0->analyze_matrix(mat_);
    
    this->mat = mat_;
std::cout << "-----out: rocsparseCPR<block_size>::analyze_matrix(mat_)\n";
    return success;
}

template <unsigned int block_size>
bool rocsparseCPR<block_size>::analyze_matrix(BlockedMatrix *mat_, BlockedMatrix *jacMat) {
std::cout << "-----in : rocsparseCPR<block_size>::analyze_matrix(mat_, jacMat)\n";
    this->Nb = mat_->Nb;
    this->nnzb = mat_->nnzbs;
    this->N = Nb * block_size;
    this->nnz = nnzb * block_size * block_size;

    bool success = bilu0->analyze_matrix(mat_, jacMat);
    this->mat = mat_;
std::cout << "-----out: rocsparseCPR<block_size>::analyze_matrix(mat_, jacMat)\n";

    return success;
}

template <unsigned int block_size>
bool rocsparseCPR<block_size>::create_preconditioner(BlockedMatrix *mat_, BlockedMatrix *jacMat) {
    Dune::Timer t_bilu0;
    bool result = bilu0->create_preconditioner(mat_, jacMat);
    if (verbosity >= 3) {
        std::ostringstream out;
        out << "rocsparseCPR create_preconditioner bilu0(): " << t_bilu0.stop() << " s";
        OpmLog::info(out.str());
    }

    Dune::Timer t_amg;
    this->create_preconditioner_amg(this->mat); // already points to bilu0::rmat if needed

    //init_data_transfers();--->TODO-Razvan: factor out below code to this method
    // initialize cMatrices and Buffers if needed
    auto init_func = std::bind(&rocsparseCPR::init_rocm_buffers, this);
    std::call_once(rocm_buffers_allocated, init_func);

    // upload matrices and vectors to GPU
    rocm_upload();

    if (verbosity >= 3) {
        std::ostringstream out;
        out << "rocsparseCPR create_preconditioner_amg(): " << t_amg.stop() << " s";
        OpmLog::info(out.str());
    }
    return result;
}

template <unsigned int block_size>
bool rocsparseCPR<block_size>::create_preconditioner(BlockedMatrix *mat_) {
    Dune::Timer t_bilu0;
    bool result = bilu0->create_preconditioner(mat_);
    if (verbosity >= 3) {
        std::ostringstream out;
        out << "rocsparseCPR create_preconditioner bilu0(): " << t_bilu0.stop() << " s";
        OpmLog::info(out.str());
    }

    Dune::Timer t_amg;
    this->create_preconditioner_amg(this->mat); // already points to bilu0::rmat if needed
    
    //init_data_transfers();--->TODO-Razvan: factor out below code to this method
    // initialize cMatrices and Buffers if needed
    auto init_func = std::bind(&rocsparseCPR::init_rocm_buffers, this);
    std::call_once(rocm_buffers_allocated, init_func);

    // upload matrices and vectors to GPU
    rocm_upload();
    
    if (verbosity >= 3) {
        std::ostringstream out;
        out << "rocsparseCPR create_preconditioner_amg(): " << t_amg.stop() << " s";
        OpmLog::info(out.str());
    }
    return result;
}

template <unsigned int block_size>
void rocsparseCPR<block_size>::init_rocm_buffers() {
    d_Amatrices.reserve(this->num_levels);
    d_Rmatrices.reserve(this->num_levels - 1);
    d_invDiags.reserve(this->num_levels - 1);
    for (Matrix& m : this->Amatrices) {
        d_Amatrices.emplace_back(m.N, m.M, m.nnzs, 1);
    }
    for (Matrix& m : this->Rmatrices) {
        d_Rmatrices.emplace_back(m.N, m.M, m.nnzs, 1);
        d_f.emplace_back(sizeof(double) * m.N);
        d_u.emplace_back(sizeof(double) * m.N);

        d_PcolIndices.emplace_back(sizeof(int) * m.M);
        d_invDiags.emplace_back(sizeof(double) * m.M); // create a cl::Buffer
        d_t.emplace_back(sizeof(double) * m.M);
    }
    HIP_CHECK(hipMalloc((void**)&d_weights, sizeof(double) * this->N));
    HIP_CHECK(hipMalloc((void**)&d_rs, sizeof(double) * this->N));
//     d_mat = std::make_unique<OpenclMatrix>(context.get(), this->Nb, this->Nb, this->nnzb, block_size);
//     HIP_CHECK(hipMalloc((void**)&d_mat, sizeof(double) * this->N));//TODO-R: define RocmMatrix!
    HIP_CHECK(hipMalloc((void**)&d_coarse_y, sizeof(double) * this->Nb));
    HIP_CHECK(hipMalloc((void**)&d_coarse_x, sizeof(double) * this->Nb));
}


template <unsigned int block_size>
void rocsparseCPR<block_size>::rocm_upload() {
//     d_mat->upload(queue.get(), this->mat);//TODO: define RocmMatrix and upload methods!
// 
//     err = CL_SUCCESS;
//     events.resize(2 * this->Rmatrices.size() + 1);
//     err |= queue->enqueueWriteBuffer(*d_weights, CL_FALSE, 0, sizeof(double) * this->N, this->weights.data(), nullptr, &events[0]);
    HIP_CHECK(hipMemcpyAsync(d_weights.data(), this->weights.data(), sizeof(double) * this->N, hipMemcpyHostToDevice, this->stream));
//     for (unsigned int i = 0; i < this->Rmatrices.size(); ++i) {
//         d_Amatrices[i].upload(queue.get(), &this->Amatrices[i]);//TODO: define RocmMatrix and upload methods!
// 
//         err |= queue->enqueueWriteBuffer(d_invDiags[i], CL_FALSE, 0, sizeof(double) * this->Amatrices[i].N, this->invDiags[i].data(), nullptr, &events[2*i+1]);
//         err |= queue->enqueueWriteBuffer(d_PcolIndices[i], CL_FALSE, 0, sizeof(int) * this->Amatrices[i].N, this->PcolIndices[i].data(), nullptr, &events[2*i+2]);
//     }
//     cl::WaitForEvents(events);
//     events.clear();
//     if (err != CL_SUCCESS) {
//         // enqueueWriteBuffer is C and does not throw exceptions like C++ OpenCL
//         OPM_THROW(std::logic_error, "openclCPR OpenCL enqueueWriteBuffer error");
//     }
//     for (unsigned int i = 0; i < this->Rmatrices.size(); ++i) {
//         d_Rmatrices[i].upload(queue.get(), &this->Rmatrices[i]);//TODO: define RocmMatrix and upload methods!
//     }
}

template <unsigned int block_size>
void rocsparseCPR<block_size>::amg_cycle_gpu(const int level, double &y, double &x) {
    cMatrix *A = &d_Amatrices[level];
    cMatrix *R = &d_Rmatrices[level];
    int Ncur = A->Nb;
    double zero = 0.0;
    double one = 1.0;
    
    rocsparse_mat_info spmv_info;
    rocsparse_mat_descr descr_A;
    ROCSPARSE_CHECK(rocsparse_create_mat_info(&spmv_info));
    ROCSPARSE_CHECK(rocsparse_create_mat_descr(&descr_A));

    if (level == this->num_levels - 1) {
        // solve coarsest level
        std::vector<double> h_y(Ncur), h_x(Ncur, 0);

        HIP_CHECK(hipMemcpyAsync(h_y.data(), &y, sizeof(double) * Ncur, hipMemcpyDeviceToHost, this->stream));
        
        // solve coarsest level using umfpack
        this->umfpack.apply(h_x.data(), h_y.data());

        HIP_CHECK(hipMemcpyAsync(&x, h_x.data(), sizeof(double) * Ncur, hipMemcpyHostToDevice, this->stream));
        
        return;
    }
    int Nnext = d_Amatrices[level+1].Nb;

    double& t = d_t[level];
    double& f = d_f[level];
    double& u = d_u[level]; // u was 0-initialized earlier

    // presmooth
    double jacobi_damping = 0.65; // default value in amgcl: 0.72
    for (unsigned i = 0; i < this->num_pre_smooth_steps; ++i){
        HipKernels::residual(A->nnzValues.data(), A->colIndices.data(), A->rowPointers.data(), &x, &y, &t, Ncur, 1);
        HipKernels::vmul(jacobi_damping, &(d_invDiags.data()[level]), &t, &x, Ncur);
    }

    // move to coarser level
    HipKernels::residual(A->nnzValues.data(), A->colIndices.data(), A->rowPointers.data(), &x, &y, &t, Ncur, 1);
    
    std::cout << "TOCHECK: call rocsparse::spmv(R->nnzValues, R->colIndices, R->rowPointers, t, f, Nnext, 1, true);\n";
// #if HIP_VERSION >= 50400000
    ROCSPARSE_CHECK(rocsparse_dbsrmv_ex(this->handle, this->dir, this->operation,
                                        Nnext, Nnext, Nnext, &one, descr_A,
                                        R->nnzValues.data(), R->rowPointers.data(), R->colIndices.data(), 1,
                                        spmv_info, &x, &zero, &y));
// #else
//     ROCSPARSE_CHECK(rocsparse_dbsrmv(handle, dir, operation,
//                                         Nb, Nb, nnzb, &one, descr_A,
//                                         d_Avals, d_Arows, d_Acols, block_size,
//                                         d_x, &zero, d_r));
// #endif
    
    amg_cycle_gpu(level + 1, f, u);
    HipKernels::prolongate_vector(&u, &x, &(d_PcolIndices.data()[level]), Ncur);

    // postsmooth
    for (unsigned i = 0; i < this->num_post_smooth_steps; ++i){
        HipKernels::residual(A->nnzValues.data(), A->colIndices.data(), A->rowPointers.data(), &x, &y, &t, Ncur, 1);
        HipKernels::vmul(jacobi_damping, &(d_invDiags.data()[level]), &t, &x, Ncur);
    }
}


// x = prec(y)
template <unsigned int block_size>
void rocsparseCPR<block_size>::apply_amg(const double& y, double& x) {
    // initialize u and x vectors
    HIP_CHECK(hipMemsetAsync(d_coarse_x.data(), 0, sizeof(double) * this->Nb, this->stream));

    for (unsigned int i = 0; i < d_u.size(); ++i) {
        HIP_CHECK(hipMemsetAsync(&d_u[i], 0, sizeof(double) * this->Rmatrices[i].N, this->stream));
    }

    HipKernels::residual(d_mat->nnzValues.data(), d_mat->colIndices.data(), d_mat->rowPointers.data(), &x, &y, d_rs.data(), this->Nb, block_size);
    HipKernels::full_to_pressure_restriction(d_rs.data(), d_weights.data(), d_coarse_y.data(), Nb);

    amg_cycle_gpu(0, *(d_coarse_y.data()), *(d_coarse_x.data()));//TODO-Razvan: continue to implement this metod and all memory transfers!!!

    HipKernels::add_coarse_pressure_correction(d_coarse_x.data(), &x, this->pressure_idx, Nb);
}

template <unsigned int block_size>
void rocsparseCPR<block_size>::apply(double& y, double& x) {
    Dune::Timer t_bilu0;
    bilu0->apply(y, x);
    if (verbosity >= 4) {
        std::ostringstream out;
        out << "rocsparseCPR apply bilu0(): " << t_bilu0.stop() << " s";
        OpmLog::info(out.str());
    }

    Dune::Timer t_amg;
    apply_amg(y, x);
    if (verbosity >= 4) {
        std::ostringstream out;
        out << "rocsparseCPR apply amg(): " << t_amg.stop() << " s";
        OpmLog::info(out.str());
    }
}


#define INSTANTIATE_BDA_FUNCTIONS(n)  \
template class rocsparseCPR<n>;

INSTANTIATE_BDA_FUNCTIONS(1);
INSTANTIATE_BDA_FUNCTIONS(2);
INSTANTIATE_BDA_FUNCTIONS(3);
INSTANTIATE_BDA_FUNCTIONS(4);
INSTANTIATE_BDA_FUNCTIONS(5);
INSTANTIATE_BDA_FUNCTIONS(6);

#undef INSTANTIATE_BDA_FUNCTIONS
// 
} // namespace Accelerator
} // namespace Opm


