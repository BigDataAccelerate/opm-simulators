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
#include <opm/simulators/linalg/bda/opencl/openclCPR.hpp>
#include <opm/simulators/linalg/bda/opencl/OpenclMatrix.hpp>
#include <opm/simulators/linalg/bda/opencl/openclKernels.hpp>

#include <opm/simulators/linalg/bda/Misc.hpp>
// #include </home/rnane/Work/bigdataccelerate/src/opm-project/builds/dune/dune-2.8/dune-istl/dune/istl/paamg/matrixhierarchy.hh>

namespace Opm
{
namespace Accelerator
{

using Opm::OpmLog;
using Dune::Timer;

template <unsigned int block_size>
openclCPR<block_size>::openclCPR(int verbosity_, bool opencl_ilu_parallel_) :
    openclPreconditioner<block_size>(verbosity_)
{
    opencl_ilu_parallel = opencl_ilu_parallel_;
    bilu0 = std::make_unique<openclBILU0<block_size> >(opencl_ilu_parallel, verbosity_);
}


template <unsigned int block_size>
void openclCPR<block_size>::setOpencl(std::shared_ptr<cl::Context>& context_, std::shared_ptr<cl::CommandQueue>& queue_) {

    context = context_;
    queue = queue_;

    bilu0->setOpencl(context, queue);
}


template <unsigned int block_size>
bool openclCPR<block_size>::analyze_matrix(BlockedMatrix *mat_) 
{
// std::cout << "-----in : openclCPR<block_size>::analyze_matrix(BlockedMatrix *mat_) \n";
    this->Nb = mat_->Nb;
    this->nnzb = mat_->nnzbs;
    this->N = this->Nb * block_size;
    this->nnz = this->nnzb * block_size * block_size;

    bool success = bilu0->analyze_matrix(mat_);
// std::cout << " after bilu0->analyze_matrix(mat_); in openclCPR::analyze_matrix(..) -- 1 \n";
    this->mat = mat_;
    return success;
}

template <unsigned int block_size>
bool openclCPR<block_size>::analyze_matrix(BlockedMatrix *mat_, BlockedMatrix *jacMat) 
{
// std::cout << "-----in : openclCPR<block_size>::analyze_matrix(BlockedMatrix *mat_, BlockedMatrix *jacMat) \n";
    this->Nb = mat_->Nb;
    this->nnzb = mat_->nnzbs;
    this->N = this->Nb * block_size;
    this->nnz = this->nnzb * block_size * block_size;

    bool success = bilu0->analyze_matrix(mat_, jacMat);
// std::cout << " after bilu0->analyze_matrix(mat_); in openclCPR::analyze_matrix(..) -- 2 \n";
    this->mat = mat_;

    return success;
}

template <unsigned int block_size>
bool openclCPR<block_size>::create_preconditioner(BlockedMatrix *mat_, BlockedMatrix *jacMat) {
    Dune::Timer t_bilu0;
    bool result = bilu0->create_preconditioner(mat_, jacMat);
    if (verbosity >= 3) {
        std::ostringstream out;
        out << "openclCPR create_preconditioner bilu0(): " << t_bilu0.stop() << " s";
        OpmLog::info(out.str());
    }

    Dune::Timer t_amg;
    this->create_preconditioner_amg(this->mat); // already points to bilu0::rmat if needed
    
    // initialize OpenclMatrices and Buffers if needed
    auto init_func = std::bind(&openclCPR::init_opencl_buffers, this);
    std::call_once(opencl_buffers_allocated, init_func);

    // upload matrices and vectors to GPU
    opencl_upload();

    if (verbosity >= 3) {
        std::ostringstream out;
        out << "openclCPR create_preconditioner_amg(): " << t_amg.stop() << " s";
        OpmLog::info(out.str());
    }
    std::cout << "out: openclCPR<block_size>::create_preconditioner(BlockedMatrix *mat_, BlockedMatrix *jacMat) .... exiting\n";exit(0);//Razvan
    return result;
}

template <unsigned int block_size>
bool openclCPR<block_size>::create_preconditioner(BlockedMatrix *mat_) {
    Dune::Timer t_bilu0;
    bool result = bilu0->create_preconditioner(mat_);
    if (verbosity >= 3) {
        std::ostringstream out;
        out << "openclCPR create_preconditioner bilu0(): " << t_bilu0.stop() << " s";
        OpmLog::info(out.str());
    }

    Dune::Timer t_amg;
    this->create_preconditioner_amg(this->mat); // already points to bilu0::rmat if needed
    
    // initialize OpenclMatrices and Buffers if needed
    auto init_func = std::bind(&openclCPR::init_opencl_buffers, this);
    std::call_once(opencl_buffers_allocated, init_func);

    // upload matrices and vectors to GPU
    opencl_upload();
    
    if (verbosity >= 3) {
        std::ostringstream out;
        out << "openclCPR create_preconditioner_amg(): " << t_amg.stop() << " s";
        OpmLog::info(out.str());
    }
    return result;
}

template <unsigned int block_size>
void openclCPR<block_size>::init_opencl_buffers() {
    d_Amatrices.reserve(this->num_levels);
    d_Rmatrices.reserve(this->num_levels - 1);
    d_invDiags.reserve(this->num_levels - 1);
    for (Matrix& m : this->Amatrices) {
        d_Amatrices.emplace_back(context.get(), m.N, m.M, m.nnzs, 1);
    }
    for (Matrix& m : this->Rmatrices) {
        d_Rmatrices.emplace_back(context.get(), m.N, m.M, m.nnzs, 1);
        d_f.emplace_back(*context, CL_MEM_READ_WRITE, sizeof(double) * m.N);
        d_u.emplace_back(*context, CL_MEM_READ_WRITE, sizeof(double) * m.N);

        d_PcolIndices.emplace_back(*context, CL_MEM_READ_WRITE, sizeof(int) * m.M);
        d_invDiags.emplace_back(*context, CL_MEM_READ_WRITE, sizeof(double) * m.M); // create a cl::Buffer
        d_t.emplace_back(*context, CL_MEM_READ_WRITE, sizeof(double) * m.M);
    }
    d_weights = std::make_unique<cl::Buffer>(*context, CL_MEM_READ_WRITE, sizeof(double) * this->N);
    d_rs = std::make_unique<cl::Buffer>(*context, CL_MEM_READ_WRITE, sizeof(double) * this->N);
    d_mat = std::make_unique<OpenclMatrix>(context.get(), this->Nb, this->Nb, this->nnzb, block_size);
    d_coarse_y = std::make_unique<cl::Buffer>(*context, CL_MEM_READ_WRITE, sizeof(double) * this->Nb);
    d_coarse_x = std::make_unique<cl::Buffer>(*context, CL_MEM_READ_WRITE, sizeof(double) * this->Nb);
}


template <unsigned int block_size>
void openclCPR<block_size>::opencl_upload() {
    d_mat->upload(queue.get(), this->mat);

    err = CL_SUCCESS;
    events.resize(2 * this->Rmatrices.size() + 1);
    err |= queue->enqueueWriteBuffer(*d_weights, CL_FALSE, 0, sizeof(double) * this->N, this->weights.data(), nullptr, &events[0]);
    for (unsigned int i = 0; i < this->Rmatrices.size(); ++i) {
        d_Amatrices[i].upload(queue.get(), &this->Amatrices[i]);

        err |= queue->enqueueWriteBuffer(d_invDiags[i], CL_FALSE, 0, sizeof(double) * this->Amatrices[i].N, this->invDiags[i].data(), nullptr, &events[2*i+1]);
        err |= queue->enqueueWriteBuffer(d_PcolIndices[i], CL_FALSE, 0, sizeof(int) * this->Amatrices[i].N, this->PcolIndices[i].data(), nullptr, &events[2*i+2]);
    }
    cl::WaitForEvents(events);
    events.clear();
    if (err != CL_SUCCESS) {
        // enqueueWriteBuffer is C and does not throw exceptions like C++ OpenCL
        OPM_THROW(std::logic_error, "openclCPR OpenCL enqueueWriteBuffer error");
    }
    for (unsigned int i = 0; i < this->Rmatrices.size(); ++i) {
        d_Rmatrices[i].upload(queue.get(), &this->Rmatrices[i]);
    }
}

template <unsigned int block_size>
void openclCPR<block_size>::amg_cycle_gpu(const int level, cl::Buffer &y, cl::Buffer &x) {
    OpenclMatrix *A = &d_Amatrices[level];
    OpenclMatrix *R = &d_Rmatrices[level];
    int Ncur = A->Nb;

    if (level == this->num_levels - 1) {
        // solve coarsest level
        std::vector<double> h_y(Ncur), h_x(Ncur, 0);

        events.resize(1);
        err = queue->enqueueReadBuffer(y, CL_FALSE, 0, sizeof(double) * Ncur, h_y.data(), nullptr, &events[0]);
        cl::WaitForEvents(events);
        events.clear();
        if (err != CL_SUCCESS) {
            // enqueueWriteBuffer is C and does not throw exceptions like C++ OpenCL
            OPM_THROW(std::logic_error, "openclCPR OpenCL enqueueReadBuffer error");
        }

        // solve coarsest level using umfpack
        this->umfpack.apply(h_x.data(), h_y.data());

        events.resize(1);
        err = queue->enqueueWriteBuffer(x, CL_FALSE, 0, sizeof(double) * Ncur, h_x.data(), nullptr, &events[0]);
        cl::WaitForEvents(events);
        events.clear();
        if (err != CL_SUCCESS) {
            // enqueueWriteBuffer is C and does not throw exceptions like C++ OpenCL
            OPM_THROW(std::logic_error, "openclCPR OpenCL enqueueWriteBuffer error");
        }
        return;
    }
    int Nnext = d_Amatrices[level+1].Nb;

    cl::Buffer& t = d_t[level];
    cl::Buffer& f = d_f[level];
    cl::Buffer& u = d_u[level]; // u was 0-initialized earlier

    // presmooth
    double jacobi_damping = 0.65; // default value in amgcl: 0.72
    for (unsigned i = 0; i < this->num_pre_smooth_steps; ++i){
        OpenclKernels::residual(A->nnzValues, A->colIndices, A->rowPointers, x, y, t, Ncur, 1);
        OpenclKernels::vmul(jacobi_damping, d_invDiags[level], t, x, Ncur);
    }

    // move to coarser level
    OpenclKernels::residual(A->nnzValues, A->colIndices, A->rowPointers, x, y, t, Ncur, 1);
    OpenclKernels::spmv(R->nnzValues, R->colIndices, R->rowPointers, t, f, Nnext, 1, true);
    amg_cycle_gpu(level + 1, f, u);
    OpenclKernels::prolongate_vector(u, x, d_PcolIndices[level], Ncur);

    // postsmooth
    for (unsigned i = 0; i < this->num_post_smooth_steps; ++i){
        OpenclKernels::residual(A->nnzValues, A->colIndices, A->rowPointers, x, y, t, Ncur, 1);
        OpenclKernels::vmul(jacobi_damping, d_invDiags[level], t, x, Ncur);
    }
}


// x = prec(y)
template <unsigned int block_size>
void openclCPR<block_size>::apply_amg(const cl::Buffer& y, cl::Buffer& x) {
    // 0-initialize u and x vectors
    events.resize(d_u.size() + 1);
    err = queue->enqueueFillBuffer(*d_coarse_x, 0, 0, sizeof(double) * this->Nb, nullptr, &events[0]);

    for (unsigned int i = 0; i < d_u.size(); ++i) {
        err |= queue->enqueueFillBuffer(d_u[i], 0, 0, sizeof(double) * this->Rmatrices[i].N, nullptr, &events[i + 1]);
    }
    cl::WaitForEvents(events);
    events.clear();
    if (err != CL_SUCCESS) {
        // enqueueWriteBuffer is C and does not throw exceptions like C++ OpenCL
        OPM_THROW(std::logic_error, "CPR OpenCL enqueueWriteBuffer error");
    }
    
    OpenclKernels::residual(d_mat->nnzValues, d_mat->colIndices, d_mat->rowPointers, x, y, *d_rs, this->Nb, block_size);
    OpenclKernels::full_to_pressure_restriction(*d_rs, *d_weights, *d_coarse_y, this->Nb);
    
    amg_cycle_gpu(0, *d_coarse_y, *d_coarse_x);

    OpenclKernels::add_coarse_pressure_correction(*d_coarse_x, x, this->pressure_idx, this->Nb);
}

template <unsigned int block_size>
void openclCPR<block_size>::apply(const cl::Buffer& y, cl::Buffer& x) {
    Dune::Timer t_bilu0;
    bilu0->apply(y, x);
    if (verbosity >= 4) {
        std::ostringstream out;
        out << "openclCPR apply bilu0(): " << t_bilu0.stop() << " s";
        OpmLog::info(out.str());
    }
    
    Dune::Timer t_amg;
    apply_amg(y, x);
    if (verbosity >= 4) {
        std::ostringstream out;
        out << "openclCPR apply amg(): " << t_amg.stop() << " s";
        OpmLog::info(out.str());
    }
}


#define INSTANTIATE_BDA_FUNCTIONS(n)  \
template class openclCPR<n>;

INSTANTIATE_BDA_FUNCTIONS(1);
INSTANTIATE_BDA_FUNCTIONS(2);
INSTANTIATE_BDA_FUNCTIONS(3);
INSTANTIATE_BDA_FUNCTIONS(4);
INSTANTIATE_BDA_FUNCTIONS(5);
INSTANTIATE_BDA_FUNCTIONS(6);

#undef INSTANTIATE_BDA_FUNCTIONS

} // namespace Accelerator
} // namespace Opm


