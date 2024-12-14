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
#include <opm/common/TimingMacros.hpp>
#include <opm/common/OpmLog/OpmLog.hpp>
#include <opm/common/ErrorMacros.hpp>
#include <dune/common/timer.hh>

#include <dune/common/shared_ptr.hh>

#include <opm/simulators/linalg/PreconditionerFactory.hpp>
#include <opm/simulators/linalg/PropertyTree.hpp>

#include <opm/simulators/linalg/gpubridge/GpuBridge.hpp>
#include <opm/simulators/linalg/gpubridge/BlockedMatrix.hpp>
#include <opm/simulators/linalg/gpubridge/rocm/rocsparseCPR.hpp>
#include <opm/simulators/linalg/gpubridge/rocm/hipKernels.hpp>

#include <opm/simulators/linalg/gpubridge/Misc.hpp>

#include <type_traits>

namespace Opm::Accelerator {

using Opm::OpmLog;
using Dune::Timer;

template <class Scalar, unsigned int block_size>
rocsparseCPR<Scalar, block_size>::rocsparseCPR(int verbosity_) :
    rocsparsePreconditioner<Scalar, block_size>(verbosity_)
{
    bilu0 = std::make_unique<rocsparseBILU0<Scalar, block_size> >(verbosity_);
}

template <class Scalar, unsigned int block_size>
bool rocsparseCPR<Scalar, block_size>::
initialize(std::shared_ptr<BlockedMatrix<Scalar>> matrix,
           std::shared_ptr<BlockedMatrix<Scalar>> jacMatrix,
           rocsparse_int *d_Arows,
           rocsparse_int *d_Acols)
{
    this->Nb = matrix->Nb;
    this->nnzb = matrix->nnzbs;
    this->N = Nb * block_size;
    this->nnz = nnzb * block_size * block_size;

    bilu0->set_context(this->handle, this->dir, this->operation, this->stream);
    bilu0->setJacMat(*this->jacMat.get());
    bilu0->initialize(matrix,jacMatrix,d_Arows,d_Acols);
    return true;
}

template <class Scalar, unsigned int block_size>
void rocsparseCPR<Scalar, block_size>::
copy_system_to_gpu(Scalar *b) {
    bilu0->copy_system_to_gpu(b);
}

template <class Scalar, unsigned int block_size>
void rocsparseCPR<Scalar, block_size>::
update_system_on_gpu(Scalar *vals) {
    bilu0->update_system_on_gpu(vals);
}

//TODO-RN: can be merged with the other method below
template <class Scalar, unsigned int block_size>
bool rocsparseCPR<Scalar, block_size>::
analyze_matrix(BlockedMatrix<Scalar> *mat_) {
    this->Nb = mat_->Nb;
    this->nnzb = mat_->nnzbs;
    this->N = Nb * block_size;
    this->nnz = nnzb * block_size * block_size;
    
    bool success = bilu0->analyze_matrix(mat_);
    
    this->mat = mat_;  
    
    return success;
}

template <class Scalar, unsigned int block_size>
bool rocsparseCPR<Scalar, block_size>::
analyze_matrix(BlockedMatrix<Scalar> *mat_,
               BlockedMatrix<Scalar> *jacMat_)
{
    this->Nb = mat_->Nb;
    this->nnzb = mat_->nnzbs;
    this->N = Nb * block_size;
    this->nnz = nnzb * block_size * block_size;

    bool success = bilu0->analyze_matrix(mat_, jacMat_);
    this->mat = mat_;

    return success;
}

template <class Scalar, unsigned int block_size>
bool rocsparseCPR<Scalar, block_size>::
create_preconditioner(BlockedMatrix<Scalar> *mat_,
                      BlockedMatrix<Scalar> *jacMat_)
{
    Dune::Timer t_bilu0;
    bool result = bilu0->create_preconditioner(mat_, jacMat_);
    if (verbosity >= 3) {
        c_decomp_ilu += t_bilu0.stop(); 
        std::ostringstream out;
        out << "-----rocsparseCPR cum create_preconditioner bilu0(): " << c_decomp_ilu << " s (+" << t_bilu0.elapsed() <<" s)";
        OpmLog::info(out.str());
    }
    
    Dune::Timer t_amg;
    this->create_preconditioner_amg(this->mat);

    if (verbosity >= 3) {
        c_decomp_amg += t_amg.stop();
        std::ostringstream out;
        out << "-----rocsparseCPR cum create_preconditioner_amg():   " << c_decomp_amg << " s (+" << t_amg.elapsed() <<" s)";
        OpmLog::info(out.str());
    }
    
    auto init_func = std::bind(&rocsparseCPR::init_rocm_buffers, this);
    std::call_once(rocm_buffers_allocated, init_func);

    Dune::Timer t_copy_cpr;
    // upload matrices and vectors to GPU
    rocm_upload();
    if (verbosity >= 3) {
        c_copy_cpr += t_copy_cpr.stop();
        std::ostringstream out;
        out << "-----rocsparseCPR cum rocm_upload():               " << c_copy_cpr << " s (+" << t_copy_cpr.elapsed() <<" s)";
        OpmLog::info(out.str());
    }

    return result;
}

template <class Scalar, unsigned int block_size>
bool rocsparseCPR<Scalar, block_size>::
create_preconditioner(BlockedMatrix<Scalar> *mat_) {
    Dune::Timer t_bilu0;
    bool result = bilu0->create_preconditioner(mat_);
    if (verbosity >= 3) {
        c_decomp_ilu += t_bilu0.stop(); 
        std::ostringstream out;
        out << "-----rocsparseCPR cum create_preconditioner bilu0(): " << c_decomp_ilu << " s (+" << t_bilu0.elapsed() <<" s)";
        OpmLog::info(out.str());
    }

    Dune::Timer t_amg;
    this->create_preconditioner_amg(this->mat); 
    
    if (verbosity >= 3) {
        c_decomp_amg += t_amg.stop();
        std::ostringstream out;
        out << "-----rocsparseCPR cum create_preconditioner_amg():   " << c_decomp_amg << " s (+" << t_amg.elapsed() <<" s)";
        OpmLog::info(out.str());
    }
    
    auto init_func = std::bind(&rocsparseCPR::init_rocm_buffers, this);
    std::call_once(rocm_buffers_allocated, init_func);

    Dune::Timer t_copy_cpr;
    // upload matrices and vectors to GPU
    rocm_upload();
    if (verbosity >= 3) {
        c_copy_cpr += t_copy_cpr.stop();
        std::ostringstream out;
        out << "-----rocsparseCPR cum rocm_upload():               " << c_copy_cpr << " s (+" << t_copy_cpr.elapsed() <<" s)";
        OpmLog::info(out.str());
    }
    
    return result;
}

template <class Scalar, unsigned int block_size>
void rocsparseCPR<Scalar, block_size>::
init_rocm_buffers() {
    d_Amatrices.reserve(this->num_levels);
    d_Rmatrices.reserve(this->num_levels - 1);
    d_invDiags.reserve(this->num_levels - 1);
    for (Matrix<Scalar>& m : this->Amatrices) {
        d_Amatrices.emplace_back(m.N, m.M, m.nnzs, 1);
    }
    
    for (Matrix<Scalar>& m : this->Rmatrices) {
        d_Rmatrices.emplace_back(m.N, m.M, m.nnzs, 1);
        d_f.emplace_back(m.N);
        d_u.emplace_back(m.N);
        d_PcolIndices.emplace_back(m.M);
        d_invDiags.emplace_back(m.M);
        d_t.emplace_back(m.M);
    }
    
    d_weights.emplace_back(this->N);
    d_rs.emplace_back(this->N);
    d_mat = std::make_unique<RocmMatrix<Scalar>>(this->Nb, this->Nb, this->nnzb, block_size);
    
    d_coarse_y.emplace_back(this->Nb);
    d_coarse_x.emplace_back(this->Nb);
}

template <class Scalar, unsigned int block_size>
void rocsparseCPR<Scalar, block_size>::
rocm_upload() {
     d_mat->upload(this->mat, this->stream);
     
     HIP_CHECK(hipMemcpyAsync(d_weights.data()->nnzValues, this->weights.data(), sizeof(Scalar) * this->N, hipMemcpyHostToDevice, this->stream));
    
    for (unsigned int i = 0; i < this->Rmatrices.size(); ++i) {
        d_Amatrices[i].upload(&this->Amatrices[i], this->stream);
        
        HIP_CHECK(hipMemcpyAsync(d_invDiags[i].nnzValues, this->invDiags[i].data(), sizeof(Scalar) * this->Amatrices[i].N, hipMemcpyHostToDevice, this->stream));
        HIP_CHECK(hipMemcpyAsync(d_PcolIndices[i].nnzValues, this->PcolIndices[i].data(), sizeof(int) * this->Amatrices[i].N, hipMemcpyHostToDevice, this->stream));
    }
    
    for (unsigned int i = 0; i < this->Rmatrices.size(); ++i) {
        d_Rmatrices[i].upload(&this->Rmatrices[i], this->stream);
    }
}

template <class Scalar, unsigned int block_size>
void rocsparseCPR<Scalar, block_size>::
amg_cycle_gpu(const int level,
              Scalar &y,
              Scalar &x)
{
    RocmMatrix<Scalar> *A = &d_Amatrices[level];
    RocmMatrix<Scalar> *R = &d_Rmatrices[level];
    int Ncur = A->Nb;
    
    rocsparse_mat_info spmv_info;
    rocsparse_mat_descr descr_R;
    ROCSPARSE_CHECK(rocsparse_create_mat_info(&spmv_info));
    ROCSPARSE_CHECK(rocsparse_create_mat_descr(&descr_R));

    if (level == this->num_levels - 1) {
        Timer t_amg_copy1;
        // solve coarsest level
        std::vector<Scalar> h_y(Ncur), h_x(Ncur, 0);

        HIP_CHECK(hipMemcpyAsync(h_y.data(), &y, sizeof(Scalar) * Ncur, hipMemcpyDeviceToHost, this->stream));
        
        if (verbosity >= 3) {
            HIP_CHECK(hipStreamSynchronize(this->stream));
            c_amg_coursecopy += t_amg_copy1.stop();
            if(verbosity >= 4){
                std::ostringstream out;
                out << "---------------DH copy: " << t_amg_copy1.elapsed() << " s";
                OpmLog::info(out.str());
            }
        }
        
        Timer t_amg_compute;
        // The if constexpr is needed to make the code compile
        // since the umfpack member is an 'int' with float Scalar.
        // We will never get here with float Scalar as we throw earlier.
        // Solve coarsest level using umfpack
        if constexpr (std::is_same_v<Scalar,double>) {
            this->umfpack.apply(h_x.data(), h_y.data());
        }
    
        if (verbosity >= 3) {
            c_amg_coursecompute += t_amg_compute.stop();
            if(verbosity >= 4){
                std::ostringstream out;
                out << "---------------umf compute: " << t_amg_compute.elapsed() << " s";
                OpmLog::info(out.str());
            }
        }
        
        Timer t_amg_copy2;
        HIP_CHECK(hipMemcpyAsync(&x, h_x.data(), sizeof(Scalar) * Ncur, hipMemcpyHostToDevice, this->stream));
        
        if (verbosity >= 3) {
            HIP_CHECK(hipStreamSynchronize(this->stream));
            c_amg_coursecopy += t_amg_copy2.stop();
            if(verbosity >= 4){
                std::ostringstream out;
                out << "---------------HD copy: " << t_amg_copy2.elapsed() << " s";
                OpmLog::info(out.str());
            }
        }
        
        return;
    }
    
    int Nnext = d_Amatrices[level+1].Nb;

    RocmVector<Scalar>& t = d_t[level];
    RocmVector<Scalar>& f = d_f[level];
    RocmVector<Scalar>& u = d_u[level]; // u was 0-initialized earlier

    Timer t_presmooth;
    // presmooth
    Scalar jacobi_damping = 0.65; // default value in amgcl: 0.72
    for (unsigned i = 0; i < this->num_pre_smooth_steps; ++i){
        HipKernels<Scalar>::residual(A->nnzValues, A->colIndices, A->rowPointers, &x, &y, t.nnzValues, Ncur, 1, this->stream);
        HipKernels<Scalar>::vmul(jacobi_damping, d_invDiags[level].nnzValues, t.nnzValues, &x, Ncur, this->stream);
    }
    if (verbosity >= 3) {
        HIP_CHECK(hipStreamSynchronize(this->stream));
        c_amg_presmooth += t_presmooth.stop();
        if(verbosity >= 4){
            std::ostringstream out;
            out << "---------------presmooth time: " << t_presmooth.elapsed() << " s";
            OpmLog::info(out.str());
        }
     }
    
    Timer t_residual;
    // move to coarser level
    HipKernels<Scalar>::residual(A->nnzValues, A->colIndices, A->rowPointers, &x, &y, t.nnzValues, Ncur, 1, this->stream);

    if (verbosity >= 3) {
        HIP_CHECK(hipStreamSynchronize(this->stream));
        c_amg_residual += t_residual.stop();
        if(verbosity >= 4){
            std::ostringstream out;
            out << "---------------residual time1: " << t_residual.elapsed() << " s";
            OpmLog::info(out.str());
        }
     }

     Timer t_spmv;
// TODO: understand why rocsparse spmv library function does not here. 
//     ROCSPARSE_CHECK(rocsparse_dbsrmv(this->handle, this->dir, this->operation,
//                                          R->Nb, R->Mb, R->nnzbs, &one, descr_R,
//                                          R->nnzValues, R->rowPointers, R->colIndices, 1,
//                                          t.nnzValues, &zero, f.nnzValues));
    HipKernels<Scalar>::spmv(R->nnzValues, R->colIndices, R->rowPointers, t.nnzValues, f.nnzValues, Nnext, 1, this->stream);

    if (verbosity >= 3) {
        HIP_CHECK(hipStreamSynchronize(this->stream));
        c_amg_spmv += t_spmv.stop();
        if(verbosity >= 4){
            std::ostringstream out;
            out << "---------------spmv time: " << t_spmv.elapsed() << " s";
            OpmLog::info(out.str());
        }
     }
     
    amg_cycle_gpu(level + 1, *f.nnzValues, *u.nnzValues);
    
    Timer t_prolongate;
    HipKernels<Scalar>::prolongate_vector(u.nnzValues, &x, d_PcolIndices[level].nnzValues, Ncur, this->stream);
    if (verbosity >= 3) {
        HIP_CHECK(hipStreamSynchronize(this->stream));
        c_amg_prolongate += t_prolongate.stop();
        if(verbosity >= 4){
            std::ostringstream out;
            out << "---------------prolongate time: " << t_prolongate.elapsed() << " s";
            OpmLog::info(out.str());
        }
     }

    Timer t_postsmooth;
    // postsmooth
    for (unsigned i = 0; i < this->num_post_smooth_steps; ++i){
        HipKernels<Scalar>::residual(A->nnzValues, A->colIndices, A->rowPointers, &x, &y, t.nnzValues, Ncur, 1, this->stream);
        HipKernels<Scalar>::vmul(jacobi_damping, d_invDiags[level].nnzValues, t.nnzValues, &x, Ncur, this->stream);
    }
    if (verbosity >= 3) {
        HIP_CHECK(hipStreamSynchronize(this->stream));
        c_amg_postsmooth += t_postsmooth.stop();
        if(verbosity >= 4){
            std::ostringstream out;
            out << "---------------postsmooth time: " << t_postsmooth.elapsed() << " s";
            OpmLog::info(out.str());
        }
     }
}

// x = prec(y)
template <class Scalar, unsigned int block_size>
void rocsparseCPR<Scalar, block_size>::
apply_amg(const Scalar& y, 
          Scalar& x)
{
    HIP_CHECK(hipMemsetAsync(d_coarse_x.data()->nnzValues, 0, sizeof(Scalar) * this->Nb, this->stream));
    
    Timer t_upload;
    for (unsigned int i = 0; i < d_u.size(); ++i) {
        d_u[i].upload(this->Rmatrices[i].nnzValues.data(), this->stream);
    }
    if (verbosity >= 3) {
        HIP_CHECK(hipStreamSynchronize(this->stream));
        c_amg_Dupload += t_upload.stop();
        if(verbosity >= 4){
            std::ostringstream out;
            out << "---------------d_u upload time: " << t_upload.elapsed() << " s";
            OpmLog::info(out.str());
        }
     }
    
    Timer t_residual;
    HipKernels<Scalar>::residual(d_mat->nnzValues, d_mat->colIndices, d_mat->rowPointers, &x, &y, d_rs.data()->nnzValues, this->Nb, block_size, this->stream);

    if (verbosity >= 3) {
        HIP_CHECK(hipStreamSynchronize(this->stream));
        c_amg_residual += t_residual.stop();
        if(verbosity >= 4){
            std::ostringstream out;
            out << "---------------residual time2: " << t_residual.elapsed() << " s";
            OpmLog::info(out.str());
        }
     }
    
    Timer t_restriction;
    HipKernels<Scalar>::full_to_pressure_restriction(d_rs.data()->nnzValues, d_weights.data()->nnzValues, d_coarse_y.data()->nnzValues, Nb, this->stream);
    if (verbosity >= 3) {
        HIP_CHECK(hipStreamSynchronize(this->stream));
        c_amg_restriction += t_restriction.stop();
        if(verbosity >= 4){
            std::ostringstream out;
            out << "---------------P extraction time: " << t_restriction.elapsed() << " s";
            OpmLog::info(out.str());
        }
     }
    
    amg_cycle_gpu(0, *(d_coarse_y.data()->nnzValues), *(d_coarse_x.data()->nnzValues));

    Timer t_correction;
    HipKernels<Scalar>::add_coarse_pressure_correction(d_coarse_x.data()->nnzValues, &x, this->pressure_idx, Nb, this->stream);
    
    if (verbosity >= 3) {
        HIP_CHECK(hipStreamSynchronize(this->stream));
        c_amg_correction += t_correction.stop();
        if(verbosity >= 4){
            std::ostringstream out;
            out << "---------------P correction time: " << t_correction.elapsed() << " s";
            OpmLog::info(out.str());
        }
     }
}

template <class Scalar, unsigned int block_size>
void rocsparseCPR<Scalar, block_size>::
apply(Scalar& y,
      Scalar& x)
{
    Dune::Timer t_bilu0;

    bilu0->apply(y, x);

    if (verbosity >= 3) {
        HIP_CHECK(hipStreamSynchronize(this->stream));
        c_cprilu0_apply += t_bilu0.stop();
        if(verbosity >= 4){
            std::ostringstream out;
            out << "---------rocsparseCPR apply bilu0(): " << t_bilu0.elapsed() << " s";
            OpmLog::info(out.str());
        }
    }

    Dune::Timer t_amg;
    apply_amg(y, x);
    if (verbosity >= 3) {
        HIP_CHECK(hipStreamSynchronize(this->stream));
        c_amg_apply += t_amg.stop();
        if(verbosity >= 4){
            std::ostringstream out;
            out << "---------rocsparseCPR apply amg(): " << t_amg.elapsed() << " s";
            OpmLog::info(out.str());
        }
    }
}

template<class Scalar, unsigned int block_size>
void rocsparseCPR<Scalar,block_size>::
printPrecApplyTimes(std::ostringstream* out)
{
        *out << "-------rocsparseCPR::cum ilu0_apply:  " << c_cprilu0_apply << " s\n";
        *out << "-------rocsparseCPR::cum amg_apply:   " << c_amg_apply << " s\n";
        *out << "--------------------  P restriction:    " << c_amg_restriction << " s\n";
        *out << "--------------------  residual:         " << c_amg_residual << " s\n";
        *out << "--------------------  spmv:             " << c_amg_spmv << " s\n";
        *out << "--------------------  Dupload:          " << c_amg_Dupload << " s\n";
        *out << "--------------------  copy umfpack:     " << c_amg_coursecopy << " s\n";
        *out << "--------------------  compute umfpack:  " << c_amg_coursecompute << " s\n";
        *out << "--------------------  prolongate:       " << c_amg_prolongate << " s\n";
        *out << "--------------------  P correction:     " << c_amg_correction << " s\n";
        *out << "--------------------  presmooth:        " << c_amg_presmooth << " s\n";
        *out << "--------------------  postsmooth:       " << c_amg_postsmooth << " s\n";
}

#define INSTANTIATE_TYPE(T)           \
    template class rocsparseCPR<T,1>; \
    template class rocsparseCPR<T,2>; \
    template class rocsparseCPR<T,3>; \
    template class rocsparseCPR<T,4>; \
    template class rocsparseCPR<T,5>; \
    template class rocsparseCPR<T,6>;

INSTANTIATE_TYPE(double)

#if FLOW_INSTANTIATE_FLOAT
INSTANTIATE_TYPE(float)
#endif

} // namespace Opm
