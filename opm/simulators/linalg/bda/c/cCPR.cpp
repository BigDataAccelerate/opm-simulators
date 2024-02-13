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
#include <opm/simulators/linalg/bda/c/cCPR.hpp>

namespace Opm
{
namespace Accelerator
{

using Opm::OpmLog;
using Dune::Timer;

template <unsigned int block_size>
cCPR<block_size>::cCPR(int verbosity_, bool opencl_ilu_parallel_) :
    cPreconditioner<block_size>(verbosity_)
{
    opencl_ilu_parallel = opencl_ilu_parallel_;
    bilu0 = std::make_unique<cBILU0<block_size> >(opencl_ilu_parallel, verbosity_);
    diagIndices.resize(1);
}

template <unsigned int block_size>
bool cCPR<block_size>::analyze_matrix(BlockedMatrix *mat_) {
std::cout << "-----in : cCPR<block_size>::analyze_matrix(mat_)\n";
    this->Nb = mat_->Nb;
    this->nnzb = mat_->nnzbs;
    this->N = Nb * block_size;
    this->nnz = nnzb * block_size * block_size;
    
    bool success = bilu0->analyze_matrix(mat_);
    
    mat = mat_;
std::cout << "-----out: cCPR<block_size>::analyze_matrix(mat_)\n";
    return success;
}

template <unsigned int block_size>
bool cCPR<block_size>::analyze_matrix(BlockedMatrix *mat_, BlockedMatrix *jacMat) {
std::cout << "-----in : cCPR<block_size>::analyze_matrix(mat_, jacMat)\n";
    this->Nb = mat_->Nb;
    this->nnzb = mat_->nnzbs;
    this->N = Nb * block_size;
    this->nnz = nnzb * block_size * block_size;

    bool success = bilu0->analyze_matrix(mat_, jacMat);
    mat = mat_;
std::cout << "-----out: cCPR<block_size>::analyze_matrix(mat_, jacMat)\n";

    return success;
}

template <unsigned int block_size>
bool cCPR<block_size>::create_preconditioner(BlockedMatrix *mat_, BlockedMatrix *jacMat) {
    Dune::Timer t_bilu0;
    bool result = bilu0->create_preconditioner(mat_, jacMat);
    if (verbosity >= 3) {
        std::ostringstream out;
        out << "cCPR create_preconditioner bilu0(): " << t_bilu0.stop() << " s";
        OpmLog::info(out.str());
    }

    Dune::Timer t_amg;
    create_preconditioner_amg(mat); // already points to bilu0::rmat if needed
    if (verbosity >= 3) {
        std::ostringstream out;
        out << "cCPR create_preconditioner_amg(): " << t_amg.stop() << " s";
        OpmLog::info(out.str());
    }
    return result;
}

template <unsigned int block_size>
bool cCPR<block_size>::create_preconditioner(BlockedMatrix *mat_) {
    Dune::Timer t_bilu0;
    bool result = bilu0->create_preconditioner(mat_);
    if (verbosity >= 3) {
        std::ostringstream out;
        out << "cCPR create_preconditioner bilu0(): " << t_bilu0.stop() << " s";
        OpmLog::info(out.str());
    }

    Dune::Timer t_amg;
    create_preconditioner_amg(mat); // already points to bilu0::rmat if needed
    if (verbosity >= 3) {
        std::ostringstream out;
        out << "cCPR create_preconditioner_amg(): " << t_amg.stop() << " s";
        OpmLog::info(out.str());
    }
    return result;
}


// return the absolute value of the N elements for which the absolute value is highest
double get_absmax(const double *data, const int N) {
    return std::abs(*std::max_element(data, data + N, [](double a, double b){return std::fabs(a) < std::fabs(b);}));
}


// solve A^T * x = b
void solve_transposed_3x3(const double *A, const double *b, double *x) {
    const int B = 3;
    // from dune-common/densematrix.hh, but transposed, so replace [r*B+c] with [r+c*B]
    double t4  = A[0+0*B] * A[1+1*B];
    double t6  = A[0+0*B] * A[1+2*B];
    double t8  = A[0+1*B] * A[1+0*B];
    double t10 = A[0+2*B] * A[1+0*B];
    double t12 = A[0+1*B] * A[2+0*B];
    double t14 = A[0+2*B] * A[2+0*B];

    double d = (t4*A[2+2*B]-t6*A[2+1*B]-t8*A[2+2*B]+
          t10*A[2+1*B]+t12*A[1+2*B]-t14*A[1+1*B]); //determinant

    x[0] = (b[0]*A[1+1*B]*A[2+2*B] - b[0]*A[2+1*B]*A[1+2*B]
          - b[1] *A[0+1*B]*A[2+2*B] + b[1]*A[2+1*B]*A[0+2*B]
          + b[2] *A[0+1*B]*A[1+2*B] - b[2]*A[1+1*B]*A[0+2*B]) / d;

    x[1] = (A[0+0*B]*b[1]*A[2+2*B] - A[0+0*B]*b[2]*A[1+2*B]
          - A[1+0*B] *b[0]*A[2+2*B] + A[1+0*B]*b[2]*A[0+2*B]
          + A[2+0*B] *b[0]*A[1+2*B] - A[2+0*B]*b[1]*A[0+2*B]) / d;

    x[2] = (A[0+0*B]*A[1+1*B]*b[2] - A[0+0*B]*A[2+1*B]*b[1]
          - A[1+0*B] *A[0+1*B]*b[2] + A[1+0*B]*A[2+1*B]*b[0]
          + A[2+0*B] *A[0+1*B]*b[1] - A[2+0*B]*A[1+1*B]*b[0]) / d;
}




template <unsigned int block_size>
void cCPR<block_size>::create_preconditioner_amg(BlockedMatrix *mat_) {
    this->mat = mat_;

    coarse_vals.resize(nnzb);
    coarse_x.resize(Nb);
    coarse_y.resize(Nb);
    weights.resize(N);

    try{
        double rhs[] = {0, 0, 0};
        rhs[pressure_idx] = 1;

        // find diagonal index for each row
        if (diagIndices[0].empty()) {
            diagIndices[0].resize(Nb);
            for (int row = 0; row < Nb; ++row) {
                int start = mat->rowPointers[row];
                int end = mat->rowPointers[row + 1];
                auto candidate = std::find(mat->colIndices + start, mat->colIndices + end, row);
                assert(candidate != mat->colIndices + end);
                diagIndices[0][row] = candidate - mat->colIndices;
            }
        }

        // calculate weights for each row
        for (int row = 0; row < Nb; ++row) {
            // solve to find weights
            double *row_weights = weights.data() + block_size * row; // weights for this row
            solve_transposed_3x3(mat->nnzValues + block_size * block_size * diagIndices[0][row], rhs, row_weights);

            // normalize weights for this row
            double abs_max = get_absmax(row_weights, block_size);
            for(unsigned int i = 0; i < block_size; i++){
                row_weights[i] /= abs_max;
            }
        }

        // extract pressure
        // transform blocks to scalars to create scalar linear system
        for (int row = 0; row < Nb; ++row) {
            int start = mat->rowPointers[row];
            int end = mat->rowPointers[row + 1];
            for (int idx = start; idx < end; ++idx) {
                double *block = mat->nnzValues + idx * block_size * block_size;
                double *row_weights = weights.data() + block_size * row;
                double value = 0.0;
                for (unsigned int i = 0; i < block_size; ++i) {
                    value += block[block_size * i + pressure_idx] * row_weights[i];
                }
                coarse_vals[idx] = value;
            }
        }

#if HAVE_MPI
        using Communication = Dune::OwnerOverlapCopyCommunication<int, int>;
#else
        using Communication = Dune::Amg::SequentialInformation;
#endif
        using OverlapFlags = Dune::NegateSet<Communication::OwnerSet>;
        if (recalculate_aggregates) {
            dune_coarse = std::make_unique<DuneMat>(Nb, Nb, nnzb, DuneMat::row_wise);

            typedef DuneMat::CreateIterator Iter;

            // setup sparsity pattern
            for(Iter row = dune_coarse->createbegin(); row != dune_coarse->createend(); ++row){
                int start = mat->rowPointers[row.index()];
                int end = mat->rowPointers[row.index() + 1];
                for (int idx = start; idx < end; ++idx) {
                    int col = mat->colIndices[idx];
                    row.insert(col);
                }
            }

            // set values
            for (int row = 0; row < Nb; ++row) {
                int start = mat->rowPointers[row];
                int end = mat->rowPointers[row + 1];
                for (int idx = start; idx < end; ++idx) {
                    int col = mat->colIndices[idx];
                    (*dune_coarse)[row][col] = coarse_vals[idx];
                }
            }

            dune_op = std::make_shared<MatrixOperator>(*dune_coarse);
            Dune::Amg::SequentialInformation seqinfo;
            dune_amg = std::make_unique<DuneAmg>(dune_op, Dune::stackobject_to_shared_ptr(seqinfo));

            Opm::PropertyTree property_tree;
            property_tree.put("alpha", 0.333333333333);

            // The matrix has a symmetric sparsity pattern, but the values are not symmetric
            // Yet a SymmetricDependency is used in AMGCPR
            // An UnSymmetricCriterion is also available
            // using Criterion = Dune::Amg::CoarsenCriterion<Dune::Amg::UnSymmetricCriterion<DuneMat, Dune::Amg::FirstDiagonal> >;
            using CriterionBase = Dune::Amg::AggregationCriterion<Dune::Amg::SymmetricDependency<DuneMat, Dune::Amg::FirstDiagonal>>;
            using Criterion = Dune::Amg::CoarsenCriterion<CriterionBase>;
            const Criterion c = Opm::AMGHelper<MatrixOperator,Dune::Amg::SequentialInformation,DuneMat,DuneVec>::criterion(property_tree);
            num_pre_smooth_steps = c.getNoPreSmoothSteps();
            num_post_smooth_steps = c.getNoPostSmoothSteps();

            dune_amg->build<OverlapFlags>(c);

            analyzeHierarchy();
            analyzeAggregateMaps();

            recalculate_aggregates = false;
        } else {
            // update values of coarsest level in AMG
            // this works because that level is actually a reference to the DuneMat held by dune_coarse
            for (int row = 0; row < Nb; ++row) {
                int start = mat->rowPointers[row];
                int end = mat->rowPointers[row + 1];
                for (int idx = start; idx < end; ++idx) {
                    int col = mat->colIndices[idx];
                    (*dune_coarse)[row][col] = coarse_vals[idx];
                }
            }

            // update the rest of the AMG hierarchy
            dune_amg->recalculateGalerkin(OverlapFlags());
            analyzeHierarchy();
        }

        // initialize OpenclMatrices and Buffers if needed
//         auto init_func = std::bind(&cCPR::init_opencl_buffers, this);
//         std::call_once(opencl_buffers_allocated, init_func);

        // upload matrices and vectors to GPU
//         opencl_upload();

    } catch (const std::exception& ex) {
        std::cerr << "Caught exception: " << ex.what() << std::endl;
        throw ex;
    }
}


template <unsigned int block_size>
void cCPR<block_size>::analyzeHierarchy() {
    const DuneAmg::ParallelMatrixHierarchy& matrixHierarchy = dune_amg->matrices();

    // store coarsest AMG level in umfpack format, also performs LU decomposition
    umfpack.setMatrix((*matrixHierarchy.coarsest()).getmat());

    num_levels = dune_amg->levels();
    level_sizes.resize(num_levels);
    diagIndices.resize(num_levels);

    Amatrices.reserve(num_levels);
    Rmatrices.reserve(num_levels - 1);  // coarsest level does not need one
    invDiags.reserve(num_levels);

    Amatrices.clear();
    invDiags.clear();

    // matrixIter.dereference() returns MatrixAdapter
    // matrixIter.dereference().getmat() returns BCRSMat
    DuneAmg::ParallelMatrixHierarchy::ConstIterator matrixIter = matrixHierarchy.finest();
    for(int level = 0; level < num_levels; ++matrixIter, ++level) {
        const auto& A = matrixIter.dereference().getmat();
        level_sizes[level] = A.N();
        diagIndices[level].reserve(A.N());

        // extract matrix A
        Amatrices.emplace_back(A.N(), A.nonzeroes());
        // contiguous copy is not possible
        // std::copy(&(A[0][0][0][0]), &(A[0][0][0][0]) + A.nonzeroes(), Amatrices.back().nnzValues.data());
        // also update diagonal indices if needed, level 0 is already filled in create_preconditioner()
        int nnz_idx = 0;
        const bool fillDiagIndices = diagIndices[level].empty();
        for (typename DuneMat::const_iterator r = A.begin(); r != A.end(); ++r) {
            for (auto c = r->begin(); c != r->end(); ++c) {
                Amatrices.back().nnzValues[nnz_idx] = A[r.index()][c.index()];
                if (fillDiagIndices && r.index() == c.index()) {
                    diagIndices[level].emplace_back(nnz_idx);
                }
                nnz_idx++;
            }
        }

        Opm::BdaBridge<DuneMat, DuneVec, 1>::copySparsityPatternFromISTL(A, Amatrices.back().rowPointers, Amatrices.back().colIndices);

        // compute inverse diagonal values for current level
        invDiags.emplace_back(A.N());
        for (unsigned int row = 0; row < A.N(); ++row) {
            invDiags.back()[row] = 1 / Amatrices.back().nnzValues[diagIndices[level][row]];
        }
    }
}


template <unsigned int block_size>
void cCPR<block_size>::analyzeAggregateMaps() {

    PcolIndices.resize(num_levels - 1);
    Rmatrices.clear();

    const DuneAmg::AggregatesMapList& aggregatesMaps = dune_amg->aggregatesMaps();

    DuneAmg::AggregatesMapList::const_iterator mapIter = aggregatesMaps.begin();
    for(int level = 0; level < num_levels - 1; ++mapIter, ++level) {
        DuneAmg::AggregatesMap *map = *mapIter;

        Rmatrices.emplace_back(level_sizes[level+1], level_sizes[level], level_sizes[level]);
        std::fill(Rmatrices.back().nnzValues.begin(), Rmatrices.back().nnzValues.end(), 1.0);

        // get indices for each row of P and R
        std::vector<std::vector<unsigned> > indicesR(level_sizes[level+1]);
        PcolIndices[level].resize(level_sizes[level]);

        using AggregateIterator = DuneAmg::AggregatesMap::const_iterator;
        for(AggregateIterator ai = map->begin(); ai != map->end(); ++ai){
            if (*ai != DuneAmg::AggregatesMap::ISOLATED) {
                const long int diff = ai - map->begin();
                PcolIndices[level][diff] = *ai;
                indicesR[*ai].emplace_back(diff);
            }
        }

        int col_idx = 0;
        // set sparsity pattern of R
        Rmatrices.back().rowPointers[0] = 0;
        for (unsigned int i = 0; i < indicesR.size(); ++i) {
            Rmatrices.back().rowPointers[i + 1] = Rmatrices.back().rowPointers[i] + indicesR[i].size();
            for (auto it = indicesR[i].begin(); it != indicesR[i].end(); ++it) {
                Rmatrices.back().colIndices[col_idx++] = *it;
            }
        }
    }
}


template <unsigned int block_size>
void cCPR<block_size>::amg_cycle_gpu(const int level, double &y, double &x) {
    cMatrix *A = &d_Amatrices[level];
    cMatrix *R = &d_Rmatrices[level];
    int Ncur = A->Nb;

    if (level == num_levels - 1) {
        // solve coarsest level
        std::vector<double> h_y(Ncur), h_x(Ncur, 0);

        // solve coarsest level using umfpack
        umfpack.apply(h_x.data(), h_y.data());

        return;
    }
    int Nnext = d_Amatrices[level+1].Nb;

    double& t = d_t[level];
    double& f = d_f[level];
    double& u = d_u[level]; // u was 0-initialized earlier

    // presmooth
    double jacobi_damping = 0.65; // default value in amgcl: 0.72
    for (unsigned i = 0; i < num_pre_smooth_steps; ++i){
        std::cout << "TODO: OpenclKernels::residual(A->nnzValues, A->colIndices, A->rowPointers, x, y, t, Ncur, 1);\n";
        std::cout << "TODO: OpenclKernels::vmul(jacobi_damping, invDiags[level], t, x, Ncur);\n";
    }

    // move to coarser level
    std::cout << "TODO: OpenclKernels::residual(A->nnzValues, A->colIndices, A->rowPointers, x, y, t, Ncur, 1);\n";
    std::cout << "TODO: OpenclKernels::spmv(R->nnzValues, R->colIndices, R->rowPointers, t, f, Nnext, 1, true);\n";
    amg_cycle_gpu(level + 1, f, u);
    std::cout << "TODO: OpenclKernels::prolongate_vector(u, x, PcolIndices[level], Ncur);\n";

    // postsmooth
    for (unsigned i = 0; i < num_post_smooth_steps; ++i){
        std::cout << "TODO: OpenclKernels::residual(A->nnzValues, A->colIndices, A->rowPointers, x, y, t, Ncur, 1);\n";
        std::cout << "TODO: OpenclKernels::vmul(jacobi_damping, invDiags[level], t, x, Ncur);\n";
    }
}


// x = prec(y)
template <unsigned int block_size>
void cCPR<block_size>::apply_amg(const double& y, double& x) {

    std::cout << " TODO: OpenclKernels::residual(mat->nnzValues, mat->colIndices, mat->rowPointers, x, y, *rs, Nb, block_size);\n";
    std::cout << " TODO: OpenclKernels::full_to_pressure_restriction(*rs, *weights, *coarse_y, Nb);\n";
    std::cout << " TODO: OpenclKernels::residual(mat->nnzValues, mat->colIndices, mat->rowPointers, x, y, *rs, Nb, block_size);\n";
    std::cout << " TODO: OpenclKernels::full_to_pressure_restriction(*rs, *weights, *coarse_y, Nb);\n";

    amg_cycle_gpu(0, *d_coarse_y, *d_coarse_x);

    //OpenclKernels::add_coarse_pressure_correction(*d_coarse_x, x, pressure_idx, Nb);
    std::cout << " TODO: OpenclKernels::add_coarse_pressure_correction(*d_coarse_x, x, pressure_idx, Nb);\n";
}

template <unsigned int block_size>
void cCPR<block_size>::apply(double& y, double& x) {
    Dune::Timer t_bilu0;
    bilu0->apply(y, x);
    if (verbosity >= 4) {
        std::ostringstream out;
        out << "cCPR apply bilu0(): " << t_bilu0.stop() << " s";
        OpmLog::info(out.str());
    }

    Dune::Timer t_amg;
    apply_amg(y, x);
    if (verbosity >= 4) {
        std::ostringstream out;
        out << "cCPR apply amg(): " << t_amg.stop() << " s";
        OpmLog::info(out.str());
    }
}


#define INSTANTIATE_BDA_FUNCTIONS(n)  \
template class cCPR<n>;

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


