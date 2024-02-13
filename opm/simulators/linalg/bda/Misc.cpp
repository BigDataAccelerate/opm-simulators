#include <cmath>
#include <algorithm>

#include <opm/simulators/linalg/bda/Misc.hpp>

namespace Opm
{
namespace Accelerator
{

// divide A by B, and round up: return (int)ceil(A/B)
unsigned int ceilDivision(const unsigned int A, const unsigned int B)
{
    return A / B + (A % B > 0);
}

// return the absolute value of the N elements for which the absolute value is highest
double get_absmax(const double *data, const int N)
{
    return std::abs(*std::max_element(data, data + N, [](double a, double b){return std::fabs(a) < std::fabs(b);}));
}

// solve A^T * x = b
void solve_transposed_3x3(const double *A, const double *b, double *x)
{
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

}
}
