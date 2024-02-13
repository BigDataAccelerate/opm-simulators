#ifndef MISC_HPP
#define MISC_HPP

namespace Opm
{
namespace Accelerator
{

unsigned int ceilDivision(const unsigned int A, const unsigned int B);
double get_absmax(const double *data, const int N);
void solve_transposed_3x3(const double *A, const double *b, double *x);

}
}

#endif
