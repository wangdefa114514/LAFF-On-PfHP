#define alpha(i, j) A[(j) * ldA + i] // map alpha( i,j ) to array A
#define chi(i) x[(i) * incx]         // map chi( i )  to array x
#define psi(i) y[(i) * incy]         // map psi( i )  to array y

void Axpy(int n, double alpha, double *x, int incx, double *y, int incy);
// here alpha is the ith ele of x, x is the input vec which is y
// y is the output vec which is the ith row of A
void MyGer(int m, int n, double *x, int incx,
           double *y, int incy, double *A, int ldA)
//// MyGer computes the outer product of x and y, storing the result in A
{
  for (int i = 0; i < m; i++)
    Axpy(n, chi(i), y, incy, &alpha(i, 0), ldA);
}
