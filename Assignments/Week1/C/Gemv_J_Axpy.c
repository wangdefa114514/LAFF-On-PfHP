#define alpha( i,j ) A[ (j)*ldA + i ]   // map alpha( i,j ) to array A 
#define chi( i )  x[ (i)*incx ]         // map chi( i )  to array x
#define psi( i )  y[ (i)*incy ]         // map psi( i )  to array y

void Axpy( int n, double alpha, double *x, int incx, double *y, int incy );
//here y is the output vector, x is the input vector, which is the column of A

void MyGemv( int m, int n, double *A, int ldA,
           double *x, int incx, double *y, int incy )
{
  for ( int j=0; j<n; j++ )
    Axpy(  m  ,    chi(j) ,     &alpha(0,j)   ,  1  ,   y ,  incy     );
}
