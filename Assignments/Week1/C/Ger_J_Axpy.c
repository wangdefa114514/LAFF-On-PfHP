#define alpha( i,j ) A[ (j)*ldA + i ]   // map alpha( i,j ) to array A 
#define chi( i )  x[ (i)*incx ]         // map chi( i )  to array x
#define psi( i )  y[ (i)*incy ]         // map psi( i )  to array y

void Axpy( int n, double alpha, double *x, int incx, double *y, int incy );
//here y is the output vector, x is the input vector which is x
//alpha is the jth ele of y
//y is the output vector which is the jth column of A
void MyGer( int m, int n, double *x, int incx,
	  double *y, int incy, double *A, int ldA )
{
  for ( int j=0; j<n; j++ )
    Axpy(   n ,   psi(j)   ,  x   ,  incx    ,    &alpha(0,j)   ,  1      );
}
