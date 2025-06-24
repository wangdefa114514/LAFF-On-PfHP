#define alpha( i,j ) A[ (j)*ldA + i ]   // map alpha( i,j ) to array A 
#define beta( i,j )  B[ (j)*ldB + i ]   // map beta( i,j )  to array B
#define gamma( i,j ) C[ (j)*ldC + i ]   // map gamma( i,j ) to array C


void MyGemv( int m, int n, double *A, int ldA, double *x, int incx, double *y, int incy );
//here m is the number of rows in A, n is the number of columns in A, 
//x is the column of B
// and y is the output vector, which is the column of C
void MyGemm( int m, int n, int k,
	     double *A, int ldA,
	     double *B, int ldB,
	     double *C, int ldC )
{
  for ( int j=0; j<n; j++ )
    MyGemv(  m, k , A , ldA , &beta(0,j) , 1 , &gamma(0,j) , 1  );
}
  
