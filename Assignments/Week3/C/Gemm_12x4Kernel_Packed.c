#define alpha(i, j) A[(j) * ldA + (i)] // map alpha( i,j ) to array A
#define beta(i, j) B[(j) * ldB + (i)]  // map beta( i,j ) to array B
#define gamma(i, j) C[(j) * ldC + (i)] // map gamma( i,j ) to array C

#include <immintrin.h>

void Gemm_MRxNRKernel_Packed( int k,
		        double *MP_A, double *MP_B, double *C, int ldC )
{
   /* Declare vector registers to hold 4x4 C and load them */
   __m256d gamma_0123_0 = _mm256_loadu_pd(&gamma(0, 0));
   __m256d gamma_0123_1 = _mm256_loadu_pd(&gamma(0, 1));
   __m256d gamma_0123_2 = _mm256_loadu_pd(&gamma(0, 2));
   __m256d gamma_0123_3 = _mm256_loadu_pd(&gamma(0, 3));

   __m256d gamma_4567_0 = _mm256_loadu_pd(&gamma(4, 0));
   __m256d gamma_4567_1 = _mm256_loadu_pd(&gamma(4, 1));
   __m256d gamma_4567_2 = _mm256_loadu_pd(&gamma(4, 2));
   __m256d gamma_4567_3 = _mm256_loadu_pd(&gamma(4, 3));

   __m256d gamma_89AB_0 = _mm256_loadu_pd(&gamma(8, 0));
   __m256d gamma_89AB_1 = _mm256_loadu_pd(&gamma(8, 1));
   __m256d gamma_89AB_2 = _mm256_loadu_pd(&gamma(8, 2));
   __m256d gamma_89AB_3 = _mm256_loadu_pd(&gamma(8, 3));

   for (int p = 0; p < k; p++)
   {
      /* Declare vector register for load/broadcasting beta( p,j ) */
      __m256d beta_p_j;

      /* Declare a vector register to hold the current column of A and load
         it with the four elements of that column. */
      __m256d alpha_0123_p = _mm256_loadu_pd(MP_A);
      __m256d alpha_4567_p = _mm256_loadu_pd(MP_A + 4);
      __m256d alpha_89AB_p = _mm256_loadu_pd(MP_A + 8);
      /* Load/broadcast beta( p,0 ). */
      beta_p_j = _mm256_broadcast_sd(MP_B);

      /* update the first column of C with the current column of A times
         beta ( p,0 ) */
      gamma_0123_0 = _mm256_fmadd_pd(alpha_0123_p, beta_p_j, gamma_0123_0);
      gamma_4567_0 = _mm256_fmadd_pd(alpha_4567_p, beta_p_j, gamma_4567_0);
      gamma_89AB_0 = _mm256_fmadd_pd(alpha_89AB_p, beta_p_j, gamma_89AB_0);
      /* REPEAT for second, third, and fourth columns of C.  Notice that the
         current column of A needs not be reloaded. */

      beta_p_j = _mm256_broadcast_sd(MP_B + 1);
      gamma_0123_1 = _mm256_fmadd_pd(alpha_0123_p, beta_p_j, gamma_0123_1);
      gamma_4567_1 = _mm256_fmadd_pd(alpha_4567_p, beta_p_j, gamma_4567_1);
      gamma_89AB_1 = _mm256_fmadd_pd(alpha_89AB_p, beta_p_j, gamma_89AB_1);

      beta_p_j = _mm256_broadcast_sd(MP_B +2);
      gamma_0123_2 = _mm256_fmadd_pd(alpha_0123_p, beta_p_j, gamma_0123_2);
      gamma_4567_2 = _mm256_fmadd_pd(alpha_4567_p, beta_p_j, gamma_4567_2);
      gamma_89AB_2 = _mm256_fmadd_pd(alpha_89AB_p, beta_p_j, gamma_89AB_2);

      beta_p_j = _mm256_broadcast_sd(MP_B +3);
      gamma_0123_3 = _mm256_fmadd_pd(alpha_0123_p, beta_p_j, gamma_0123_3);
      gamma_4567_3 = _mm256_fmadd_pd(alpha_4567_p, beta_p_j, gamma_4567_3);
      gamma_89AB_3 = _mm256_fmadd_pd(alpha_89AB_p, beta_p_j, gamma_89AB_3);
      MP_A+=12;
      MP_B+=4;
   }

   /* Store the updated results */
   _mm256_storeu_pd(&gamma(0, 0), gamma_0123_0);
   _mm256_storeu_pd(&gamma(0, 1), gamma_0123_1);
   _mm256_storeu_pd(&gamma(0, 2), gamma_0123_2);
   _mm256_storeu_pd(&gamma(0, 3), gamma_0123_3);
   _mm256_storeu_pd(&gamma(4, 0), gamma_4567_0);
   _mm256_storeu_pd(&gamma(4, 1), gamma_4567_1);
   _mm256_storeu_pd(&gamma(4, 2), gamma_4567_2);
   _mm256_storeu_pd(&gamma(4, 3), gamma_4567_3);
   _mm256_storeu_pd(&gamma(8, 0), gamma_89AB_0);
   _mm256_storeu_pd(&gamma(8, 1), gamma_89AB_1);
   _mm256_storeu_pd(&gamma(8, 2), gamma_89AB_2);
   _mm256_storeu_pd(&gamma(8, 3), gamma_89AB_3);
   
}
