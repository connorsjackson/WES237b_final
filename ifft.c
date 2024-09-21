// Source: https://cplusplus.com/forum/general/265589/
#include <iostream>
#include <iomanip>
#include <complex>
using namespace std;

const double pi = 3.14159265358979323846;
using Complex = complex<double>;

void print( const char * prompt, Complex A[], int log2N );
void FFT ( Complex f[]     , Complex ftilde[], int log2N );
void iFFT( Complex ftilde[], Complex f[]     , int log2N );

//======================================================================

int main()
{
   const int N = 8, log2N = 3;                                             // Hardwired for testing
   //Complex fN] = { 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0 };
   //Complex f[N] = { 1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 1.0, -1.0 };
   Complex f[N] = { {1.0,7}, {2.0,6}, {3.0,5}, {4.0,4}, {5.0,3}, {4.0,2}, {1.0,1}, {-1.0,0} };   
Complex ft[N];
   print( "Original:", f, log2N );

   // Forward transform
   FFT ( f, ft, log2N );   print( "\nTransform:", ft, log2N );

   // Invert transform
   iFFT( ft, f, log2N );   print( "\nInvert:",     f, log2N );
}


//======================================================================

void print( const char * prompt, Complex A[], int log2N )
{
   int N = 1 << log2N;
   cout << prompt << '\n' << fixed;
   for ( int i = 0; i < N; i++ ) cout << A[i] << '\n';
}

//======================================================================

void FFT( Complex f[], Complex ftilde[], int log2N )                 // Fast Fourier Transform
{
   int N = 1 << log2N;

   // Reorder
   for ( int i = 0; i < N; i++ )
   {
      int ii = 0, x = i;
      for (int j = 0; j < log2N; j++)
      {
         ii <<= 1;
         ii |= ( x & 1 );
         x >>= 1;
      }
      ftilde[ii] = f[i];
   }

   for ( int s = 1; s <= log2N; s++ )
   {
      int m = 1 << s;
      int m2 = m / 2;
      Complex w = 1.0;
      Complex wm = polar( 1.0, -pi / m2 );
      for ( int j = 0; j < m2; j++ )
      {
         for ( int k = j; k < N; k += m )
         {
            Complex t = w * ftilde[k+m2];
            Complex u =     ftilde[k   ];
            ftilde[k   ] = u + t;
            ftilde[k+m2] = u - t;
         }
         w *= wm;
      }
   }
}

//======================================================================

void iFFT( Complex ftilde[], Complex f[], int log2N )                // Inverse Fast Fourier Transform
{
   int N = 1 << log2N;

   for ( int m = 0; m < N; m++ ) ftilde[m] = conj( ftilde[m] );      // Apply conjugate (reversed below)

   FFT( ftilde, f, log2N );

   double factor = 1.0 / N;
   for ( int m = 0; m < N; m++ ) f[m] = conj( f[m] ) * factor;

   for ( int m = 0; m < N; m++ ) ftilde[m] = conj( ftilde[m] );      // Only necessary to reinstate ftilde
}
