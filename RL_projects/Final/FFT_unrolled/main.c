#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#include "device.h"
#include "kernel.h"

#include "matrix.h"
#define CHECK_ERR(err, msg)                           \
    if (err != CL_SUCCESS)                            \
    {                                                 \
        fprintf(stderr, "%s failed: %d\n", msg, err); \
        exit(EXIT_FAILURE);                           \
    }

///////
#define DEBUG 0
void fft(float data_re[], float data_im[], const unsigned int N);

// helper functions called by the fft
// data will first be rearranged then computed
// an array of  {1, 2, 3, 4, 5, 6, 7, 8} will be
// rearranged to {1, 5, 3, 7, 2, 6, 4, 8}
void rearrange(float data_re[], float data_im[], const unsigned int N);
void compute(float data_re[], float data_im[], const unsigned int N);
///
void fft(float data_re[], float data_im[], const unsigned int N)
{
  rearrange(data_re, data_im, N);
  compute(data_re, data_im, N);
}

void rearrange(float data_re[], float data_im[], const unsigned int N)
{
  unsigned int target = 0;
  for(unsigned int position=0; position<N;position++)
    {
      if(target>position) {
        const float temp_re = data_re[target];
        const float temp_im = data_im[target];
        data_re[target] = data_re[position];
        data_im[target] = data_im[position];
        data_re[position] = temp_re;
        data_im[position] = temp_im;
      }
      unsigned int mask = N;
      while(target & (mask >>=1))
	      target &= ~mask;
        target |= mask;
    }
}

void compute(float data_re[], float data_im[], const unsigned int N)
{
  const float pi = -3.14159265358979323846;
  const int unroll = 2;
  for(unsigned int step=1; step<N; step <<=1) {
    const unsigned int jump = step << 1;
    const unsigned int jump_unroll = jump * unroll;
    const float step_d = (float) step;
    float twiddle_re = 1.0;
    float twiddle_im = 0.0;
    for(unsigned int group=0; group<step; group++){
		if(group+jump_unroll >= N){ //use naive if can't unroll at all (e.g. just 2 dft's)       
          for(unsigned int pair_N=group; pair_N<N; pair_N+=jump){
            const unsigned int match = pair_N + step;
            const float product_re = twiddle_re*data_re[match]-twiddle_im*data_im[match];
            const float product_im = twiddle_im*data_re[match]+twiddle_re*data_im[match];
            data_re[match] = data_re[pair_N]-product_re;
            data_im[match] = data_im[pair_N]-product_im;
            data_re[pair_N] += product_re;
            data_im[pair_N] += product_im;
          }
		}
		else{ //unroll if can             
          for(unsigned int pair=group; pair<N; pair+=jump_unroll){
                const unsigned int match1 = pair + step;
                const unsigned int pair2 = pair + jump;
                const unsigned int match2 = pair2 + step;
                const float product_re = twiddle_re*data_re[match1]-twiddle_im*data_im[match1];
                const float product_im = twiddle_im*data_re[match1]+twiddle_re*data_im[match1];
                const float product_re2 = twiddle_re*data_re[match2]-twiddle_im*data_im[match2];
                const float product_im2 = twiddle_im*data_re[match2]+twiddle_re*data_im[match2];
                data_re[match1] = data_re[pair]-product_re;
                data_im[match1] = data_im[pair]-product_im;
                data_re[match2] = data_re[pair2]-product_re2;
                data_im[match2] = data_im[pair2]-product_im2;
                data_re[pair] += product_re;
                data_im[pair] += product_im;
                data_re[pair2] += product_re2;
                data_im[pair2] += product_im2;
          }
		}
    if(group+1 == step)
        continue;
    float angle = pi*((float) group+1)/step_d;
    twiddle_re = cos(angle);
    twiddle_im = sin(angle);
    }
  }
}

///////
int compare_arrays(const float x[], const float y[],  const unsigned int N, const float eps);
void print_arr(const float data[], const unsigned int N);
void print_test_result(int tc_re, int tc_im, int tc_num);
void print_complex_data(float data_re[], float data_im[], int len);

// We will run 4 test cases to ensure our FFT data is correct
int main(int argc,  char **argv)
{
  clock_t start,  stop;
  double elapsed_time;

  cl_int err;
  const char *input0_file = argv[1], *input1_file = argv[2];
  const char *output0_file = argv[3], *output1_file = argv[4];
  const char *expected0_file = argv[5], *expected1_file = argv[6];

  Matrix input0; //input0 Re
  Matrix input1; //input0 I
  Matrix output0; //output0 Re
  Matrix output1; //output1 Im
  Matrix expected0; //expected0 Re
  Matrix expected1; //expected1 Im
  err = LoadMatrix(input0_file, &input0);
  CHECK_ERR(err, "LoadMatrix");
  err = LoadMatrix(input1_file, &input1);
  CHECK_ERR(err, "LoadMatrix");
  //expected
  err = LoadMatrix(expected0_file, &expected0);
  CHECK_ERR(err, "LoadMatrix");
  err = LoadMatrix(expected1_file, &expected1);
  CHECK_ERR(err, "LoadMatrix");

  unsigned int N = input0.shape[0];
  printf("N = %d\n",N);

  if (DEBUG){
    printf("---Input0 & 1:\n");
    for(unsigned int n=0;n<N;n++)
      printf("%.2f + j(%.2f)\n", (float) input0.data[n],  (float) input1.data[n]);
    
    printf("---Expected 0 & 1:\n");
    for(unsigned int n=0;n<N;n++)
      printf("%.2f + j(%.2f)\n", (float) expected0.data[n],  (float) expected1.data[n]);
    
    printf("---Input Data (Files: input0.raw & input1.raw), N = %d\n", N);
    print_complex_data(input0.data, input1.data, N);
  }
  // Test Case 1
  output0 = input0;
  output1 = input1;
  printf("loaded Input files...done\n");
  start = clock();//~
  fft(output0.data, output1.data, N);
  stop = clock();//~
  elapsed_time = ((double) (stop - start)) / CLOCKS_PER_SEC;
 
  if (DEBUG){
    printf("---Frequency Domain:\n");
    print_complex_data(input0.data, input1.data, N); //print_complex_data(data1_re, data1_im, N);
  }
  // Save the matrix
  SaveMatrix(output0_file, &input0);
  SaveMatrix(output1_file, &input1);

  // Check the result of the matrix multiply
  printf("CheckMatrix      (Real): \t");CheckMatrix(&output0, &expected0);
  printf("CheckMatrix (Imaginary): \t");CheckMatrix(&output1, &expected1);

  printf("Computational-time, FFT algorithm (N= %d) = %.6f Seconds\n", N, elapsed_time);
 
}

void print_complex_data(float data_re[], float data_im[], int len)
{
  printf("\tReal: ");
  for (int i=0;i<len;i++)
    printf("%.2f ", data_re[i]);
  printf("\n\tImag: ");
  for (int i=0;i<len;i++)
    printf("%.2f ", data_im[i]);
  printf("\n");
}

void print_test_result(int tc_re, int tc_im, int tc_num)
{
  int res = tc_re+tc_im;
  if(res == 2) {
    printf("Test Case %d: Passed\n", tc_num);
  } else {
    printf("Test Case %d: Failed\n", tc_num);
  }
}

int compare_arrays(const float x[], const float y[], const unsigned int N, const float eps)
{
  int result = 1;
  for(unsigned int i=0;i<N;i++)
    {
      if(fabs(x[i]-y[i])>eps) {
	result = 0;
      }
    }

  if(result==0)
    {
      printf("Expected: ");
      print_arr(y, N);
      printf("Got     : ");
      print_arr(x, N);
    }

  return 1;
}

void print_arr(const float data[], const unsigned int N)
{
  printf("{");
  for(unsigned int i=0;i<N-1;i++)
    printf("%.3f, ", data[i]);
  printf("%.3f}\n", data[N-1]);
}