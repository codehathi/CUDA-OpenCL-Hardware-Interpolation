/******************************************************************************/
//  Author: Codehathi (codehathi@gmail.com)
//  File: interpolation.cu
//  Date: 02 Sep 2012
//  Description: This program does the hardware linear interpolation in CUDA
//               in order to create a lookup table for the HFDB-He potential
//               energy function.
//
/******************************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <stdint.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include "interpolation.h"

// This is the texture memory variable.  This will hold the potential
// energy lookup table.
texture<float, 1, cudaReadModeElementType> potcurve;

double hfdbhe(double r2)
{
  double astar  = 1.8443101e5;
  double alstar = 10.43329537;
  double bestar = -2.27965105;
  double d      = 1.4826;
  double c6     = 1.36745214;
  double c8     = 0.42123807;
  double c10    = 0.17473318;
  double rm     = 5.59926;
  double eps    = 10.948;
  double hart   = 315774.65;

  if(r2 <= 0) return hart;

  double x = sqrt(r2)/rm;

  double x2 = x*x;
  double x4 = x2*x2;
  double x8 = x4*x4;

  double vstar = astar * exp(-alstar * x + bestar * x2);
  double vd    = (c6 / (x4*x2) + /* c6/x^6 */
                  c8 / (x8) + /* c8/x^8 */
                  c10 / (x8*x2)); /* c10/x^10 */
  if(x < d)
    {
      double t = d/x - 1.0;
      vd *= exp(-t * t);
    }
  return (vstar - vd) * eps / hart;
}

// Set up the texture memory
__host__ void setTextureMemory(cudaArray** d_vcurve)
{
  // Allocate space for the lookup table entries.
  float* node_v = (float*)calloc(N_TEX_PTS, sizeof(float));
  if( !node_v )
    {
      perror("node_v");
      exit(1);
    }

  // Store the log scale of the HFDB-He lookup table
  for (int n=0; n<N_TEX_PTS; n++)
    {
      float xx = (XMAX-XMIN)/( (float) (N_TEX_PTS-1) )*( (float) n ) + XMIN;
      float rr = exp(xx*log(2.0f)); // this is equivalent to (2.0f)^xx
      node_v[n] = hfdbhe(rr);
    }

  cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();

  // Allocate a CUDA array
  cudaMallocArray(&(*d_vcurve), &channelDesc, N_TEX_PTS, 1);

  // Copy the array to the device
  cudaMemcpyToArray(*d_vcurve, 0, 0, node_v,
                    N_TEX_PTS*sizeof(float), cudaMemcpyHostToDevice);

  // Set address clamping and linear filter mode
  potcurve.addressMode[0] = cudaAddressModeClamp;
  potcurve.filterMode = cudaFilterModeLinear;
  potcurve.normalized = false;

  // Bind the array to the texture
  cudaBindTextureToArray(potcurve, *d_vcurve, channelDesc);

  free(node_v);
}

bool initialize_gpu_log(uint64_t count,
                        float** energies,
                        float** d_energies,
                        cudaArray** d_vcurve)
{
  // Set the texture memory
  setTextureMemory(d_vcurve);

  // Allocate host memory for calculated energies
  *energies = (float*)calloc(count, sizeof(float));
  if(!(*energies))
    {
      perror("energies");
      return false;
    }

  // Allocate device memory for the calculated energies
  alloc_and_copy((*d_energies), (*energies), sizeof(float)*count);
  return true;
}

// Test kernel to show how to use the interpolation This is only for
// testing.  This does not show performance since I'm limiting to a
// single thread.
__global__ void interpolate_lookup(float * energies)
{
  if(threadIdx.x == 0)
    {
      int j=0;
      for(float i=2.0; i<1024; i+=.01, j++)
        {
          // Put the lookup value on the log scale and accommodate the
          // table index calculations
          float  x = log2f(i);
          x = 0.5f + (x - XMIN) * TEX_MUL;

          // Do the lookup
          energies[j] = tex1D(potcurve, x);
        }
    }
}

int main(int argc, char* argv[])
{
  float* energies;
  float* d_energies;
  cudaArray* d_vcurve;

  if( !initialize_gpu_log(10000000, &energies, &d_energies, &d_vcurve) )
    {
      return 1;
    }

  interpolate_lookup<<< 1, 64 >>> (d_energies);
  CUDA_SAFE_CALL( cudaThreadSynchronize() );
  from_device(energies, d_energies, sizeof(float)*10000000);

  for(float i=2.0, j=0; i<1024; i+=.01,j++)
    printf("%f\t%e\t%e\t%e\n",i, hfdbhe(i), energies[(int)j],
           (hfdbhe(i)-energies[(int)j])/hfdbhe(i));

  cudaFreeArray(d_vcurve);
  cudaFree(d_energies);
  free(energies);
  return 0;
}
