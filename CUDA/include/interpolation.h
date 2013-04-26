/******************************************************************************/
//  Author: Codehathi (codehathi@gmail.com)
//  File: interp.h
//  Date: 02 Sep 2012
//  Description: This header file contains some simple macros used to keep the
//               code clean.  Also, it contains some constants used to set up
//               the lookup table.
//
/******************************************************************************/

#ifndef __INTERPOLATION_H__
#define __INTERPOLATION_H__

/******************************************************************************/
/* Information about the lookup table */

/*
   This is the number of points to use in the potential
   energy lookup table for the CUDA implementation
*/
#define N_TEX_PTS 1536

/*
   These are minimum and maximum values for log_2(R^2)
   Used in the CUDA potential energy lookup table
*/
#define XMIN 1.0f
#define XMAX 10.0f

#define TEX_MUL ((N_TEX_PTS-1)/(XMAX-XMIN))
/******************************************************************************/

/* Helpful macros */
#define CUDA_SAFE_CALL_NO_SYNC( call) {                                 \
    cudaError err = call;                                               \
    if( cudaSuccess != err) {                                           \
      fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",     \
              __FILE__, __LINE__, cudaGetErrorString( err) );           \
      exit(EXIT_FAILURE);                                               \
    } }

#define CUDA_SAFE_CALL( call)     CUDA_SAFE_CALL_NO_SYNC(call);

#define device_alloc(arr, size)                         \
  CUDA_SAFE_CALL(cudaMalloc((void**)&arr, size));

#define to_device(to, from, size)                                       \
  CUDA_SAFE_CALL(cudaMemcpy(to, from, size, cudaMemcpyHostToDevice));

#define alloc_and_copy(to, from, size)          \
  device_alloc(to, size);                       \
  to_device(to, from, size);

#define from_device(to, from, size)                                     \
  CUDA_SAFE_CALL(cudaMemcpy(to, from, size, cudaMemcpyDeviceToHost));



#endif
