CUDA/OpenCL Hardware Interpolation Example
==========================================

This code shows how to use the texture hardware interpolation units on
GPUs using CUDA and OpenCL.  I coded this up only because I found
little to no examples out there that actually showed how to use the
interpolation units.

What these codes do is store a lookup table of the log-scaled
HFDB-He potential energy function [1] in the texture units of the GPU then
calls a simple kernel to do a bunch of lookups.  All the lookup
results are copy back to the host and compared with the exact HFDB-He
function.

I have successfully tested this on a Tesla C1060, GeForce GTX 480,
Tesla M2090, and an ATI Radeon 7970.

## CUDA ##

You should only need to edit the Makefile with the appropriate CUDA
architecture and the CUDA installation location. That is, change the
variables `CUDA_ARCH` and `CUDA_DIR` to reflect your system.

Compile by typing `make`

## OpenCL ##

You should only need to edit the Makefile with the OpenCL installation
location. That is, change the variable `OPENCL_DIR` to reflect your
system.

Compile by typing `make`

Note: I didn't make this code all that generic so it will have to be
run from the OpenCL directory.  That has nothing to do with utilizing
the hardware texture units, I was just lazy and used a relative path
for the kernel code.

## References ##
[1] R. A. Aziz, F. R. McCourt, and C. C. Wong, "A New Determination of the Ground State Interatomic Potential for He 2," _Molecular Physics_, vol. 61, no. 6, pp. 1487-1511, Aug. 1987

[2] NVIDIA CUDA Documentation, http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html
