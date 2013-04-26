/******************************************************************************/
//  Author: Codehathi (codehathi@gmail.com)
//  File: interpolation.c
//  Date: 02 Sep 2012
//  Description: This program does the hardware linear interpolation in OpenCL
//               in order to create a lookup table for the HFDB-He potential
//               energy function.
//
/******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include <string.h>
#include <CL/cl.h>
#include "interpolation.h"

bool safe_opencl_call(cl_int err)
{
  if(err != CL_SUCCESS)
    {
      switch(err)
        {
        case CL_SUCCESS:
          errprint("OpenCL Error: No Error.\n"); break;
        case CL_INVALID_MEM_OBJECT:
          errprint("OpenCL Error: Invalid memory object.\n"); break;
        case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:
          errprint("OpenCL Error: Invalid image format descriptor.\n"); break;
        case CL_IMAGE_FORMAT_NOT_SUPPORTED:
          errprint("OpenCL Error: Image format not supported.\n"); break;
        case CL_INVALID_IMAGE_SIZE:
          errprint("OpenCL Error: Invalid image size.\n"); break;
        case CL_INVALID_ARG_INDEX:
          errprint("OpenCL Error: Invalid argument index for this kernel.\n"); break;
        case CL_INVALID_ARG_VALUE:
          errprint("OpenCL Error: Invalid argument value.\n"); break;
        case CL_INVALID_SAMPLER:
          errprint("OpenCL Error: Invalid sampler.\n"); break;
        case CL_INVALID_ARG_SIZE:
          errprint("OpenCL Error: Invalid argument size.\n"); break;
        case CL_INVALID_BUFFER_SIZE:
          errprint("OpenCL Error: Invalid buffer size.\n"); break;
        case CL_INVALID_HOST_PTR:
          errprint("OpenCL Error: Invalid host pointer.\n"); break;
        case CL_INVALID_DEVICE:
          errprint("OpenCL Error: Invalid device.\n"); break;
        case CL_INVALID_VALUE:
          errprint("OpenCL Error: Invalid value.\n"); break;
        case CL_INVALID_CONTEXT:
          errprint("OpenCL Error: Invalid Context.\n"); break;
        case CL_INVALID_KERNEL:
          errprint("OpenCL Error: Invalid kernel.\n"); break;
        case CL_INVALID_PROGRAM:
          errprint("OpenCL Error: Invalid program object.\n"); break;
        case CL_INVALID_BINARY:
          errprint("OpenCL Error: Invalid program binary.\n"); break;
        case CL_INVALID_OPERATION:
          errprint("OpenCL Error: Invalid operation.\n"); break;
        case CL_INVALID_BUILD_OPTIONS:
          errprint("OpenCL Error: Invalid build options.\n"); break;
        case CL_INVALID_PROGRAM_EXECUTABLE:
          errprint("OpenCL Error: Invalid executable.\n"); break;
        case CL_INVALID_COMMAND_QUEUE:
          errprint("OpenCL Error: Invalid command queue.\n"); break;
        case CL_INVALID_KERNEL_ARGS:
          errprint("OpenCL Error: Invalid kernel arguments.\n"); break;
        case CL_INVALID_WORK_DIMENSION:
          errprint("OpenCL Error: Invalid work dimension.\n"); break;
        case CL_INVALID_WORK_GROUP_SIZE:
          errprint("OpenCL Error: Invalid work group size.\n"); break;
        case CL_INVALID_WORK_ITEM_SIZE:
          errprint("OpenCL Error: Invalid work item size.\n"); break;
        case CL_INVALID_GLOBAL_OFFSET:
          errprint("OpenCL Error: Invalid global offset (should be NULL).\n"); break;
        case CL_OUT_OF_RESOURCES:
          errprint("OpenCL Error: Insufficient resources.\n"); break;
        case CL_MEM_OBJECT_ALLOCATION_FAILURE:
          errprint("OpenCL Error: Could not allocate mem object.\n"); break;
        case CL_INVALID_EVENT_WAIT_LIST:
          errprint("OpenCL Error: Invalid event wait list.\n"); break;
        case CL_OUT_OF_HOST_MEMORY:
          errprint("OpenCL Error: Out of memory on host.\n"); break;
        case CL_INVALID_KERNEL_NAME:
          errprint("OpenCL Error: Invalid kernel name.\n"); break;
        case CL_INVALID_KERNEL_DEFINITION:
          errprint("OpenCL Error: Invalid kernel definition.\n"); break;
        case CL_BUILD_PROGRAM_FAILURE:
          errprint("OpenCL Error: Failed to build program.\n"); break;
        case CL_MAP_FAILURE:
          errprint("OpenCL Error: Failed to map buffer/image\n"); break;
        case CL_PROFILING_INFO_NOT_AVAILABLE:
          errprint("OpenCL Error: Profiling info not available.\n"); break;
        case -1001: //This is CL_PLATFORM_NOT_FOUND_KHR
          errprint("OpenCL Error: No platforms found. (Did you put ICD files in /etc/OpenCL?)\n"); break;
        default:
          errprint("OpenCL Error: Unknown error.\n"); break;
        }
      return false;
    }
  return true;
}

/* Reads the given kernel file */
char* read_kernel(const char* filename, const char* preamble, size_t* length)
{
  // locals
  FILE* stream = NULL;
  size_t source_length;

  // open the OpenCL source code file
  stream = fopen(filename, "rb");
  if(stream == 0)
    {
      return NULL;
    }

  size_t preamble_length = strlen(preamble);

  // get the length of the source code
  fseek(stream, 0, SEEK_END);
  source_length = ftell(stream);
  fseek(stream, 0, SEEK_SET);

  // allocate a buffer for the source code string and read it in
  char* string = (char *)malloc(source_length + preamble_length + 1);
  memcpy(string, preamble, preamble_length);
  if (fread((string) + preamble_length, source_length, 1, stream) != 1)
    {
      fclose(stream);
      free(string);
      return 0;
    }

  // close the file and return the total length of the combined
  // (preamble + source) string
  fclose(stream);
  if(length != 0)
    {
      *length = source_length + preamble_length;
    }
  string[source_length + preamble_length] = '\0';

  return string;
}

/* HFDB-He function */
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

int main(int argc, char* argv[])
{
  cl_platform_id platform;
  cl_device_id device;
  cl_context context;
  cl_command_queue commandQueue;
  cl_mem potcurve;
  cl_mem write_to;
  cl_program program;
  cl_mem d_lookup;
  cl_kernel kernel;

  float* energies_texture = NULL;
  alloc_and_test_return(energies_texture, float, 1000000);

  size_t source_size = 0;
  const char* sourcemain = read_kernel("src/kernel.cl", "", &source_size);
  cl_int err;
  err = clGetPlatformIDs(1, &platform, NULL);
  safe_call_return(safe_opencl_call(err));
  err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
  safe_call_return(safe_opencl_call(err));
  context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
  safe_call_return(safe_opencl_call(err));
  commandQueue = clCreateCommandQueue(context, device, 0, &err);
  safe_call_return(safe_opencl_call(err));
  program = clCreateProgramWithSource(context, 1, &sourcemain, 0, &err);
  safe_call_return(safe_opencl_call(err));
  err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);

  size_t log_size;
  clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
  if(log_size != 0)
    {
      char build_log[log_size];
      clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, build_log, NULL);

      printf("%s\n", build_log);
    }
  safe_call_return(safe_opencl_call(err));

  float* lookup = NULL;
  alloc_and_test_return(lookup, float, N_TEX_PTS);
  for (int n=0; n<N_TEX_PTS; n++)
    {
      float xx = (XMAX-XMIN)/( (float) (N_TEX_PTS-1) )*( (float) n ) + XMIN;
      float rr = exp(xx*log(2.0f)); // this is equivalent to (2.0f)^xx
      lookup[n] = hfdbhe(rr);
    }

  /* Create the kernel */
  cl_kernel fillkernel = clCreateKernel(program, "fill_image", &err);

  /* Set up the potential energy image */
  cl_image_format imageFormat = {CL_A, CL_FLOAT};
  potcurve = clCreateImage2D(context,
                             CL_MEM_READ_WRITE,
                             &imageFormat,
                             N_TEX_PTS,
                             1,
                             0,
                             NULL,
                             &err);
  safe_call_return(safe_opencl_call(err));

  d_lookup      = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float)*N_TEX_PTS, lookup, &err);
  safe_call_return(safe_opencl_call(err));

  err = clSetKernelArg(fillkernel, 0, sizeof(cl_mem), &potcurve);
  safe_call_return(safe_opencl_call(err));
  err = clSetKernelArg(fillkernel, 1, sizeof(cl_mem), &d_lookup);
  safe_call_return(safe_opencl_call(err));

  size_t global = N_TEX_PTS;
  size_t local = 64;

  err = clEnqueueNDRangeKernel(commandQueue,
                               fillkernel,
                               1,
                               NULL,
                               &global,
                               &local,
                               0,
                               NULL,
                               NULL);
  safe_call_return(safe_opencl_call(err));
  err = clFinish(commandQueue);
  safe_call_return(safe_opencl_call(err));

  write_to      = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float)*1000000, energies_texture, &err);
  safe_call_return(safe_opencl_call(err));

  kernel = clCreateKernel(program, "calculate_values", &err);
  err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &potcurve);
  safe_call_return(safe_opencl_call(err));
  err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &write_to);
  safe_call_return(safe_opencl_call(err));

  global = 1;
  local = 1;
  err = clEnqueueNDRangeKernel(commandQueue,
                               kernel,
                               1,
                               NULL,
                               &global,
                               &local,
                               0,
                               NULL,
                               NULL);

  err = clEnqueueReadBuffer(commandQueue,
                            write_to,
                            CL_TRUE,
                            0,
                            sizeof(float)*1000000,
                            energies_texture,
                            0,
                            NULL,
                            NULL);

  for(float i=2.0, j=0; i<1024; i+=.01,j++)
    {
      printf("%f\t%e\t%e\t%e\n",i, hfdbhe(i),energies_texture[(int)j],
             (energies_texture[(int)j]-hfdbhe(i))/hfdbhe(i));
    }

  return 0;
}
