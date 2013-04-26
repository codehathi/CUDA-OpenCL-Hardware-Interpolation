
/******************************************************************************/
/* I'm too lazy to figure out how to include files inside OpenCL
   kernel code so I'm just writing all the table constants here  */

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

__kernel void fill_image(__write_only image2d_t image,
                         __global const float* lookup)
{
  if(get_global_id(0)<N_TEX_PTS)
    {
      float4 t = {0.0f,0.0f,0.0f,0.0f};
      t.w = lookup[get_global_id(0)];

      int2 coord;
      coord.x = get_global_id(0);
      coord.y = 0;

      write_imagef(image, coord, t);
    }
  mem_fence(CLK_GLOBAL_MEM_FENCE);
}

__constant sampler_t s0 = CLK_NORMALIZED_COORDS_FALSE |
  CLK_ADDRESS_CLAMP_TO_EDGE |
  CLK_FILTER_LINEAR;

float lookup_potential_energy(const float r2,
                              __read_only image2d_t potcurve)
{
  float x=log2(r2);

  x=0.5f + (x - XMIN) * TEX_MUL;

  float2 coord;
  coord.x = x;
  coord.y = -1;

  float4 temp = read_imagef(potcurve, s0, coord);
  return temp.w;
}

__kernel void calculate_values(__read_only image2d_t potcurve,
                               __global float* energies)
{
  if( get_global_id(0) == 0 )
    {
      int j=0;
      for(float i=2.0; i<1024; i+=.01, j++)
        {
          float en = lookup_potential_energy(i, potcurve);
          energies[j] = en;
        }
    }
}

