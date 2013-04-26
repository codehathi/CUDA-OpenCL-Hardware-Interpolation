/******************************************************************************/
//  Author: Codehathi (codehathi@gmail.com)
//  File: interpolation.h
//  Date: 02 Sep 2012
//  Description:
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

/* Some helpful macros */

/* Prints the file and line number to stderr where an error occurs.           */
#define errprint(format,...) fprintf(stderr,"%s, function %s, line %d: " \
                                     format, __FILE__, __FUNCTION__, __LINE__, \
                                     ##__VA_ARGS__ )

#define safe_call_return(var)                                  \
  if(!(var))                                                   \
    {                                                          \
      errprint("%s failed\n", #var);                           \
      return false;                                            \
    }

#define alloc_and_test_return(var, type, size)          \
  safe_call_return((var = tcalloc(type, size)));

/* Does the typecasting for calloc */
#define tcalloc(type, num) (type*)calloc((num),sizeof(type))

#endif
