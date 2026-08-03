#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#include <math.h>
#include <omp.h>
#include <stdlib.h>
#include <string.h>

/* A test program for solving the 3D heat conduction problem
         du/dt = div ( kappa(x,y,z) grad u ) + f(x,y,z)
   where kappa(x,y,z) = 0.5+0.45*sin(pi*x)*sin(pi*y)*sin(pi*z).
   That is, the heat conduction coefficient is a variable field.
   An explicit time stepping scheme is used.
*/

static int should_write_txt(void) {
  const char *e = getenv("WRITE_TXT");
  if (!e) return 1;                 // default: write
  if (!strcmp(e, "0")) return 0;
  if (!strcasecmp(e, "false")) return 0;
  if (!strcasecmp(e, "no")) return 0;
  return 1;
}


void set_pointers (double*** u_ptr, int n)
{
  /* assuming u_ptr[0][0][0] is already pointing to the correct position */
  /* we also assume the same number of points in all three directions */
  int j, k;
  for (k=0; k<n; k++)
    {
      for (j=1; j<n; j++)
	u_ptr[k][j] = u_ptr[k][j-1] + n;
      if (k<n-1)
	u_ptr[k+1][0] = u_ptr[k][n-1] + n;
    }
}


void write_u_values(const char *fname, double ***u, int n)
{
  FILE *f = fopen(fname, "w");
  if (!f)
    {    
      perror("fopen"); 
      return; 
    }
  
  // Write one value per line in deterministic order (k, j, i)
  for (int k=0; k<n; k++)
    for (int j=0; j<n; j++)
      for (int i=0; i<n; i++)
	fprintf(f, "%.17g\n", u[k][j][i]);
  
  fclose(f);
  printf("Solution written to %s\n", fname);
}


int main (int nargs, char** args)
{
  int n;      /* number of points in each direction */
  double h;   /* grid spacing, same in all the directions */
  double ***u_old, ***u_new, ***rhs, ***kappa, *tmp_ptr;
  int i,j,k,num_time_steps;
  double t, T, dt, factor;
  
  if (nargs>1)
    n = atoi(args[1]);
  else
    n = 51;
  
  if (nargs>2)
    T = atof(args[2]);  // Now correctly reads the 2nd argument as a float // T = atoi(args[1]);
  else
    T = 1.0;
  
  h = 1.0/(n-1);
  double kappa_max = 0.95;
  //  dt = h*h/6.0;   /* a safe choice of dt when k(x,y,z)<1 */
  dt = h*h / (12.0 * kappa_max);       // Safety factor of 2
  //  double dt = h*h/20.0;            // dt = "delta t" (time step)
  factor = dt/h/h/2.0;
  
  u_old = (double***)malloc(n*sizeof(double**));
  u_new = (double***)malloc(n*sizeof(double**));
  rhs = (double***)malloc(n*sizeof(double**));
  kappa = (double***)malloc(n*sizeof(double**));
  
  for (k=0; k<n; k++)
    {
      u_old[k] = (double**)malloc(n*sizeof(double*));
      u_new[k] = (double**)malloc(n*sizeof(double*));
      rhs[k] = (double**)malloc(n*sizeof(double*));
      kappa[k] = (double**)malloc(n*sizeof(double*));
    }
  
  u_old[0][0] = (double*)malloc(n*n*n*sizeof(double));
  u_new[0][0] = (double*)malloc(n*n*n*sizeof(double));
  rhs[0][0] = (double*)malloc(n*n*n*sizeof(double));
  kappa[0][0] = (double*)malloc(n*n*n*sizeof(double));
  
  set_pointers(u_old, n);
  set_pointers(u_new, n);
  set_pointers(rhs, n);
  set_pointers(kappa, n);
  
  /* fill values of rhs */
#pragma omp parallel for default(shared) private(i,j) schedule(static)
  for (k=0; k<n; k++)
    for (j=0; j<n; j++)
      for (i=0; i<n; i++)
        rhs[k][j][i] = dt*(1.0+cos(M_PI*i*h)*cos(M_PI*j*h)*cos(M_PI*k*h));
  
  /* fill values of kappa */
#pragma omp parallel for default(shared) private(i,j) schedule(static)
  for (k=0; k<n; k++)
    for (j=0; j<n; j++)
      for (i=0; i<n; i++)
        kappa[k][j][i] = 0.5+0.45*sin(M_PI*i*h)*sin(M_PI*j*h)*sin(M_PI*k*h);
  
  /*
    for (k=0; k<n; k++)
    for (j=0; j<n; j++)
    for (i=0; i<n; i++) {
    u_old[k][j][i] = sin(M_PI*i*h)*sin(M_PI*j*h)*sin(M_PI*k*h);
    u_new[k][j][i] = 0.;
    }
  */
  
  /* fill initial values */
#pragma omp parallel for default(shared) private(i,j) schedule(static)
  // In heat3D_serial.c - FIXED initialization
  for (k=0; k<n; k++)
    for (j=0; j<n; j++)
      for (i=0; i<n; i++)
	{
	  // Force exact zeros at boundaries
	  if (i == 0 || i == n-1 || j == 0 || j == n-1 || k == 0 || k == n-1)
	    {
	      u_old[k][j][i] = 0.0;  // EXACT zero
	      u_new[k][j][i] = 0.0;
	    }
	  else
	    {
	      u_old[k][j][i] = sin(M_PI*i*h)*sin(M_PI*j*h)*sin(M_PI*k*h);
	      u_new[k][j][i] = 0.0;
	    }
	}


  
  /* main time loop */
  t = 0.;
  num_time_steps = 0;

  double start_time = omp_get_wtime(); 
#pragma omp parallel default(shared) private(i,j)
  {
    while (t<T)
      {
#pragma omp for private(i,j) schedule(static)
	for (k=1; k<n-1; k++)
	  for (j=1; j<n-1; j++)
	    for (i=1; i<n-1; i++)
	      //	   u_new[k][j][i] = u_old[k][j][i] + rhs[k][j][i] 
	      //	     +factor*((kappa[k][j][i+1]+kappa[k][j][i])*(u_old[k][j][i+1]-u_old[k][j][i])
	      //		      +(kappa[k][j][i]+kappa[k][j][i-1])*(u_old[k][j][i]-u_old[k][j][i-1])
	      //		      +(kappa[k][j+1][i]+kappa[k][j][i])*(u_old[k][j+1][i]-u_old[k][j][i])
	      //		      +(kappa[k][j][i]+kappa[k][j-1][i])*(u_old[k][j][i]-u_old[k][j-1][i])
	      //		      +(kappa[k+1][j][i]+kappa[k][j][i])*(u_old[k+1][j][i]-u_old[k][j][i])
	      //		      +(kappa[k][j][i]+kappa[k-1][j][i])*(u_old[k][j][i]-u_old[k-1][j][i]));
	      u_new[k][j][i] = u_old[k][j][i] + rhs[k][j][i] 
		+ factor * (
			    (kappa[k][j][i+1]+kappa[k][j][i])*(u_old[k][j][i+1]-u_old[k][j][i])
			    - (kappa[k][j][i]+kappa[k][j][i-1])*(u_old[k][j][i]-u_old[k][j][i-1])
			    + (kappa[k][j+1][i]+kappa[k][j][i])*(u_old[k][j+1][i]-u_old[k][j][i])
			    - (kappa[k][j][i]+kappa[k][j-1][i])*(u_old[k][j][i]-u_old[k][j-1][i])
			    + (kappa[k+1][j][i]+kappa[k][j][i])*(u_old[k+1][j][i]-u_old[k][j][i])
			    - (kappa[k][j][i]+kappa[k-1][j][i])*(u_old[k][j][i]-u_old[k-1][j][i]));
#pragma omp single
	{
	  /* pointer swap */
	  tmp_ptr = u_new[0][0];
	  u_new[0][0] = u_old[0][0];
	  u_old[0][0] = tmp_ptr;
	  set_pointers (u_new, n);
	  set_pointers (u_old, n);
	  
	  num_time_steps++;
	  t += dt;
	}
      }
  }

#pragma omp barrier
  
  double end_time = omp_get_wtime();
  double elapsed_time = end_time - start_time;
  
  printf("%d timesteps complete\n", num_time_steps);
  printf("Time = %.6f seconds\n", elapsed_time);
  
  printf("h=%g, dt=%g, T=%g, # time steps=%d\n",h,dt,T,num_time_steps); 
  
  /* Write solution to file */
  if (should_write_txt()) {
  write_u_values("u_serial.txt", u_old, n);
  }
  
  free (u_new[0][0]);
  free (u_old[0][0]);
  free (rhs[0][0]);
  free (kappa[0][0]);
  
  for (k=0; k<n; k++)
    {
      free (u_new[k]);
      free (u_old[k]);
      free (rhs[k]);
      free (kappa[k]);
    }
  
  free(u_new);
  free(u_old);
  free(rhs);
  free(kappa);
  
  return 0;
}
