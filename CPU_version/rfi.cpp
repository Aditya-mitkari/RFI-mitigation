/////////////////////////////////////////////////////////
// Auther: Aditya Mitkari
// Date: 4/5/19
//
// Description: Serial code for RFI mitigation
// Input: file / array of frequecy values for given DM
// Ouput: Frequecy values free of RFI
/////////////////////////////////////////////////////////

#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#if (_OPENACC)
#include <openacc.h>
#include <accelmath.h>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#else
#include <math.h>
#endif
//#include "headers/params.h"

//Multidimensional array class to handle 2D arrays
#include "arrayMD.h"

//Setting N in a way such that we do not need to change anything later in terms of number of channels launched
#if defined(PRODUCTION_VERSION)
#define N 1
#else
#define N 50
#endif

inline float rand_local()
{
#if (_OPENACC)
//  unsigned int seed = 1234;
//  curandState s;
//  curand_init(seed, 0, 0, &s);
//  return(curand_uniform(&s));
  return(0.1435);

#else
  return(0.1435);
//  return(rand());
#endif
}

#if(_OPENACC)
inline void allocate_ON_Device(Array2D<float> stage)
{
#pragma acc enter data copyin(stage[:1])
#pragma acc enter data copyin(stage.dptr[:stage.size])
}
#endif

void write_output_file(int nchans, int nsamp, int file_reducer, double orig_mean, double orig_var, float sigma_cut, double var_rescale, double mean_rescale, Array2D<float>& stage)
{
  //Writing the output to a file
	FILE *fp_mask = fopen ("masked_chans.txt", "w+");

	for( int c = 0; c < nchans; c++ )
	{
		for( int t = 0; t < (nsamp) / file_reducer; t++ )
		{
			fprintf(fp_mask, "%d ", (unsigned char)((stage(t,c)*orig_var)+orig_mean));
			fprintf( fp_mask, "%d ", (unsigned char)( ( stage(t,c) * var_rescale ) + mean_rescale ) );
		}

		fprintf(fp_mask, "\n");
	}
  fclose(fp_mask);
}

inline void reduce_orig_mean(Array2D<float>& stage, int nsamp, int nchans, double& orig_mean)
{
  double orig_mean_loc = orig_mean;
  for(int t = 0; t < nsamp; ++t)
	{
    for( int c = 0; c < nchans; c++ )
		{
			orig_mean_loc += stage(t,c);
    }
 	}
  orig_mean = orig_mean_loc;

	orig_mean /= ( nsamp * nchans );
}

void reduce_orig_var(Array2D<float>& stage, int nsamp, int nchans, double orig_mean, double& orig_var)
{
  double orig_var_loc = orig_var;
  double orig_mean_loc = orig_mean;
#if (_OPENACC)
//#pragma acc parallel loop gang vector  \
//  present(stage[:1]) \
//  reduction(+:orig_var_loc)
#endif
  for(int iter = 0; iter < nsamp*N; ++iter)
	{
    for( int c = 0; c < nchans; c++ )
		{
      int t = iter % nsamp;
      double store1 = stage(t,c) - orig_mean_loc;
			orig_var_loc += store1*store1;
		}
	}

//  printf("orig_var_loc = %f\n", orig_var_loc);

  orig_var = orig_var_loc;
	orig_var /= ( nsamp * nchans );
	orig_var = sqrt( orig_var );
}

void random_chan_1_2(float* random_chan_one, float* random_chan_two, int nsamp)
{
  for(int iter = 0; iter < nsamp*N; ++iter)
  {
    int t = iter % nsamp;
		float x1, x2, w, y1, y2;
		do {
			x1 = 2.0 * ( ( float )rand() / ( float )RAND_MAX ) - 1.0;
			x2 = 2.0 * ( ( float )rand() / ( float )RAND_MAX ) - 1.0;
			w = x1 * x1 + x2 * x2;
		} while( w >= 1.0 );

		w = sqrt( ( -2.0 * log( w ) ) / w);
		y1 = x1 * w;
		y2 = x2 * w;


		random_chan_one[ t ] = y1;
		random_chan_two[ t ] = y2;
	}
}

void random_spectra_1_2(float* random_spectra_one, float* random_spectra_two, int nchans)
{
	for(int c = 0; c < nchans; c++)
	{
		float x1, x2, w, y1, y2;
		do {
			x1 = 2.0 * ( ( float )rand() / ( float )RAND_MAX ) - 1.0;
			x2 = 2.0 * ( ( float )rand() / ( float )RAND_MAX ) - 1.0;
			w = x1 * x1 + x2 * x2;
		} while( w >= 1.0 );

		w = sqrt( ( -2.0 * log( w ) ) /  w );
		y1 = x1 * w;
		y2 = x2 * w;


		random_spectra_one[ c ] = y1;
		random_spectra_two[ c ] = y2;
	}

}

void channel_process(int *spectra_mask, Array2D<float>& stage, double *chan_mean, double *chan_var, int *chan_mask,
    int nsamp, int nchans, float *random_chan_one, float *random_chan_two)
{
	float	sigma_cut	= 2.0f;
  for( int c = 0; c < nchans; c++ )
	{
		int counter = 0;

	for( int t = 0; t < nsamp; t++ ) spectra_mask[ t ] = 1;

		int finish = 0;
		int rounds = 1;

		double old_mean = 0.0;
		double old_var = 0.0;

		while( finish == 0 )
		{
			counter = 0;
			chan_mean[ c ] = 0.0;
      for( int t = 0; t < ( nsamp ); t++ )
			{
				if( spectra_mask[ t ] == 1 )
				{
	        chan_mean[ c ] += stage(t,c);
					counter++;
				}
      }

			if( counter == 0 )
			{
				chan_mask[ c ] = 0;
				finish = 1;
				break;
			}
			chan_mean[ c ] /= ( counter );

			counter = 0;
			chan_var[ c ] = 0.0;
			for( int t = 0; t < ( nsamp ); t++)
			{
				if( spectra_mask[ t ] == 1 )
				{
					chan_var[ c ] += ( stage(t,c) - chan_mean[ c ] ) * ( stage(t,c) - chan_mean[ c ] );
					counter++;
				}
			}

			chan_var[ c ] /= ( counter );
			chan_var[ c ] = sqrt( chan_var[ c ] );

			if( ( chan_var[ c ] ) * 1000000.0 < 0.1 )
			{
				chan_mask[ c ] = 0;
				finish = 1;
				break;
			}

			for( int t = 0; t < ( nsamp ); t++ )
			{
				if( ( ( stage(t,c) - chan_mean[ c ] ) / chan_var[ c ] ) > sigma_cut || ( ( stage(t,c) - chan_mean[ c ] ) / chan_var[ c ] ) < -sigma_cut )
				{
					spectra_mask[ t ] = 0;
				}
				else
				{
					spectra_mask[ t ] = 1;
				}
			}

			if( fabs( chan_mean[ c ] - old_mean ) < 0.001 && fabs( chan_var[ c ] - old_var ) < 0.0001 && rounds > 1)
			{
				////printf("\n%d\t%d\t%.16lf\t%.16lf\t%.16lf\t%.16lf", c, rounds, (chan_mean[c]-old_mean), (chan_var[c]-old_var), chan_mean[c], chan_var[c]);
				finish = 1;
			}

			old_mean = chan_mean[ c ];
			old_var = chan_var[ c ];
			rounds++;
		}


		//printf( "\nChan mean, var: %lf %lf\n", chan_mean[ c ], chan_var[ c ] );


		if( chan_mask[ c ] != 0 )
		{
			for( int t = 0; t < ( nsamp ); t++ )
			{
				stage(t,c) = ( stage(t,c) - ( float )chan_mean[ c ] ) / ( float )chan_var[ c ];
			}
		}
		else
		{
			int perm_one = ( int )( ( ( float )rand() / ( float )RAND_MAX ) * nsamp );

			for( int t = 0; t < nsamp; t++ )
			{
				stage(t,c) = random_chan_one[ ( t + perm_one ) % nsamp ];
			}

			chan_mean[ c ] = 0.0;
			chan_var[ c ]  = 1.0;
			chan_mask[ c ] = 1;
		}
	}
}

void sample_process(int *chan_mask, double *spectra_mean, Array2D<float>& stage, int *spectra_mask,
    double *spectra_var, int nsamp, int nchans, float *random_spectra_one, float *random_spectra_two)
{
	float	sigma_cut	= 2.0f;

	for( int t = 0; t < nsamp; ++t )
	{
		int counter = nchans;
		bool finish = false;
		int rounds = 1;
		double old_mean = 0.0;
		double old_var = 0.0;
    double spectra_mean_t = spectra_mean[t];
    double spectra_var_t = spectra_var[t];

#if(_OPENACC)
//#pragma acc enter data \
//      copyin(chan_mask[:nchans], stage, stage.dptr[0:stage.size])
//
//#pragma acc parallel loop gang vector \
//    present(chan_mask)
#endif
		for( int c = 0; c < nchans; c++ )
		 chan_mask[ c ]=1;

		while( finish == false )
		{
			spectra_mean_t = 0.0;
			spectra_var_t  = 0.0;

#if(_OPENACC)
//#pragma acc parallel loop gang vector\
//      reduction(+:spectra_mean_t) \
//      present(stage, chan_mask)
#endif
			for( int c = 0; c < nchans; c++ )
			{
	        spectra_mean_t += stage(t,c) * chan_mask[c];
      }

#pragma acc serial
      {
        spectra_mean_t /= counter;
      }

#if(_OPENACC)
//#pragma acc parallel loop gang vector\
//      reduction(+:spectra_var_t) \
//      present(stage, chan_mask)
#endif
			for( int c = 0; c < nchans; c++ )
			{
        double store_value = ( stage(t,c) - spectra_mean_t) * chan_mask[c];
        spectra_var_t += store_value*store_value;
			}

#pragma acc serial
      {
			spectra_var_t /= counter;
			spectra_var_t = sqrt( spectra_var_t );
      }

      counter = 0;
#if(_OPENACC)
//#pragma acc parallel loop gang vector \
//      reduction(+:counter) \
//      present(stage, chan_mask)
#endif
      for( int c = 0; c < nchans; c++ )
      {
        float threshold = (stage(t,c) - spectra_mean_t) / spectra_var_t ;
        if( threshold > sigma_cut || threshold < -sigma_cut)
        {
          chan_mask[ c ] = 0;
        }
        else
        {
          counter++;
          chan_mask[ c ] = 1;
        }
      }

			if((!counter) && ( spectra_var_t < 1e-6) )
			{
				spectra_mask[ t ] = 0;
				finish = true;
				break;
			}

			if( fabs( spectra_mean_t - old_mean ) < 0.001 && fabs( spectra_var_t - old_var ) < 0.0001 && rounds > 1)
			{
				finish = true;
			}

			old_mean = spectra_mean_t;
			old_var = spectra_var_t;
			rounds++;
		}

//#pragma acc exit data copyout(chan_mask[:nchans])

    spectra_mean[t] = spectra_mean_t;
    spectra_var[t] = spectra_var_t;

		if( spectra_mask[ t ] != 0)
		{
			for( int c = 0; c < nchans; c++ )
			{
				stage(t,c) = ( stage(t,c) - (float)spectra_mean[ t ] ) / (float)spectra_var[ t ];
			}
		}
		else
		{
			int perm_one = (int)( ( (float) rand_local() / (float)RAND_MAX ) * nchans);

			for( int c = 0; c < nchans; c++ )
			{
				stage(t,c) = random_spectra_one[ ( c + perm_one ) % nchans ];
			}

			spectra_mean[ t ] = 0.0;
			spectra_var[ t ]  = 1.0;
			spectra_mask[ t ] = 1;
		}
	}
}

void while_loop1(Array2D<float>& stage, int *chan_mask, double *chan_mean, double *chan_var,
    int nsamp, int nchans, double& mean_rescale, double& var_rescale,
    float *random_chan_one, float *random_chan_two)
{
	float	sigma_cut	= 2.0f;
	// Find the mean and SD of the mean and SD...
	int finish = 0;
	int rounds = 1;
	int counter = 0;
	double mean_of_mean = 0.0;
	double var_of_mean  = 0.0;
	double mean_of_var  = 0.0;
	double var_of_var   = 0.0;

	double old_mean_of_mean = 0.0;
	double old_var_of_mean  = 0.0;
	double old_mean_of_var  = 0.0;
	double old_var_of_var   = 0.0;

	for( int c = 0; c < nchans; c++ ) chan_mask[ c ] = 1;
	while(finish == 0)
	{
		mean_of_mean = 0.0;
		counter = 0;

		for( int c = 0; c < nchans; c++ )
		{
			if( chan_mask[ c ] == 1 )
			{
				mean_of_mean += chan_mean[ c ];
				counter++;
			}
		}

		//printf("mm is %lf\n",mean_of_mean );

		mean_of_mean /= counter;

		var_of_mean = 0.0;
		counter = 0;

		for( int c = 0; c < nchans; c++ )
		{
			if( chan_mask[ c ] == 1 )
			{
				var_of_mean += ( chan_mean[ c ] - mean_of_mean ) * ( chan_mean[ c ] - mean_of_mean );
				counter++;
			}
		}

		//printf( "\nvar_of_mean %lf\n", var_of_var );
    //printf( "\ncounter %u\n", counter );

		var_of_mean /= ( counter );
		var_of_mean = sqrt( var_of_mean );

		mean_of_var = 0.0;
		counter = 0;

		for( int c = 0; c < nchans; c++ )
		{
			if( chan_mask[ c ] == 1 )
			{
				mean_of_var += chan_var[ c ];
				counter++;
			}
		}

		//printf("\nmean_of_var %lf\n",mean_of_var );
    //printf("\ncounter %u\n",counter );

		mean_of_var /= counter;

		var_of_var = 0.0;
		counter = 0;

		for( int c = 0; c < nchans; c++ )
		{
			if( chan_mask[ c ] == 1 )
			{
				var_of_var += ( chan_var[ c ] - mean_of_var ) * ( chan_var[ c ] - mean_of_var);
				counter++;
			}
		}

		//printf("\nvar_of_var %lf\n",var_of_var );
    //printf("\ncounter %u\n",counter);

		var_of_var /= (counter);
		var_of_var = sqrt( var_of_var );

		for( int c = 0; c < nchans; c++ )
			if( fabs( chan_mean[ c ] - mean_of_mean ) / var_of_mean > sigma_cut || fabs( chan_var[ c ] - mean_of_var ) / var_of_var > sigma_cut )
		  	chan_mask[ c ] = 0;

		if(fabs(mean_of_mean - old_mean_of_mean)   < 0.001 &&
		   fabs(var_of_mean  - old_var_of_mean )   < 0.001 &&
		   fabs(mean_of_var  - old_mean_of_var )   < 0.001 &&
		   fabs(var_of_var   - old_var_of_var  )   < 0.001)
			 {

				 finish = 1;
			 }

		old_mean_of_mean = mean_of_mean;
		old_var_of_mean  = var_of_mean;
		old_mean_of_var  = mean_of_var;
		old_var_of_var   = var_of_var;

		rounds++;
	}

  //Save mean_rescal and var_rescale for use later in the main rfi loop
	mean_rescale = mean_of_mean;
	var_rescale  = mean_of_var;

	float clipping_constant = 0.0;
	for( int c = 0; c < nchans; c++ ) clipping_constant += chan_mask[ c ];
	clipping_constant = ( nchans - clipping_constant ) / nchans;
	//printf("\n clipping_constant is %f\n",clipping_constant );
	clipping_constant = sqrt( -2.0 * log( clipping_constant * 2.506628275 ) );
	//printf("This This %f\n",clipping_constant );

	// Perform channel replacement
	for( int c = 0; c < nchans; c++ )
	{
		if( fabs( ( chan_mean[ c ] - mean_of_mean ) / var_of_mean ) > clipping_constant && fabs( ( chan_var[ c ] - mean_of_var ) / var_of_var ) > clipping_constant )
		{
			////printf("\nReplacing Channel %d %lf %lf", c, chan_mean[c], chan_var[c]);
			int perm_one = (int)( ( (float)rand_local() / (float)RAND_MAX ) * nsamp );

			for( int t = 0; t < (nsamp); t++ )
			{
				stage(t,c) = random_chan_two[ ( t + perm_one ) % nsamp ];
			}
		}
	}
}

void while_loop2(Array2D<float>& stage, int *spectra_mask, double *spectra_mean, double *spectra_var, int nsamp,
    int nchans, float *random_spectra_one, float *random_spectra_two)
{
	float	sigma_cut	= 2.0f;
	float clipping_constant = 0.0;

	int finish = 0;
	int rounds = 1;
	int counter = 0;

	int mean_of_mean = 0.0;
	int var_of_mean  = 0.0;
	int mean_of_var  = 0.0;
	int var_of_var   = 0.0;

	int old_mean_of_mean = 0.0;
	int old_var_of_mean  = 0.0;
	int old_mean_of_var  = 0.0;
	int old_var_of_var   = 0.0;

	for( int t = 0; t < (nsamp); t++ ) spectra_mask[ t ] = 1;
	while( finish == 0 )
	{
		mean_of_mean = 0.0;
		counter = 0;

		for( int t = 0; t < (nsamp); t++ )
		{
			if( spectra_mask[ t ] == 1 )
			{
				mean_of_mean += spectra_mean[ t ];
				counter++;
			}
		}

		mean_of_mean /= counter;

		var_of_mean = 0.0;
		counter = 0;

		for( int t = 0; t < (nsamp); t++ )
		{
			if( spectra_mask[ t ] == 1 )
			{
				var_of_mean += ( spectra_mean[ t ] - mean_of_mean ) * ( spectra_mean[ t ]- mean_of_mean );
				counter++;
			}
		}

		var_of_mean /= (counter);
		var_of_mean = sqrt( var_of_mean );

		mean_of_var = 0.0;
		counter = 0;

		for( int t = 0; t < (nsamp); t++ )
		{
			if( spectra_mask[ t ] == 1 )
			{
				mean_of_var += spectra_var[ t ];
				counter++;
			}
		}

		mean_of_var /= counter;

		var_of_var = 0.0;
		counter = 0;

		for( int t = 0; t < (nsamp); t++ )
		{
			if( spectra_mask[ t ] == 1 )
			{
				var_of_var += ( spectra_var[ t ] - mean_of_var ) * ( spectra_var[ t ] - mean_of_var );
				counter++;
			}
		}

		var_of_var /= (counter);
		var_of_var = sqrt( var_of_var );

		for( int t = 0; t < (nsamp); t++) if( fabs( spectra_mean[ t ] - mean_of_mean ) / var_of_mean > sigma_cut || fabs( spectra_var[ t ] - mean_of_var ) / var_of_var > sigma_cut ) spectra_mask[ t ] = 0;

		if(fabs(mean_of_mean - old_mean_of_mean)   < 0.001 &&
		   fabs(var_of_mean  - old_var_of_mean )   < 0.001 &&
		   fabs(mean_of_var  - old_mean_of_var )   < 0.001 &&
		   fabs(var_of_var   - old_var_of_var  )   < 0.001)
			 {
				 finish = 1;
			 }

		old_mean_of_mean = mean_of_mean;
		old_var_of_mean  = var_of_mean;
		old_mean_of_var  = mean_of_var;
		old_var_of_var   = var_of_var;

		rounds++;
	}

	//printf("\n0 %lf %lf", mean_of_mean, var_of_mean);
	//printf("\n0 %lf %lf", mean_of_var,  var_of_var);

	for( int t = 0; t < nsamp; t++ ) clipping_constant += spectra_mask[ t ];
	clipping_constant = ( nsamp - clipping_constant ) / nsamp;
	clipping_constant = sqrt( -2.0 * log( clipping_constant * 2.506628275 ) );

	// Perform spectral replacement
	for( int t = 0; t < (nsamp); t++ )
	{
	    if( fabs( ( spectra_mean[ t ] - mean_of_mean ) / var_of_mean ) > clipping_constant && fabs( ( spectra_var[ t ] - mean_of_var ) / var_of_var ) > clipping_constant )
			{
				////printf("\nReplacing Spectral %d %lf %lf", t, spectra_mean[t], spectra_var[t]);
				int perm_one = (int)( ( rand_local() / (float)RAND_MAX) * nchans );
				for( int c = 0; c < nchans; c++ )
				{
					stage(t,c) = random_spectra_two[ ( c + perm_one ) % nchans ];
				}
     }
	}

}

//Writing a transpose from stage to ip_buffer routine
inline void transpose_stage_ip(Array2D<float>& stage, float *input_buffer, int nsamp, int nchans)
{
  for(int t = 0; t < nsamp; ++t)
	{
    for( int c = 0; c < nchans; c++ )
		{
			stage(t,c) = ( float ) ( input_buffer )[ c  + ( size_t )nchans * t ];
		}
	}
}



//This is the debug version of rfi that loads less number of samples but iterates over the samples multiple times
void rfi_debug(int nsamp, int nchans, float **input_buffer, double& elapsedTimer)
{
	int file_reducer = 1;
	float	sigma_cut	= 2.0f;
	double orig_mean = 0.0;
	double orig_var=0.0;

	float *random_chan_one = ( float* )malloc( nsamp * sizeof( float ) );
	float *random_chan_two = ( float* )malloc( nsamp * sizeof( float ) );

	float *random_spectra_one = ( float* )malloc( nchans * sizeof( float ) );
	float *random_spectra_two = ( float* )malloc( nchans * sizeof( float ) );

	// Allocate working arrays

  Array2D<float> stage(nsamp,nchans);

	int *chan_mask = ( int* )malloc( nchans * sizeof( int ) );
  for( int c = 0; c < nchans; c++ )
   chan_mask[ c ]=1;

	int *spectra_mask = ( int* )malloc( nsamp * sizeof( int ) );
	//for( int t = 0; t < nsamp; t++ ) spectra_mask[ t ] = 1;
 for( int t = 0; t < nsamp; t++ )
   spectra_mask[ t ]=1;

	double *chan_mean = ( double* )malloc( nchans * sizeof( double ) );
  memset(chan_mean, 0.0, nchans*sizeof(double));

	double *chan_var = ( double* )malloc( nsamp * sizeof( double ) );
  memset(chan_var, 0.0, nchans*sizeof(double));

	double *spectra_mean = ( double* )malloc( nsamp * sizeof( double ) );
  memset(spectra_mean, 0.0, nsamp*sizeof(double));

	double *spectra_var = ( double* )malloc( nsamp * sizeof( double ) );
  memset(spectra_var, 0.0, nsamp*sizeof(double));

	// Random Vectors
  timeval startTimer, endTimer;
  gettimeofday(&startTimer, NULL);

  //Transpose from input buffer to stage
  transpose_stage_ip(stage, *input_buffer, nsamp, nchans);

  reduce_orig_mean(stage, nsamp, nchans, orig_mean);

  //reduce into orig_var
  reduce_orig_var(stage, nsamp, nchans, orig_mean, orig_var);

  //Fill random_chan_one and random_chan_two
  random_chan_1_2(random_chan_one, random_chan_two, nsamp);

  //Fill random_spectra and random_spectra_two
  random_spectra_1_2(random_spectra_one, random_spectra_two, nchans);

	// Find the BLN and try to flatten the input data per channel (remove non-stationary component).

  channel_process(spectra_mask, stage, chan_mean, chan_var, chan_mask, nsamp, nchans, random_chan_one, random_chan_two);

  sample_process( chan_mask, spectra_mean, stage, spectra_mask, spectra_var, nsamp, nchans, random_spectra_one, random_spectra_two);

  //First while loop
  double mean_rescale = 0.0, var_rescale = 0.0;
  while_loop1(stage, chan_mask, chan_mean, chan_var, nsamp, nchans, mean_rescale, var_rescale,
      random_chan_one, random_chan_two);


  while_loop2(stage, spectra_mask, spectra_mean, spectra_var, nsamp, nchans,
      random_spectra_one, random_spectra_two);


  gettimeofday(&endTimer, NULL);
  elapsedTimer = (endTimer.tv_sec - startTimer.tv_sec) + 1e-6 * (endTimer.tv_usec - startTimer.tv_usec);

	for( int c = 0; c < nchans; c++ )
	{
		for( int t = 0; t < (nsamp); t++ )
		{
			//(*input_buffer)[c  + (size_t)nchans * t] = (unsigned char) ((stage[c * (size_t)nsamp + t]*orig_var)+orig_mean);
			(*input_buffer)[ c  + (size_t)nchans * t ] = (unsigned char) ( ( stage(t,c) * var_rescale ) + mean_rescale );
		}
	}

  //Transerred the writing of output file to a different routine
  write_output_file(nchans, nsamp, file_reducer, orig_mean, orig_var, sigma_cut, var_rescale, mean_rescale, stage);

	//printf("\n%lf %lf", mean_rescale / orig_mean, var_rescale / orig_var);


	free(chan_mask);
	free(spectra_mask);
	free(chan_mean);
	free(chan_var);
	free(spectra_mean);
	free(spectra_var);
//	free(stage);
}

/******************************************************************************************************************************/

//void rfi(int nsamp, int nchans, unsigned short **input_buffer)
//{
//	int file_reducer = 1;
//	float	sigma_cut	= 2.0f;
//
//	float *stage = ( float* )malloc( ( size_t ) nsamp * ( size_t )nchans * sizeof( float ) );
//
//	int cn = 0;
//	for( int c = 0; c < nchans; c++ )
//	{
//		for( int t = 0; t < nsamp; t++ )
//		{
//			stage[ c * ( size_t )nsamp + t ] = ( float ) ( *input_buffer )[ c  + ( size_t )nchans * t ];
//		}
//	}
//
//
//	// ~~~ RFI Correct ~~~ //
//
//	double orig_mean = 0.0;
//	double orig_var=0.0;
//
//	// Find the mean and SD of the input data (we'll use this to rescale the data at the end of the process.
//
//  for( int c = 0; c < nchans; c++ )
//	{
// 	        for( int t = 0; t < ( nsamp ); t++)
//						orig_mean+=stage[ c * ( size_t )nsamp + t ];
// 	}
//
//	orig_mean /= ( nsamp * nchans );
//
//  for( int c = 0; c < nchans; c++ )
//	{
//		for( int t = 0; t < ( nsamp ); t++ )
//		{
//			orig_var += ( stage[ c * ( size_t )nsamp + t ] - orig_mean ) * ( stage[ c * ( size_t )nsamp + t ] - orig_mean );
//		}
//	}
//	orig_var /= ( nsamp * nchans );
//	orig_var = sqrt( orig_var );
//
//	//printf( "orig_mean %f\n", orig_mean );
//	//printf( "orig_var %f\n", orig_var );
//
//	// Random Vectors
//
//	float *random_chan_one = ( float* )malloc( nsamp * sizeof( float ) );
//	float *random_chan_two = ( float* )malloc( nsamp * sizeof( float ) );
//
//	for( int t = 0; t < nsamp; t++ )
//	{
//		float x1, x2, w, y1, y2;
//		do {
//			x1 = 2.0 * ( ( float )rand() / ( float )RAND_MAX ) - 1.0;
//			x2 = 2.0 * ( ( float )rand() / ( float )RAND_MAX ) - 1.0;
//			w = x1 * x1 + x2 * x2;
//		} while( w >= 1.0 );
//
//		w = sqrt( ( -2.0 * log( w ) ) / w);
//		y1 = x1 * w;
//		y2 = x2 * w;
//
//
//		random_chan_one[ t ] = y1;
//		random_chan_two[ t ] = y2;
//	}
//
//	float *random_spectra_one = ( float* )malloc( nchans * sizeof( float ) );
//	float *random_spectra_two = ( float* )malloc( nchans * sizeof( float ) );
//
//	for(int c = 0; c < nchans; c++)
//	{
//		float x1, x2, w, y1, y2;
//		do {
//			x1 = 2.0 * ( ( float )rand() / ( float )RAND_MAX ) - 1.0;
//			x2 = 2.0 * ( ( float )rand() / ( float )RAND_MAX ) - 1.0;
//			w = x1 * x1 + x2 * x2;
//		} while( w >= 1.0 );
//
//		w = sqrt( ( -2.0 * log( w ) ) /  w );
//		y1 = x1 * w;
//		y2 = x2 * w;
//
//
//		random_spectra_one[ c ] = y1;
//		random_spectra_two[ c ] = y2;
//	}
//
//	// Allocate working arrays
//
//	int *chan_mask = ( int* )malloc( nchans * sizeof( int ) );
//	for( int c = 0; c < nchans; c++ ) chan_mask[ c ] = 1;
//
//	int *spectra_mask = ( int* )malloc( nsamp * sizeof( int ) );
//	for( int t = 0; t < nsamp; t++ ) spectra_mask[ t ] = 1;
//
//	double *chan_mean = ( double* )malloc( nchans * sizeof( double ) );
//  for( int c = 0; c < nchans; c++ ) chan_mean[ c ] = 0.0;
//
//	double *chan_var = ( double* )malloc( nsamp * sizeof( double ) );
//  for( int c = 0; c < nchans; c++ ) chan_var[ c ] = 0.0;
//
//	double *spectra_mean = ( double* )malloc( nsamp * sizeof( double ) );
//  for( int t = 0; t < nsamp; t++ ) spectra_mean[ t ] = 0.0;
//
//	double *spectra_var = ( double* )malloc( nsamp * sizeof( double ) );
//  for( int t = 0; t < nsamp; t++ ) spectra_var[ t ] = 0.0;
//
//	// Find the BLN and try to flatten the input data per channel (remove non-stationary component).
//
//  for( int c = 0; c < nchans; c++ )
//	{
//		int counter = 0;
//
//		for( int t = 0; t < nsamp; t++ ) spectra_mask[ t ] = 1;
//
//		int finish = 0;
//		int rounds = 1;
//
//		double old_mean = 0.0;
//		double old_var = 0.0;
//
//		while( finish == 0 )
//		{
//			counter = 0;
//			chan_mean[ c ] = 0.0;
//      for( int t = 0; t < ( nsamp ); t++ )
//			{
//				if( spectra_mask[ t ] == 1 )
//				{
//	        chan_mean[ c ] += stage[ c * ( size_t )nsamp + t ];
//					counter++;
//				}
//      }
//
//			//printf( "\nchan_mean %lf\n", chan_mean[ c ] );
//			//printf( "\ncounter%u\n", counter );
//
//			if( counter == 0 )
//			{
//				//printf( "\nCounter zero, Channel %d", c );
//				chan_mask[ c ] = 0;
//				finish = 1;
//				break;
//			}
//			chan_mean[ c ] /= ( counter );
//
//			counter = 0;
//			chan_var[ c ] = 0.0;
//			for( int t = 0; t < ( nsamp ); t++)
//			{
//				if( spectra_mask[ t ] == 1 )
//				{
//					chan_var[ c ] += ( stage[ c * (size_t)nsamp + t ] - chan_mean[ c ] ) * ( stage[ c * (size_t)nsamp + t ] - chan_mean[ c ] );
//					counter++;
//				}
//			}
//
//			//printf( "\nchan_var %lf\n", chan_var[c] );
//			//printf( "\ncounter %u\n", counter );
//
//			chan_var[ c ] /= ( counter );
//			chan_var[ c ] = sqrt( chan_var[ c ] );
//
//			if( ( chan_var[ c ] ) * 1000000.0 < 0.1 )
//			{
//				////printf("\nVarience zero, Channel %d %d %lf %.16lf\n", c, rounds, chan_mean[c], chan_var[c] );
//				chan_mask[ c ] = 0;
//				finish = 1;
//				break;
//			}
//
//			for( int t = 0; t < ( nsamp ); t++ )
//			{
//				if( ( ( stage[ c * ( size_t )nsamp + t ] - chan_mean[ c ] ) / chan_var[ c ] ) > sigma_cut || ( ( stage[ c * ( size_t )nsamp + t ] - chan_mean[ c ] ) / chan_var[ c ] ) < -sigma_cut )
//				{
//					spectra_mask[ t ] = 0;
//				}
//				else
//				{
//					spectra_mask[ t ] = 1;
//				}
//			}
//
//			if( fabs( chan_mean[ c ] - old_mean ) < 0.001 && fabs( chan_var[ c ] - old_var ) < 0.0001 && rounds > 1)
//			{
//				////printf("\n%d\t%d\t%.16lf\t%.16lf\t%.16lf\t%.16lf", c, rounds, (chan_mean[c]-old_mean), (chan_var[c]-old_var), chan_mean[c], chan_var[c]);
//				finish = 1;
//			}
//
//			old_mean = chan_mean[ c ];
//			old_var = chan_var[ c ];
//			rounds++;
//		}
//
//
//		//printf( "\nChan mean, var: %lf %lf\n", chan_mean[ c ], chan_var[ c ] );
//
//
//		if( chan_mask[ c ] != 0 )
//		{
//			for( int t = 0; t < ( nsamp ); t++ )
//			{
//				stage[ c * ( size_t )nsamp + t ] = ( stage[ c * ( size_t )nsamp + t ] - ( float )chan_mean[ c ] ) / ( float )chan_var[ c ];
//			}
//		}
//		else
//		{
//			int perm_one = ( int )( ( ( float )rand() / ( float )RAND_MAX ) * nsamp );
//
//			for( int t = 0; t < nsamp; t++ )
//			{
//				stage[ c * ( size_t )nsamp + t ] = random_chan_one[ ( t + perm_one ) % nsamp ];
//			}
//
//			chan_mean[ c ] = 0.0;
//			chan_var[ c ]  = 1.0;
//			chan_mask[ c ] = 1;
//		}
//	}
//
//	// Find the BLN and try to flatten the input data per spectra (remove non-stationary component).
//
//	unsigned int cnt = 0;
//
//	for( int t = 0; t < nsamp; t++ )
//	{
//		int counter = 0;
//
//		for( int c = 0; c < nchans; c++ )
//		 chan_mask[ c ]=1;
//
//		int finish = 0;
//		int rounds = 1;
//
//		double old_mean = 0.0;
//		double old_var = 0.0;
//
//		while( finish == 0 )
//		{
//			cnt += 1;
//			counter = 0;
//			spectra_mean[ t ] = 0.0;
//			for( int c = 0; c < nchans; c++ )
//			{
//				if( chan_mask[ c ] == 1 )
//				{
//	        spectra_mean[ t ] += stage[ c * ( size_t )nsamp + t ];
//					counter++;
//				}
//      }
//
//			//printf( "\nSpectra mean %lf\n", spectra_mean[ t ] );
//			//printf( "counter %d\n", counter );
//
//			if( counter == 0 )
//			{
//				//printf( "\nCounter zero, Spectra %d", t );
//				spectra_mask[ t ] = 0;
//				finish = 1;
//				break;
//			}
//
//			spectra_mean[ t ] /= (counter);
//
//			counter = 0;
//			spectra_var[ t ] = 0.0;
//			for( int c = 0; c < nchans; c++ )
//			{
//				if( chan_mask[ c ] == 1 )
//				{
//					spectra_var[ t ] += ( stage[ c * ( size_t )nsamp + t ] - spectra_mean[ t ] ) * ( stage[ c * ( size_t )nsamp + t ] - spectra_mean[ t ] );
//					counter++;
//				}
//			}
//
//			//printf( "spectra_var %lf\n", spectra_var[ t ] );
//			//printf( "counter %u\n", counter );
//
//			spectra_var[ t ] /= (counter);
//			spectra_var[ t ] = sqrt( spectra_var[ t ] );
//
//			if( ( spectra_var[ t ] ) * 1000000.0 < 0.1 )
//			{
//				////printf("\nVarience zero, Spectra %d %d %lf %.16lf", t, rounds, spectra_mean[t], spectra_var[t] );
//				spectra_mask[ t ] = 0;
//				finish = 1;
//				break;
//			}
//
//			if( spectra_mask[ t ] != 0 )
//			{
//				for( int c = 0; c < nchans; c++ )
//				{
//					if( ( ( stage[ c * (size_t)nsamp + t ] - spectra_mean[ t ] ) / spectra_var[ t ] ) > sigma_cut || ( ( stage[ c * (size_t)nsamp + t ] - spectra_mean[ t ] ) / spectra_var[ t ] ) < -sigma_cut)
//					{
//						chan_mask[ c ] = 0;
//					}
//					else
//					{
//						chan_mask[ c ] = 1;
//					}
//				}
//			}
//
//			if( fabs( spectra_mean[ t ] - old_mean ) < 0.001 && fabs( spectra_var[ t ] - old_var ) < 0.0001 && rounds > 1)
//			{
//				////printf("\n%d\t%d\t%.16lf\t%.16lf\t%.16lf\t%.16lf", t, rounds, (spectra_mean[t] - old_mean), (spectra_var[t] - old_var), spectra_mean[t], spectra_var[t]);
//				finish = 1;
//			}
//
//			old_mean = spectra_mean[ t ];
//			old_var = spectra_var[ t ];
//			rounds++;
//		}
//
//		//return;
//		////printf("Spectra mean, var: %lf %d\n", spectra_mean[t], spectra_var[t] );
//
//		if( spectra_mask[ t ] != 0)
//		{
//			for( int c = 0; c < nchans; c++ )
//			{
//				stage[ c * (size_t)nsamp + t ] = ( stage[ c * (size_t)nsamp + t ] - (float)spectra_mean[ t ] ) / (float)spectra_var[ t ];
//			}
//		}
//		else
//		{
//			int perm_one = (int)( ( (float)rand() / (float)RAND_MAX ) * nchans);
//
//			for( int c = 0; c < nchans; c++ )
//			{
//				stage[ c * (size_t)nsamp + t ] = random_spectra_one[ ( c + perm_one ) % nchans ];
//			}
//
//			spectra_mean[ t ] = 0.0;
//			spectra_var[ t ]  = 1.0;
//			spectra_mask[ t ] = 1;
//		}
//	}
//
//	//printf( "cnt is %u\n", cnt );
//
//
//	double mean_rescale = 0.0;
//	double var_rescale  = 0.0;
//
//	// Find the mean and SD of the mean and SD...
//	int finish = 0;
//	int rounds = 1;
//	int counter = 0;
//
//	double mean_of_mean = 0.0;
//	double var_of_mean  = 0.0;
//	double mean_of_var  = 0.0;
//	double var_of_var   = 0.0;
//
//	double old_mean_of_mean = 0.0;
//	double old_var_of_mean  = 0.0;
//	double old_mean_of_var  = 0.0;
//	double old_var_of_var   = 0.0;
//
//	for( int c = 0; c < nchans; c++ ) chan_mask[ c ] = 1;
//
//	while(finish == 0)
//	{
//		mean_of_mean = 0.0;
//		counter = 0;
//
//		for( int c = 0; c < nchans; c++ )
//		{
//			if( chan_mask[ c ] == 1 )
//			{
//				mean_of_mean += chan_mean[ c ];
//				counter++;
//			}
//		}
//
//		//printf("mm is %lf\n",mean_of_mean );
//
//		mean_of_mean /= counter;
//
//		var_of_mean = 0.0;
//		counter = 0;
//
//		for( int c = 0; c < nchans; c++ )
//		{
//			if( chan_mask[ c ] == 1 )
//			{
//				var_of_mean += ( chan_mean[ c ] - mean_of_mean ) * ( chan_mean[ c ] - mean_of_mean );
//				counter++;
//			}
//		}
//
//		//printf( "\nvar_of_mean %lf\n", var_of_var );
//    //printf( "\ncounter %u\n", counter );
//
//		var_of_mean /= ( counter );
//		var_of_mean = sqrt( var_of_mean );
//
//		mean_of_var = 0.0;
//		counter = 0;
//
//		for( int c = 0; c < nchans; c++ )
//		{
//			if( chan_mask[ c ] == 1 )
//			{
//				mean_of_var += chan_var[ c ];
//				counter++;
//			}
//		}
//
//		//printf("\nmean_of_var %lf\n",mean_of_var );
//    //printf("\ncounter %u\n",counter );
//
//		mean_of_var /= counter;
//
//		var_of_var = 0.0;
//		counter = 0;
//
//		for( int c = 0; c < nchans; c++ )
//		{
//			if( chan_mask[ c ] == 1 )
//			{
//				var_of_var += ( chan_var[ c ] - mean_of_var ) * ( chan_var[ c ] - mean_of_var);
//				counter++;
//			}
//		}
//
//		//printf("\nvar_of_var %lf\n",var_of_var );
//    //printf("\ncounter %u\n",counter);
//
//		var_of_var /= (counter);
//		var_of_var = sqrt( var_of_var );
//
//		for( int c = 0; c < nchans; c++ )
//			if( fabs( chan_mean[ c ] - mean_of_mean ) / var_of_mean > sigma_cut || fabs( chan_var[ c ] - mean_of_var ) / var_of_var > sigma_cut )
//		  	chan_mask[ c ] = 0;
//
//		if(fabs(mean_of_mean - old_mean_of_mean)   < 0.001 &&
//		   fabs(var_of_mean  - old_var_of_mean )   < 0.001 &&
//		   fabs(mean_of_var  - old_mean_of_var )   < 0.001 &&
//		   fabs(var_of_var   - old_var_of_var  )   < 0.001)
//			 {
//
//				 finish = 1;
//			 }
//
//		old_mean_of_mean = mean_of_mean;
//		old_var_of_mean  = var_of_mean;
//		old_mean_of_var  = mean_of_var;
//		old_var_of_var   = var_of_var;
//
//		rounds++;
//	}
//
//	//printf("\n0 %lf %lf", mean_of_mean, var_of_mean);
//	//printf("\n0 %lf %lf", mean_of_var,  var_of_var);
//
//	mean_rescale = mean_of_mean;
//	var_rescale  = mean_of_var;
//
//	float clipping_constant = 0.0;
//
//	for( int c = 0; c < nchans; c++ ) clipping_constant += chan_mask[ c ];
//	clipping_constant = ( nchans - clipping_constant ) / nchans;
//	//printf("\n clipping_constant is %f\n",clipping_constant );
//	clipping_constant = sqrt( -2.0 * log( clipping_constant * 2.506628275 ) );
//	//printf("This This %f\n",clipping_constant );
//
//	// Perform channel replacement
//	for( int c = 0; c < nchans; c++ )
//	{
//		if( fabs( ( chan_mean[ c ] - mean_of_mean ) / var_of_mean ) > clipping_constant && fabs( ( chan_var[ c ] - mean_of_var ) / var_of_var ) > clipping_constant )
//		{
//			////printf("\nReplacing Channel %d %lf %lf", c, chan_mean[c], chan_var[c]);
//			int perm_one = (int)( ( (float)rand() / (float)RAND_MAX ) * nsamp );
//
//			for( int t = 0; t < (nsamp); t++ )
//			{
//				stage[ ( c * (size_t)nsamp + t) ] = random_chan_two[ ( t + perm_one ) % nsamp ];
//			}
//		}
//	}
//
//	finish = 0;
//	rounds = 1;
//	counter = 0;
//
//	mean_of_mean = 0.0;
//	var_of_mean  = 0.0;
//	mean_of_var  = 0.0;
//	var_of_var   = 0.0;
//
//	old_mean_of_mean = 0.0;
//	old_var_of_mean  = 0.0;
//	old_mean_of_var  = 0.0;
//	old_var_of_var   = 0.0;
//
//	for( int t = 0; t < (nsamp); t++ ) spectra_mask[ t ] = 1;
//
//	while( finish == 0 )
//	{
//		mean_of_mean = 0.0;
//		counter = 0;
//
//		for( int t = 0; t < (nsamp); t++ )
//		{
//			if( spectra_mask[ t ] == 1 )
//			{
//				mean_of_mean += spectra_mean[ t ];
//				counter++;
//			}
//		}
//
//		mean_of_mean /= counter;
//
//		var_of_mean = 0.0;
//		counter = 0;
//
//		for( int t = 0; t < (nsamp); t++ )
//		{
//			if( spectra_mask[ t ] == 1 )
//			{
//				var_of_mean += ( spectra_mean[ t ] - mean_of_mean ) * ( spectra_mean[ t ]- mean_of_mean );
//				counter++;
//			}
//		}
//
//		var_of_mean /= (counter);
//		var_of_mean = sqrt( var_of_mean );
//
//		mean_of_var = 0.0;
//		counter = 0;
//
//		for( int t = 0; t < (nsamp); t++ )
//		{
//			if( spectra_mask[ t ] == 1 )
//			{
//				mean_of_var += spectra_var[ t ];
//				counter++;
//			}
//		}
//
//		mean_of_var /= counter;
//
//		var_of_var = 0.0;
//		counter = 0;
//
//		for( int t = 0; t < (nsamp); t++ )
//		{
//			if( spectra_mask[ t ] == 1 )
//			{
//				var_of_var += ( spectra_var[ t ] - mean_of_var ) * ( spectra_var[ t ] - mean_of_var );
//				counter++;
//			}
//		}
//
//		var_of_var /= (counter);
//		var_of_var = sqrt( var_of_var );
//
//		for( int t = 0; t < (nsamp); t++) if( fabs( spectra_mean[ t ] - mean_of_mean ) / var_of_mean > sigma_cut || fabs( spectra_var[ t ] - mean_of_var ) / var_of_var > sigma_cut ) spectra_mask[ t ] = 0;
//
//		if(fabs(mean_of_mean - old_mean_of_mean)   < 0.001 &&
//		   fabs(var_of_mean  - old_var_of_mean )   < 0.001 &&
//		   fabs(mean_of_var  - old_mean_of_var )   < 0.001 &&
//		   fabs(var_of_var   - old_var_of_var  )   < 0.001)
//			 {
//				 finish = 1;
//			 }
//
//		old_mean_of_mean = mean_of_mean;
//		old_var_of_mean  = var_of_mean;
//		old_mean_of_var  = mean_of_var;
//		old_var_of_var   = var_of_var;
//
//		rounds++;
//	}
//
//	//printf("\n0 %lf %lf", mean_of_mean, var_of_mean);
//	//printf("\n0 %lf %lf", mean_of_var,  var_of_var);
//
//	clipping_constant = 0.0;
//	for( int t = 0; t < nsamp; t++ ) clipping_constant += spectra_mask[ t ];
//	clipping_constant = ( nsamp - clipping_constant ) / nsamp;
//	clipping_constant = sqrt( -2.0 * log( clipping_constant * 2.506628275 ) );
//
//	// Perform spectral replacement
//	for( int t = 0; t < (nsamp); t++ )
//	{
//	    if( fabs( ( spectra_mean[ t ] - mean_of_mean ) / var_of_mean ) > clipping_constant && fabs( ( spectra_var[ t ] - mean_of_var ) / var_of_var ) > clipping_constant )
//			{
//				////printf("\nReplacing Spectral %d %lf %lf", t, spectra_mean[t], spectra_var[t]);
//				int perm_one = (int)( ( (float)rand() / (float)RAND_MAX) * nchans );
//				for( int c = 0; c < nchans; c++ )
//				{
//					stage[ c * (size_t)nsamp + t ] = random_spectra_two[ ( c + perm_one ) % nchans ];
//				}
//     }
//	}
//
//
//	for( int c = 0; c < nchans; c++ )
//	{
//		for( int t = 0; t < (nsamp); t++ )
//		{
//			//(*input_buffer)[c  + (size_t)nchans * t] = (unsigned char) ((stage[c * (size_t)nsamp + t]*orig_var)+orig_mean);
//			(*input_buffer)[ c  + (size_t)nchans * t ] = (unsigned char) ( ( stage[ c * (size_t)nsamp + t ] * var_rescale ) + mean_rescale );
//		}
//	}
//
//	FILE *fp_mask = fopen ("masked_chans.txt", "w+");
//
//	for( int c = 0; c < nchans; c++ )
//	{
//		for( int t = 0; t < (nsamp) / file_reducer; t++ )
//		{
//			fprintf(fp_mask, "%d ", (unsigned char)((stage[c * (size_t)nsamp + t]*orig_var)+orig_mean));
//			fprintf( fp_mask, "%d ", (unsigned char)( ( stage[ c * (size_t)nsamp + t] * var_rescale ) + mean_rescale ) );
//		}
//
//		fprintf(fp_mask, "\n");
//	}
//  fclose(fp_mask);
//
//	//printf("\n%lf %lf", mean_rescale / orig_mean, var_rescale / orig_var);
//
//
//	free(chan_mask);
//	free(spectra_mask);
//	free(chan_mean);
//	free(chan_var);
//	free(spectra_mean);
//	free(spectra_var);
//	free(stage);
//}

void readInputBuffer(float *input_buffer, int nsamp, int nchans, char fname[100])
{

	FILE *fp_inputBuffer = fopen( fname, "r" );

  if(fp_inputBuffer == NULL)
  {
    printf("File buffer empty, No file to read. EXITING PROGRAM!!!! \n");
    exit(EXIT_FAILURE);
  }

	if( fp_inputBuffer == NULL )
	{
		//printf("Error opeing file\n" );
		exit(0);
	}

  for( int i=0; i < nsamp * nchans; i++ )
	{
    fscanf(fp_inputBuffer, "%hu", &input_buffer[ i ] );
		////printf("%hu  ",input_buffer[i] );
	}
	//printf("done reading\n" );
	fclose( fp_inputBuffer );
}

int main()
{

#if (_OPENACC)
  printf("Running OpenACC version \n");
#else
  printf("Running SEQ version \n");
#endif
  timeval  startTimer, endTimer,
           startReadTimer, endReadTimer,
           startRFITimer, endRFITimer;

  //Total Time of the application
  gettimeofday(&startTimer, NULL);

  int nsamp = 491522/N;//10977282;
  int  nchans = 4096;
  unsigned long long in_buffer_bytes = nsamp * nchans * sizeof( float );
  float *in_buffer = ( float* )malloc( in_buffer_bytes );
  float *input_buffer = in_buffer;

  //unsigned long long i;
  //#pragma omp parallel
  // #pragma omp parallel for
  // for(i=0; i<nsamp*nchans; i++)
  // {
  //   in_buffer[i] = 100;
  // }

  //Total Time of the application
  gettimeofday(&startReadTimer, NULL);
	char fname[100] = "input_buffer.txt";
	readInputBuffer(in_buffer, nsamp, nchans, fname);
  gettimeofday(&endReadTimer, NULL);
  double elapsedReadTimer = (endReadTimer.tv_sec - startReadTimer.tv_sec) + 1e-6 * (endReadTimer.tv_usec - startReadTimer.tv_usec);

  printf("DONE READING with nchan = %d\t nsamp = %d\t time taken = %f !!!!!\n", nchans, nsamp, elapsedReadTimer);

  double elapsedRFITimer = 0.0;
  gettimeofday(&startRFITimer, NULL);
  rfi_debug(nsamp, nchans, &input_buffer, elapsedRFITimer);

  //Write output in a different routine and then free the stage storage
  printf("RFI routine time = %f !!!!!\n", elapsedRFITimer);

  gettimeofday(&endTimer, NULL);
  double elapsedTimer = (endTimer.tv_sec - startTimer.tv_sec) + 1e-6 * (endTimer.tv_usec - startTimer.tv_usec);

  printf("Total time = %f\n", elapsedTimer);

  return 0;
}
