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

#ifdef USE_NVTX
#include <nvToolsExt.h>

const uint32_t colors[] = { 0xff00ff00, 0xff0000ff, 0xffffff00, 0xffff00ff, 0xff00ffff, 0xffff0000, 0xffffffff };
const int num_colors = sizeof(colors)/sizeof(uint32_t);

#define PUSH_RANGE(name,cid) { \
      int color_id = cid; \
      color_id = color_id%num_colors;\
      nvtxEventAttributes_t eventAttrib = {0}; \
      eventAttrib.version = NVTX_VERSION; \
      eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE; \
      eventAttrib.colorType = NVTX_COLOR_ARGB; \
      eventAttrib.color = colors[color_id]; \
      eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII; \
      eventAttrib.message.ascii = name; \
      nvtxRangePushEx(&eventAttrib); \
}
#define POP_RANGE nvtxRangePop();
#else
#define PUSH_RANGE(name,cid)
#define POP_RANGE
#endif

#define chan_mask(t,c) chan_mask[t*nchans+c]
#if (_OPENACC)
#include <openacc.h>
#include <accelmath.h>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>

#else
#include <math.h>
#endif

#include "arrayMD.h"

//Setting N in a way such that we do not need to change anything later in terms of number of channels launched
#if defined(PRODUCTION_VERSION)
#define N 1
#else
#define N 50
#endif

inline void*
safe_malloc(size_t n)
{
  void* p = malloc(n);
  if(p == NULL)
  {
    fprintf(stderr, "Fatal: failed to allocate %zu bytes.\n", n);
    abort();
  }
  return p;
}

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
//#pragma acc enter data copyin(stage[:1])
//#pragma acc enter data copyin(stage.dptr[:stage.size])
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
#if(_OPENACC)
#pragma acc parallel loop gang vector collapse(2)\
  present(stage) \
  reduction(+:orig_mean_loc) async(1)
#endif
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
#if(_OPENACC)
#pragma acc parallel loop gang vector collapse(2)\
  copyin(orig_mean_loc) \
  present(stage) \
  reduction(+:orig_var_loc) async(1)
#endif
  for(int t = 0; t < nsamp; ++t)
	{
    for( int c = 0; c < nchans; c++ )
		{
      double store1 = stage(t,c) - orig_mean_loc;
			orig_var_loc += store1*store1;
		}
	}

//  printf("0: orig_var_loc = %f\n", orig_var_loc);

  orig_var = orig_var_loc;
	orig_var /= ( nsamp * nchans );
	orig_var = sqrt( orig_var );
//  printf("1: orig_var_loc = %f\n", orig_var);
}

void random_chan(float* random_chan_one, float* random_chan_two, int nsamp)
{
  for(int iter = 0; iter < nsamp; ++iter)
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

void random_spectra(float* random_spectra_one, float* random_spectra_two, int nchans)
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

void channel_process(Array2D<char>& spectra_mask, Array2D<float>& stage, double *chan_mean, double *chan_var,
    int nsamp, int nchans, float *random_chan_one, float *random_chan_two)
{
	short int *chan_mask = ( short int* )safe_malloc( nchans * sizeof( short int ) );
	short int *Nchan_mask = ( short int* )safe_malloc( nchans * sizeof( short int ) );

#if(_OPENACC)
#pragma acc enter data copyin (spectra_mask[0:1], spectra_mask.dptr[0:spectra_mask.size])
#endif

	float	sigma_cut	= 2.0f;
#if(_OPENACC)
#pragma acc parallel loop gang \
  present(spectra_mask, stage)
#endif
  for( int c = 0; c < nchans; c++ )
	{
    chan_mask[ c ] = 1;
#if(_OPENACC)
#pragma acc loop vector
#endif
    for( int t = 0; t < nsamp; t++ ) spectra_mask(c,t) = 1;

		int counter = nsamp;
		int rounds = 1;
		double old_mean = 0.0;
		double old_var = 0.0;
    bool finish = false;

    double chan_mean_c = 0.0, chan_var_c = 0.0;
		while(!finish)
		{
      chan_mean_c = 0.0;
			chan_var_c = 0.0;

#if(_OPENACC)
#pragma acc loop vector \
      reduction(+:chan_mean_c)
#endif
      for( int t = 0; t < nsamp ; t++ )
			{
	        chan_mean_c += stage(t,c) * spectra_mask(c,t);
      }

			if( counter == 0 )
			{
				chan_mask[ c ] = 0;
				finish = true;
			}
			chan_mean_c /= ( counter );

#if(_OPENACC)
#pragma acc loop vector \
      reduction(+:chan_var_c)
#endif
			for( int t = 0; t < nsamp ; t++)
			{
        float store1 = stage(t,c) - chan_mean_c;
        chan_var_c += store1 * store1 * spectra_mask(c,t);
			}

			chan_var_c /= ( counter );
			chan_var_c = sqrt( chan_var_c );

			if( chan_var_c < 1e-7 )
			{
				chan_mask[ c ] = 0;
				finish = true;
			}

      counter = 0;
#if(_OPENACC)
#pragma acc loop vector \
      reduction(+:counter)
#endif
			for( int t = 0; t < ( nsamp ); t++ )
			{
        float threshold = (stage(t,c) - chan_mean_c) / chan_var_c;
				if( threshold > sigma_cut || threshold < -sigma_cut )
				{
					spectra_mask(c,t) = 0;
				}
				else
				{
          counter++;
					spectra_mask(c,t) = 1;
				}
			}

			if( fabs( chan_mean_c - old_mean ) < 0.001 && fabs( chan_var_c - old_var ) < 0.0001 && rounds > 1)
			{
				finish = true;
			}

			old_mean = chan_mean_c;
			old_var = chan_var_c;
			rounds++;
		}

    chan_mean[c] = chan_mean_c;
    chan_var[c] = chan_var_c;
  }

#if (_OPENACC)
#pragma acc enter data copyin (chan_mask[0:nchans], chan_mean[0:nchans], chan_var[0:nchans], \
    random_chan_one[0:nsamp])
#pragma acc parallel loop gang vector collapse(2) \
  present(stage, chan_mean, chan_var, random_chan_one, chan_mask)
#endif
	for( int t = 0; t < nsamp ; t++ )
	{
      for( int c = 0; c < nchans; c++ )
			{
				stage(t,c) = ( stage(t,c) - ( float )chan_mean[ c ] ) / ( float )chan_var[ c ] * chan_mask[c];

        if(!chan_mask[c])
        {
          int perm_one = ( int )( ( ( float )rand_local() / ( float )RAND_MAX ) * nsamp );
          stage(t,c) = random_chan_one[ ( t + perm_one ) % nsamp ] ;
    			chan_mean[ c ] = 0.0;
    			chan_var[ c ]  = 1.0;
        }
			}
  }

#pragma acc exit data copyout(chan_mean[0:nchans], chan_var[0:nchans])
}

void sample_process_V2(double *spectra_mean, Array2D<float>& stage, Array2D<char>& spectra_mask,
    double *spectra_var, int nsamp, int nchans, float *random_spectra_one, float *random_spectra_two)
{
	float	sigma_cut	= 2.0f;
  int nchans_loc = nchans;

//  Array2D<char> chan_mask(nsamp,nchans);
#if(_OPENACC)
  char *chan_mask = (char*) acc_malloc(nsamp*nchans*sizeof(char));
#else
  char *chan_mask = (char*) malloc(nsamp*nchans*sizeof(char));
#endif

  unsigned stage_size = stage.size;

#if(_OPENACC)
#pragma acc enter data copyin(spectra_mean[0:nsamp], spectra_var[0:nsamp],\
    random_spectra_one[0:nsamp])
#endif

#if(_OPENACC)
#pragma acc parallel loop gang \
      present(stage, spectra_mean, spectra_var, spectra_mask) \
  deviceptr(chan_mask) \
  num_gangs(nsamp) vector_length(64)
#endif
	for( int t = 0; t < nsamp; t++ )
	{
		int counter = nchans_loc;
		bool finish = false;
		int rounds = 1;
		double old_mean = 0.0;
		double old_var = 0.0;
    double spectra_mean_t = 0.0, spectra_var_t = 0.0;

    spectra_mean[t] = 0.0;
    spectra_var[t] = 0.0;

#if(_OPENACC)
#pragma acc loop vector
#endif
		for( int c = 0; c < nchans_loc; c++ )
		 chan_mask(t,c) =1;

		while(!finish )
		{
      spectra_mean_t = 0.0, spectra_var_t = 0.0;
#if(_OPENACC)
#pragma acc loop vector \
      reduction(+:spectra_mean_t)
#endif
			for( int c = 0; c < nchans_loc; c++ )
			{
	        spectra_mean_t += stage(t,c) * chan_mask(t,c);
      }

			spectra_mean_t /= (counter);

#if(_OPENACC)
#pragma acc loop vector \
      reduction(+:spectra_var_t)
#endif
			for( int c = 0; c < nchans_loc; c++ )
			{
        float store1 = stage(t,c) - spectra_mean_t;
				spectra_var_t += store1 * store1 * chan_mask(t,c);
			}

			spectra_var_t /= (counter);
			spectra_var_t = sqrt( spectra_var_t );

      counter = 0;

#if(_OPENACC)
#pragma acc loop vector\
      reduction(+:counter)
#endif
      for( int c = 0; c < nchans_loc; c++ )
      {
        float threshold = (float) ((stage(t,c) - spectra_mean_t)) / spectra_var_t;
        if( threshold > sigma_cut || threshold < -sigma_cut)
        {
          chan_mask(t,c) = 0;
        }
        else
        {
          counter++;
          chan_mask(t,c) = 1;
        }
      }

			if( (spectra_var_t < 1e-7) || (!counter) )
			{
				spectra_mask(nchans-1,t) = 0;
				finish = true;
			}
			if(fabs( spectra_mean_t - old_mean ) < 0.001 && fabs( spectra_var_t - old_var ) < 0.0001)
			{
				finish = true;
			}

			old_mean = spectra_mean_t ;
			old_var = spectra_var_t;
		}
      spectra_mean[t] = spectra_mean_t;
      spectra_var[t] = spectra_var_t;

#if(_OPENACC)
#pragma acc loop vector
#endif
			for( int c = 0; c < nchans; c++ )
			{
				stage(t,c) = ( stage(t,c) - (float)spectra_mean[t] ) / (float)spectra_var[t] * spectra_mask(nchans-1,t);
			}

		if( !spectra_mask(nchans-1,t) )
		{
			int perm_one = (int)( ( (float)rand_local() / (float)RAND_MAX ) * nchans);

#if(_OPENACC)
#pragma acc loop vector
#endif
			for( int c = 0; c < nchans; c++ )
			{
				stage(t,c) = random_spectra_one[ ( c + perm_one ) % nchans ];
			}

			spectra_mean[ t ] = 0.0;
			spectra_var[ t ]  = 1.0;
//			spectra_mask(c,t) = 1;
		}
  }


#if(_OPENACC)
acc_free(chan_mask);
#pragma acc update self(stage.dptr[0:stage_size])
#pragma acc exit data copyout(spectra_mean[0:nsamp], spectra_var[0:nsamp])
#else
  free(chan_mask);
#endif

}

void update_input_buffer(int nsamp, int nchans, unsigned short *input_buffer, Array2D<float>& stage, double var_rescale, double mean_rescale)
{
#if(_OPENACC)
#pragma acc parallel loop gang vector collapse(2) \
  present(stage,input_buffer)
#endif
	for( int c = 0; c < nchans; c++ )
	{
		for( int t = 0; t < (nsamp); t++ )
		{
			//(*input_buffer)[c  + (size_t)nchans * t] = (unsigned char) ((stage[c * (size_t)nsamp + t]*orig_var)+orig_mean);
			input_buffer[ c  + (size_t)nchans * t ] = (unsigned char) ( ( stage(t,c) * var_rescale ) + mean_rescale );
    }
  }
}

void whileLoop_V2(Array2D<float>& stage, int *chan_mask, double *chan_mean, double *chan_var,
    int nsamp, int nchans, double& mean_rescale, double& var_rescale,
    float *random_chan_one, float *random_chan_two, double *spectra_mean,
    double *spectra_var, float *random_spectra_one, float *random_spectra_two )
{
	float	sigma_cut	= 2.0f;
	int *spectra_mask = ( int* )safe_malloc( nsamp * sizeof( int ) );
	for( int t = 0; t < nsamp; t++ ) spectra_mask[ t ] = 1;

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

	//printf("\n0 %lf %lf", mean_of_mean, var_of_mean);
	//printf("\n0 %lf %lf", mean_of_var,  var_of_var);

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

	finish = 0;
	rounds = 1;
	counter = 0;

	mean_of_mean = 0.0;
	var_of_mean  = 0.0;
	mean_of_var  = 0.0;
	var_of_var   = 0.0;

	old_mean_of_mean = 0.0;
	old_var_of_mean  = 0.0;
	old_mean_of_var  = 0.0;
	old_var_of_var   = 0.0;

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

	clipping_constant = 0.0;
	for( int t = 0; t < nsamp; t++ ) clipping_constant += spectra_mask[ t ];
	clipping_constant = ( nsamp - clipping_constant ) / nsamp;
	clipping_constant = sqrt( -2.0 * log( clipping_constant * 2.506628275 ) );

	// Perform spectral replacement
	for( int t = 0; t < (nsamp); t++ )
	{
	    if( fabs( ( spectra_mean[ t ] - mean_of_mean ) / var_of_mean ) > clipping_constant && fabs( ( spectra_var[ t ] - mean_of_var ) / var_of_var ) > clipping_constant )
			{
				////printf("\nReplacing Spectral %d %lf %lf", t, spectra_mean[t], spectra_var[t]);
				int perm_one = (int)( ( (float)rand_local() / (float)RAND_MAX) * nchans );
				for( int c = 0; c < nchans; c++ )
				{
					stage(t,c) = random_spectra_two[ ( c + perm_one ) % nchans ];
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
inline void transpose_stage_ip(Array2D<float>& stage, unsigned short *input_buffer, int nsamp, int nchans)
{
#if(_OPENACC)
#pragma acc parallel loop gang vector collapse(2) \
  present(stage, input_buffer)
#endif
  for(int t = 0; t < nsamp; ++t)
	{
    for( int c = 0; c < nchans; c++ )
		{
			stage(t,c) = ( float ) ( input_buffer )[ c  + ( size_t )nchans * t ];
		}
	}
}


/******************************************************************************************************************************/

void rfi_debug3(int nsamp, int nchans, unsigned short *input_buffer, double& elapsedTimer)
{
  cout << "working with rfi_debug V3 " << endl;
	// ~~~ RFI Correct ~~~ //
	int file_reducer = 1;
	float	sigma_cut	= 2.0f;

  Array2D<float> stage(nsamp,nchans);

	// Random Vectors
	float *random_chan_one = ( float* )safe_malloc( nsamp * sizeof( float ) );
	float *random_chan_two = ( float* )safe_malloc( nsamp * sizeof( float ) );
	float *random_spectra_one = ( float* )safe_malloc( nchans * sizeof( float ) );
	float *random_spectra_two = ( float* )safe_malloc( nchans * sizeof( float ) );
	// Allocate working arrays

	int *chan_mask = ( int* )safe_malloc( nchans * sizeof( int ) );
	for( int c = 0; c < nchans; c++ ) chan_mask[ c ] = 1;

//	int *spectra_mask = ( int* )safe_malloc( nsamp * sizeof( int ) );
//	for( int t = 0; t < nsamp; t++ ) spectra_mask[ t ] = 1;
	Array2D<char> spectra_mask(nchans,nsamp);

	double *chan_mean = ( double* )safe_malloc( nchans * sizeof( double ) );
  for( int c = 0; c < nchans; c++ ) chan_mean[ c ] = 0.0;

	double *chan_var = ( double* )safe_malloc( nsamp * sizeof( double ) );
  for( int c = 0; c < nchans; c++ ) chan_var[ c ] = 0.0;

	double *spectra_mean = ( double* )safe_malloc( nsamp * sizeof( double ) );
  for( int t = 0; t < nsamp; t++ ) spectra_mean[ t ] = 0.0;

	double *spectra_var = ( double* )safe_malloc( nsamp * sizeof( double ) );
  for( int t = 0; t < nsamp; t++ ) spectra_var[ t ] = 0.0;


	double orig_mean = 0.0;
	double orig_var=0.0;
  timeval startTimer, endTimer;
  timeval startRoutine, endRoutine;
  gettimeofday(&startTimer, NULL);
  double routine_add_time = 0.0;

#if(_OPENACC)
#pragma acc enter data copyin(stage[0:1], stage.dptr[0:stage.size], \
    input_buffer[0:nsamp*nchans])
#endif


  PUSH_RANGE("transpose",1)
  gettimeofday(&startRoutine, NULL);
  transpose_stage_ip(stage, input_buffer, nsamp, nchans);
  gettimeofday(&endRoutine, NULL);
  double Timer = (endRoutine.tv_sec - startRoutine.tv_sec) + 1e-6 * (endRoutine.tv_usec - startRoutine.tv_usec);
  routine_add_time += Timer;
  POP_RANGE

//  cout << "Done with transpose_stage with T[secs] = " << Timer << endl;

  PUSH_RANGE("random_chan",2)
  gettimeofday(&startRoutine, NULL);
  random_chan(random_chan_one, random_chan_two, nsamp);
  gettimeofday(&endRoutine, NULL);
  Timer = (endRoutine.tv_sec - startRoutine.tv_sec) + 1e-6 * (endRoutine.tv_usec - startRoutine.tv_usec);
  routine_add_time += Timer;
  POP_RANGE
//  cout << "Done with random chan with T[secs] = " << Timer << endl;

  PUSH_RANGE("random_spectra",3)
  gettimeofday(&startRoutine, NULL);
  random_spectra(random_spectra_one, random_spectra_two, nchans);
  gettimeofday(&endRoutine, NULL);
  Timer = (endRoutine.tv_sec - startRoutine.tv_sec) + 1e-6 * (endRoutine.tv_usec - startRoutine.tv_usec);
  routine_add_time += Timer;
  POP_RANGE
//  cout << "Done with random spectra with T[secs] = " << Timer << endl;

  PUSH_RANGE("channel_process",4)
  gettimeofday(&startRoutine, NULL);
  channel_process(spectra_mask, stage, chan_mean, chan_var, nsamp, nchans, random_chan_one, random_chan_two);
  gettimeofday(&endRoutine, NULL);
  Timer = (endRoutine.tv_sec - startRoutine.tv_sec) + 1e-6 * (endRoutine.tv_usec - startRoutine.tv_usec);
  routine_add_time += Timer;
  POP_RANGE
//  cout << "Done with channel process with T[secs] = " << Timer << endl;

  PUSH_RANGE("sample_process",5)
  gettimeofday(&startRoutine, NULL);
  sample_process_V2( spectra_mean, stage, spectra_mask, spectra_var, nsamp, nchans, random_spectra_one, random_spectra_two);
  gettimeofday(&endRoutine, NULL);
  Timer = (endRoutine.tv_sec - startRoutine.tv_sec) + 1e-6 * (endRoutine.tv_usec - startRoutine.tv_usec);
  routine_add_time += Timer;
  POP_RANGE
//  cout << "Done with sample process with T[secs] = " << Timer << endl;

//  free(spectra_mask_t);
#if(_OPENACC)
#pragma acc exit data delete (spectra_mask, spectra_mask.dptr[0:spectra_mask.size])
#endif

  double mean_rescale = 0.0, var_rescale = 0.0;

  gettimeofday(&startRoutine, NULL);
  reduce_orig_mean(stage, nsamp, nchans, orig_mean);
  gettimeofday(&endRoutine, NULL);
  Timer = (endRoutine.tv_sec - startRoutine.tv_sec) + 1e-6 * (endRoutine.tv_usec - startRoutine.tv_usec);
  routine_add_time += Timer;
//  cout << "Done with reduce_mean with T[secs] = " << Timer << endl;

  gettimeofday(&startRoutine, NULL);
  reduce_orig_var(stage, nsamp, nchans, orig_mean, orig_var);
  gettimeofday(&endRoutine, NULL);
  Timer = (endRoutine.tv_sec - startRoutine.tv_sec) + 1e-6 * (endRoutine.tv_usec - startRoutine.tv_usec);
  routine_add_time += Timer;
//  cout << "Done with reduce_var with T[secs] = " << Timer << endl;

  PUSH_RANGE("whileLoop",6)
  gettimeofday(&startRoutine, NULL);
  whileLoop_V2(stage, chan_mask, chan_mean, chan_var, nsamp, nchans, mean_rescale, var_rescale,
      random_chan_one, random_chan_two, spectra_mean, spectra_var,
      random_spectra_one, random_spectra_two);
  gettimeofday(&endRoutine, NULL);
  Timer = (endRoutine.tv_sec - startRoutine.tv_sec) + 1e-6 * (endRoutine.tv_usec - startRoutine.tv_usec);
  routine_add_time += Timer;
  POP_RANGE
//  cout << "Done with mean_of_mean with T[secs] = " << Timer << endl;

#if(_OPENACC)
#pragma acc exit data delete (random_spectra_one[0:nchans], random_spectra_two[0:nchans], \
    spectra_mean[0:nsamp], spectra_var[0:nsamp])
//#pragma acc update device(stage.dptr[0:stage.size])
#endif

  PUSH_RANGE("updateIpBuffer",7)
  gettimeofday(&startRoutine, NULL);
  update_input_buffer(nsamp, nchans, input_buffer, stage, var_rescale, mean_rescale);
  gettimeofday(&endRoutine, NULL);
  Timer = (endRoutine.tv_sec - startRoutine.tv_sec) + 1e-6 * (endRoutine.tv_usec - startRoutine.tv_usec);
  routine_add_time += Timer;
  POP_RANGE
//  cout << "Done with Ip Buffer Update with T[secs] = " << Timer << endl;

  cout << "Only routine addition timings = " << routine_add_time << endl;

  gettimeofday(&endTimer, NULL);
  elapsedTimer = (endTimer.tv_sec - startTimer.tv_sec) + 1e-6 * (endTimer.tv_usec - startTimer.tv_usec);


//  write_output_file(nchans, nsamp, file_reducer, orig_mean, orig_var, sigma_cut, var_rescale, mean_rescale, stage);


	free(chan_mask);
//	free(spectra_mask);
	free(chan_mean);
	free(chan_var);
	free(spectra_mean);
	free(spectra_var);
}

/******************************************************************************************************************************/

void rfi(int nsamp, int nchans, unsigned short **input_buffer, double& elapsedTimer)
{
	int file_reducer = 1;
	float	sigma_cut	= 2.0f;

	float *stage = ( float* )safe_malloc( ( size_t ) nsamp * ( size_t )nchans * sizeof( float ) );

	int cn = 0;

  timeval startTimer, endTimer;
  gettimeofday(&startTimer, NULL);
	for( int c = 0; c < nchans; c++ )
	{
		for( int t = 0; t < nsamp; t++ )
		{
			stage[ c * ( size_t )nsamp + t ] = ( float ) ( *input_buffer )[ c  + ( size_t )nchans * t ];
		}
	}


	// ~~~ RFI Correct ~~~ //

	double orig_mean = 0.0;
	double orig_var=0.0;

	// Find the mean and SD of the input data (we'll use this to rescale the data at the end of the process.

  for( int c = 0; c < nchans; c++ )
	{
 	        for( int t = 0; t < ( nsamp ); t++)
						orig_mean+=stage[ c * ( size_t )nsamp + t ];
 	}

	orig_mean /= ( nsamp * nchans );

  for( int c = 0; c < nchans; c++ )
	{
		for( int t = 0; t < ( nsamp ); t++ )
		{
			orig_var += ( stage[ c * ( size_t )nsamp + t ] - orig_mean ) * ( stage[ c * ( size_t )nsamp + t ] - orig_mean );
		}
	}
	orig_var /= ( nsamp * nchans );
	orig_var = sqrt( orig_var );

	//printf( "orig_mean %f\n", orig_mean );
	//printf( "orig_var %f\n", orig_var );

	// Random Vectors

	float *random_chan_one = ( float* )safe_malloc( nsamp * sizeof( float ) );
	float *random_chan_two = ( float* )safe_malloc( nsamp * sizeof( float ) );

	for( int t = 0; t < nsamp; t++ )
	{
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

	float *random_spectra_one = ( float* )safe_malloc( nchans * sizeof( float ) );
	float *random_spectra_two = ( float* )safe_malloc( nchans * sizeof( float ) );

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

	// Allocate working arrays

	int *chan_mask = ( int* )safe_malloc( nchans * sizeof( int ) );
	for( int c = 0; c < nchans; c++ ) chan_mask[ c ] = 1;

	int *spectra_mask = ( int* )safe_malloc( nsamp * sizeof( int ) );
	for( int t = 0; t < nsamp; t++ ) spectra_mask[ t ] = 1;

	double *chan_mean = ( double* )safe_malloc( nchans * sizeof( double ) );
  for( int c = 0; c < nchans; c++ ) chan_mean[ c ] = 0.0;

	double *chan_var = ( double* )safe_malloc( nsamp * sizeof( double ) );
  for( int c = 0; c < nchans; c++ ) chan_var[ c ] = 0.0;

	double *spectra_mean = ( double* )safe_malloc( nsamp * sizeof( double ) );
  for( int t = 0; t < nsamp; t++ ) spectra_mean[ t ] = 0.0;

	double *spectra_var = ( double* )safe_malloc( nsamp * sizeof( double ) );
  for( int t = 0; t < nsamp; t++ ) spectra_var[ t ] = 0.0;

	// Find the BLN and try to flatten the input data per channel (remove non-stationary component).

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
	        chan_mean[ c ] += stage[ c * ( size_t )nsamp + t ];
					counter++;
				}
      }

			//printf( "\nchan_mean %lf\n", chan_mean[ c ] );
			//printf( "\ncounter%u\n", counter );

			if( counter == 0 )
			{
				//printf( "\nCounter zero, Channel %d", c );
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
					chan_var[ c ] += ( stage[ c * (size_t)nsamp + t ] - chan_mean[ c ] ) * ( stage[ c * (size_t)nsamp + t ] - chan_mean[ c ] );
					counter++;
				}
			}

			//printf( "\nchan_var %lf\n", chan_var[c] );
			//printf( "\ncounter %u\n", counter );

			chan_var[ c ] /= ( counter );
			chan_var[ c ] = sqrt( chan_var[ c ] );

			if( ( chan_var[ c ] ) * 1000000.0 < 0.1 )
			{
				////printf("\nVarience zero, Channel %d %d %lf %.16lf\n", c, rounds, chan_mean[c], chan_var[c] );
				chan_mask[ c ] = 0;
				finish = 1;
				break;
			}

			for( int t = 0; t < ( nsamp ); t++ )
			{
				if( ( ( stage[ c * ( size_t )nsamp + t ] - chan_mean[ c ] ) / chan_var[ c ] ) > sigma_cut || ( ( stage[ c * ( size_t )nsamp + t ] - chan_mean[ c ] ) / chan_var[ c ] ) < -sigma_cut )
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
				stage[ c * ( size_t )nsamp + t ] = ( stage[ c * ( size_t )nsamp + t ] - ( float )chan_mean[ c ] ) / ( float )chan_var[ c ];
			}
		}
		else
		{
			int perm_one = ( int )( ( ( float )rand_local() / ( float )RAND_MAX ) * nsamp );

			for( int t = 0; t < nsamp; t++ )
			{
				stage[ c * ( size_t )nsamp + t ] = random_chan_one[ ( t + perm_one ) % nsamp ];
			}

			chan_mean[ c ] = 0.0;
			chan_var[ c ]  = 1.0;
			chan_mask[ c ] = 1;
		}
	}

	// Find the BLN and try to flatten the input data per spectra (remove non-stationary component).

	for( int t = 0; t < nsamp; t++ )
	{
		int counter = 0;

		for( int c = 0; c < nchans; c++ )
		 chan_mask[ c ]=1;

		int finish = 0;
		int rounds = 1;

		double old_mean = 0.0;
		double old_var = 0.0;

		while( finish == 0 )
		{
			counter = 0;
			spectra_mean[ t ] = 0.0;
			for( int c = 0; c < nchans; c++ )
			{
				if( chan_mask[ c ] == 1 )
				{
	        spectra_mean[ t ] += stage[ c * ( size_t )nsamp + t ];
					counter++;
				}
      }

			//printf( "\nSpectra mean %lf\n", spectra_mean[ t ] );
			//printf( "counter %d\n", counter );

			if( counter == 0 )
			{
				//printf( "\nCounter zero, Spectra %d", t );
				spectra_mask[ t ] = 0;
				finish = 1;
				break;
			}

			spectra_mean[ t ] /= (counter);

			counter = 0;
			spectra_var[ t ] = 0.0;
			for( int c = 0; c < nchans; c++ )
			{
				if( chan_mask[ c ] == 1 )
				{
					spectra_var[ t ] += ( stage[ c * ( size_t )nsamp + t ] - spectra_mean[ t ] ) * ( stage[ c * ( size_t )nsamp + t ] - spectra_mean[ t ] );
					counter++;
				}
			}

			//printf( "spectra_var %lf\n", spectra_var[ t ] );
			//printf( "counter %u\n", counter );

			spectra_var[ t ] /= (counter);
			spectra_var[ t ] = sqrt( spectra_var[ t ] );

			if( ( spectra_var[ t ] ) * 1000000.0 < 0.1 )
			{
				////printf("\nVarience zero, Spectra %d %d %lf %.16lf", t, rounds, spectra_mean[t], spectra_var[t] );
				spectra_mask[ t ] = 0;
				finish = 1;
				break;
			}

			if( spectra_mask[ t ] != 0 )
			{
				for( int c = 0; c < nchans; c++ )
				{
					if( ( ( stage[ c * (size_t)nsamp + t ] - spectra_mean[ t ] ) / spectra_var[ t ] ) > sigma_cut || ( ( stage[ c * (size_t)nsamp + t ] - spectra_mean[ t ] ) / spectra_var[ t ] ) < -sigma_cut)
					{
						chan_mask[ c ] = 0;
					}
					else
					{
						chan_mask[ c ] = 1;
					}
				}
			}

			if( fabs( spectra_mean[ t ] - old_mean ) < 0.001 && fabs( spectra_var[ t ] - old_var ) < 0.0001 && rounds > 1)
			{
				////printf("\n%d\t%d\t%.16lf\t%.16lf\t%.16lf\t%.16lf", t, rounds, (spectra_mean[t] - old_mean), (spectra_var[t] - old_var), spectra_mean[t], spectra_var[t]);
				finish = 1;
			}

			old_mean = spectra_mean[ t ];
			old_var = spectra_var[ t ];
			rounds++;
		}

		//return;
		////printf("Spectra mean, var: %lf %d\n", spectra_mean[t], spectra_var[t] );

		if( spectra_mask[ t ] != 0)
		{
			for( int c = 0; c < nchans; c++ )
			{
				stage[ c * (size_t)nsamp + t ] = ( stage[ c * (size_t)nsamp + t ] - (float)spectra_mean[ t ] ) / (float)spectra_var[ t ];
			}
		}
		else
		{
			int perm_one = (int)( ( (float)rand_local() / (float)RAND_MAX ) * nchans);

			for( int c = 0; c < nchans; c++ )
			{
				stage[ c * (size_t)nsamp + t ] = random_spectra_one[ ( c + perm_one ) % nchans ];
			}

			spectra_mean[ t ] = 0.0;
			spectra_var[ t ]  = 1.0;
			spectra_mask[ t ] = 1;
		}
	}

	double mean_rescale = 0.0;
	double var_rescale  = 0.0;

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

	//printf("\n0 %lf %lf", mean_of_mean, var_of_mean);
	//printf("\n0 %lf %lf", mean_of_var,  var_of_var);

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
				stage[ ( c * (size_t)nsamp + t) ] = random_chan_two[ ( t + perm_one ) % nsamp ];
			}
		}
	}

	finish = 0;
	rounds = 1;
	counter = 0;

	mean_of_mean = 0.0;
	var_of_mean  = 0.0;
	mean_of_var  = 0.0;
	var_of_var   = 0.0;

	old_mean_of_mean = 0.0;
	old_var_of_mean  = 0.0;
	old_mean_of_var  = 0.0;
	old_var_of_var   = 0.0;

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

	clipping_constant = 0.0;
	for( int t = 0; t < nsamp; t++ ) clipping_constant += spectra_mask[ t ];
	clipping_constant = ( nsamp - clipping_constant ) / nsamp;
	clipping_constant = sqrt( -2.0 * log( clipping_constant * 2.506628275 ) );

	// Perform spectral replacement
	for( int t = 0; t < (nsamp); t++ )
	{
	    if( fabs( ( spectra_mean[ t ] - mean_of_mean ) / var_of_mean ) > clipping_constant && fabs( ( spectra_var[ t ] - mean_of_var ) / var_of_var ) > clipping_constant )
			{
				////printf("\nReplacing Spectral %d %lf %lf", t, spectra_mean[t], spectra_var[t]);
				int perm_one = (int)( ( (float)rand_local() / (float)RAND_MAX) * nchans );
				for( int c = 0; c < nchans; c++ )
				{
					stage[ c * (size_t)nsamp + t ] = random_spectra_two[ ( c + perm_one ) % nchans ];
				}
     }
	}


	for( int c = 0; c < nchans; c++ )
	{
		for( int t = 0; t < (nsamp); t++ )
		{
			//(*input_buffer)[c  + (size_t)nchans * t] = (unsigned char) ((stage[c * (size_t)nsamp + t]*orig_var)+orig_mean);
			(*input_buffer)[ c  + (size_t)nchans * t ] = (unsigned char) ( ( stage[ c * (size_t)nsamp + t ] * var_rescale ) + mean_rescale );
		}
	}
  gettimeofday(&endTimer, NULL);
  elapsedTimer = (endTimer.tv_sec - startTimer.tv_sec) + 1e-6 * (endTimer.tv_usec - startTimer.tv_usec);


	FILE *fp_mask = fopen ("masked_chans.txt", "w+");

	for( int c = 0; c < nchans; c++ )
	{
		for( int t = 0; t < (nsamp) / file_reducer; t++ )
		{
			fprintf(fp_mask, "%d ", (unsigned char)((stage[c * (size_t)nsamp + t]*orig_var)+orig_mean));
			fprintf( fp_mask, "%d ", (unsigned char)( ( stage[ c * (size_t)nsamp + t] * var_rescale ) + mean_rescale ) );
		}

		fprintf(fp_mask, "\n");
	}
  fclose(fp_mask);

	//printf("\n%lf %lf", mean_rescale / orig_mean, var_rescale / orig_var);


	free(chan_mask);
	free(spectra_mask);
	free(chan_mean);
	free(chan_var);
	free(spectra_mean);
	free(spectra_var);
	free(stage);
}

void readInputBuffer(unsigned short *input_buffer, int nsamp, int nchans, char fname[100])
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
  unsigned long long in_buffer_bytes = nsamp * nchans * sizeof( unsigned short );
  unsigned short *input_buffer = ( unsigned short* )safe_malloc( in_buffer_bytes );

  //Total Time of the application
  gettimeofday(&startReadTimer, NULL);
	char fname[100] = "input_buffer.txt";

  PUSH_RANGE("readInputBuffer",8)
	readInputBuffer(input_buffer, nsamp, nchans, fname);
  POP_RANGE

  gettimeofday(&endReadTimer, NULL);
  double elapsedReadTimer = (endReadTimer.tv_sec - startReadTimer.tv_sec) + 1e-6 * (endReadTimer.tv_usec - startReadTimer.tv_usec);

  printf("DONE READING with nchan = %d\t nsamp = %d\t time taken = %f !!!!!\n", nchans, nsamp, elapsedReadTimer);

  double elapsedRFITimer = 0.0;
  gettimeofday(&startRFITimer, NULL);
//  rfi_debug(nsamp, nchans, &input_buffer, elapsedRFITimer);
//  rfi(nsamp, nchans,  &input_buffer, elapsedRFITimer);

  rfi_debug3(nsamp, nchans,  input_buffer, elapsedRFITimer);

  //Write output in a different routine and then free the stage storage
  printf("RFI routine time = %f !!!!!\n", elapsedRFITimer);

  gettimeofday(&endTimer, NULL);
  double elapsedTimer = (endTimer.tv_sec - startTimer.tv_sec) + 1e-6 * (endTimer.tv_usec - startTimer.tv_usec);

  printf("Total time = %f\n", elapsedTimer);

  return 0;
}
