/////////////////////////////////////////////////////////
// Auther: Aditya Mitkari
// Date: 4/5/19
//
// Description: Serial code for RFI mitigation
// Input: file / array of frequecy values for given DM
// Ouput: Frequecy values free of RFI
/////////////////////////////////////////////////////////

#include <stdio.h>
#include <stdlib.h>
//#include "headers/params.h"

void rfi(int nsamp, int nchans, unsigned short **input_buffer)
{
	int file_reducer = 1;
	float	sigma_cut	= 2.0f;

	float *stage = ( float* )malloc( ( size_t ) nsamp * ( size_t )nchans * sizeof( float ) );

	int cn = 0;
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

	printf( "orig_mean %f\n", orig_mean );
	printf( "orig_var %f\n", orig_var );

	// Random Vectors

	float *random_chan_one = ( float* )malloc( nsamp * sizeof( float ) );
	float *random_chan_two = ( float* )malloc( nsamp * sizeof( float ) );

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

	float *random_spectra_one = ( float* )malloc( nchans * sizeof( float ) );
	float *random_spectra_two = ( float* )malloc( nchans * sizeof( float ) );

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

	int *chan_mask = ( int* )malloc( nchans * sizeof( int ) );
	for( int c = 0; c < nchans; c++ ) chan_mask[ c ] = 1;

	int *spectra_mask = ( int* )malloc( nsamp * sizeof( int ) );
	for( int t = 0; t < nsamp; t++ ) spectra_mask[ t ] = 1;

	double *chan_mean = ( double* )malloc( nchans * sizeof( double ) );
  for( int c = 0; c < nchans; c++ ) chan_mean[ c ] = 0.0;

	double *chan_var = ( double* )malloc( nsamp * sizeof( double ) );
  for( int c = 0; c < nchans; c++ ) chan_var[ c ] = 0.0;

	double *spectra_mean = ( double* )malloc( nsamp * sizeof( double ) );
  for( int t = 0; t < nsamp; t++ ) spectra_mean[ t ] = 0.0;

	double *spectra_var = ( double* )malloc( nsamp * sizeof( double ) );
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

			printf( "\nchan_mean %lf\n", chan_mean[ c ] );
			printf( "\ncounter%u\n", counter );

			if( counter == 0 )
			{
				printf( "\nCounter zero, Channel %d", c );
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

			printf( "\nchan_var %lf\n", chan_var[c] );
			printf( "\ncounter %u\n", counter );

			chan_var[ c ] /= ( counter );
			chan_var[ c ] = sqrt( chan_var[ c ] );

			if( ( chan_var[ c ] ) * 1000000.0 < 0.1 )
			{
				//printf("\nVarience zero, Channel %d %d %lf %.16lf\n", c, rounds, chan_mean[c], chan_var[c] );
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
				//printf("\n%d\t%d\t%.16lf\t%.16lf\t%.16lf\t%.16lf", c, rounds, (chan_mean[c]-old_mean), (chan_var[c]-old_var), chan_mean[c], chan_var[c]);
				finish = 1;
			}

			old_mean = chan_mean[ c ];
			old_var = chan_var[ c ];
			rounds++;
		}


		printf( "\nChan mean, var: %lf %lf\n", chan_mean[ c ], chan_var[ c ] );


		if( chan_mask[ c ] != 0 )
		{
			for( int t = 0; t < ( nsamp ); t++ )
			{
				stage[ c * ( size_t )nsamp + t ] = ( stage[ c * ( size_t )nsamp + t ] - ( float )chan_mean[ c ] ) / ( float )chan_var[ c ];
			}
		}
		else
		{
			int perm_one = ( int )( ( ( float )rand() / ( float )RAND_MAX ) * nsamp );

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

	unsigned int cnt = 0;

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
			cnt += 1;
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

			printf( "\nSpectra mean %lf\n", spectra_mean[ t ] );
			printf( "counter %d\n", counter );

			if( counter == 0 )
			{
				printf( "\nCounter zero, Spectra %d", t );
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

			printf( "spectra_var %lf\n", spectra_var[ t ] );
			printf( "counter %u\n", counter );

			spectra_var[ t ] /= (counter);
			spectra_var[ t ] = sqrt( spectra_var[ t ] );

			if( ( spectra_var[ t ] ) * 1000000.0 < 0.1 )
			{
				//printf("\nVarience zero, Spectra %d %d %lf %.16lf", t, rounds, spectra_mean[t], spectra_var[t] );
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
				//printf("\n%d\t%d\t%.16lf\t%.16lf\t%.16lf\t%.16lf", t, rounds, (spectra_mean[t] - old_mean), (spectra_var[t] - old_var), spectra_mean[t], spectra_var[t]);
				finish = 1;
			}

			old_mean = spectra_mean[ t ];
			old_var = spectra_var[ t ];
			rounds++;
		}

		//return;
		//printf("Spectra mean, var: %lf %d\n", spectra_mean[t], spectra_var[t] );

		if( spectra_mask[ t ] != 0)
		{
			for( int c = 0; c < nchans; c++ )
			{
				stage[ c * (size_t)nsamp + t ] = ( stage[ c * (size_t)nsamp + t ] - (float)spectra_mean[ t ] ) / (float)spectra_var[ t ];
			}
		}
		else
		{
			int perm_one = (int)( ( (float)rand() / (float)RAND_MAX ) * nchans);

			for( int c = 0; c < nchans; c++ )
			{
				stage[ c * (size_t)nsamp + t ] = random_spectra_one[ ( c + perm_one ) % nchans ];
			}

			spectra_mean[ t ] = 0.0;
			spectra_var[ t ]  = 1.0;
			spectra_mask[ t ] = 1;
		}
	}

	printf( "cnt is %u\n", cnt );


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

		printf("mm is %lf\n",mean_of_mean );

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

		printf( "\nvar_of_mean %lf\n", var_of_var );
    printf( "\ncounter %u\n", counter );

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

		printf("\nmean_of_var %lf\n",mean_of_var );
    printf("\ncounter %u\n",counter );

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

		printf("\nvar_of_var %lf\n",var_of_var );
    printf("\ncounter %u\n",counter);

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

	printf("\n0 %lf %lf", mean_of_mean, var_of_mean);
	printf("\n0 %lf %lf", mean_of_var,  var_of_var);

	mean_rescale = mean_of_mean;
	var_rescale  = mean_of_var;

	float clipping_constant = 0.0;

	for( int c = 0; c < nchans; c++ ) clipping_constant += chan_mask[ c ];
	clipping_constant = ( nchans - clipping_constant ) / nchans;
	printf("\n clipping_constant is %f\n",clipping_constant );
	clipping_constant = sqrt( -2.0 * log( clipping_constant * 2.506628275 ) );
	printf("This This %f\n",clipping_constant );

	// Perform channel replacement
	for( int c = 0; c < nchans; c++ )
	{
		if( fabs( ( chan_mean[ c ] - mean_of_mean ) / var_of_mean ) > clipping_constant && fabs( ( chan_var[ c ] - mean_of_var ) / var_of_var ) > clipping_constant )
		{
			//printf("\nReplacing Channel %d %lf %lf", c, chan_mean[c], chan_var[c]);
			int perm_one = (int)( ( (float)rand() / (float)RAND_MAX ) * nsamp );

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

	printf("\n0 %lf %lf", mean_of_mean, var_of_mean);
	printf("\n0 %lf %lf", mean_of_var,  var_of_var);

	clipping_constant = 0.0;
	for( int t = 0; t < nsamp; t++ ) clipping_constant += spectra_mask[ t ];
	clipping_constant = ( nsamp - clipping_constant ) / nsamp;
	clipping_constant = sqrt( -2.0 * log( clipping_constant * 2.506628275 ) );

	// Perform spectral replacement
	for( int t = 0; t < (nsamp); t++ )
	{
	    if( fabs( ( spectra_mean[ t ] - mean_of_mean ) / var_of_mean ) > clipping_constant && fabs( ( spectra_var[ t ] - mean_of_var ) / var_of_var ) > clipping_constant )
			{
				//printf("\nReplacing Spectral %d %lf %lf", t, spectra_mean[t], spectra_var[t]);
				int perm_one = (int)( ( (float)rand() / (float)RAND_MAX) * nchans );
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

	FILE *fp_mask = fopen ("masked_chans.txt", "w+");

	for( int c = 0; c < nchans; c++ )
	{
		for( int t = 0; t < (nsamp) / file_reducer; t++ )
		{
			//fprintf(fp_mask, "%d ", (unsigned char)((stage[c * (size_t)nsamp + t]*orig_var)+orig_mean));
			fprintf( fp_mask, "%d ", (unsigned char)( ( stage[ c * (size_t)nsamp + t] * var_rescale ) + mean_rescale ) );
		}

		fprintf(fp_mask, "\n");
	}
  fclose(fp_mask);

	printf("\n%lf %lf", mean_rescale / orig_mean, var_rescale / orig_var);


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
	printf("in readInputBuffer %s\n", fname );

	FILE *fp_inputBuffer = fopen( fname, "r" );

	if( fp_inputBuffer == NULL )
	{
		printf("Error opeing file\n" );
		exit(0);
	}

  for( int i=0; i < nsamp * nchans; i++ )
	{
    fscanf(fp_inputBuffer, "%hu", &input_buffer[ i ] );
		//printf("%hu  ",input_buffer[i] );
	}
	printf("done reading\n" );
	fclose( fp_inputBuffer );
}

int main()
{
  int nsamp = 491522;//10977282;
  int nchans = 4096;
  unsigned long long in_buffer_bytes = nsamp * nchans * sizeof( unsigned short );
  unsigned short *in_buffer = ( unsigned short* )malloc( in_buffer_bytes );
  unsigned short *input_buffer = in_buffer;

  //unsigned long long i;
  //#pragma omp parallel
  // #pragma omp parallel for
  // for(i=0; i<nsamp*nchans; i++)
  // {
  //   in_buffer[i] = 100;
  // }
	char fname[100] = "input_buffer.txt";
	readInputBuffer(in_buffer, nsamp, nchans, fname);

  rfi(nsamp, nchans, &input_buffer);


  return 0;
}
