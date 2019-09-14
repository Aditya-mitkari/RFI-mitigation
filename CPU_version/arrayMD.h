#ifndef RFI_ARRAYMD_H
#define RFI_ARRAYMD_H

/* ----------------------------------------------------------------
 * A simple FORTRAN style multi-dimensional 2D array created for
 * RFI code of GPUFreeLoaders.
 * The operator (i,j) will give access to the j-th element in the i-th row
 *
 *Rahulkumar Gayatri NERSC,LBNL.
 *----------------------------------------------------------------*/

template<typename T> class Array2D
{
   public:
     //n1 - first dimenion, n2 - second dimension, size = n1*n2, total number of elements
     unsigned n1, n2;
     unsigned size;
     T * dptr;

     //Row major access
     inline T& operator() (unsigned i1, unsigned i2)
     {
//       return dptr[i1 + (n1*i2)]; //column-major
       return dptr[i2 + (n2*i1)]; //Row-major
     }

     //Default constructor
     Array2D() { n1=n2= 0; size=0; dptr=NULL; }

     //Copy Constructor
     Array2D(const Array2D &p) { n1=p.n1; n2=p.n2; size=0; dptr=p.dptr; }

     //Create a 2D array of in1 and in2 size
     Array2D(int in1, int in2)
     {
       n1=in1; n2=in2; size=n1*n2;
       dptr=(T*)malloc(size*sizeof(T));
       dptr = new T[size];
     }

     //Destructor
     ~Array2D() { if (size && dptr) delete(dptr); }

     //If the array size is known later
     void resize(unsigned in1,unsigned in2)
     {
       if (size && dptr) delete(dptr);
       n1=in1; n2=in2; size=n1*n2;
       dptr = new T[size];
     }

     unsigned getSize() { return size * sizeof(T); }  // NB: in bytes
};

#endif
