#ifndef RFI_ARRAYMD_H
#define RFI_ARRAYMD_H

/* ----------------------------------------------------------------
 * A simple FORTRAN style multi-dimensional 2D array created for
 * RFI code of GPUFreeLoaders.
 * The operator (i,j) will give access to the j-th element in the i-th row
 *
 *Rahulkumar Gayatri NERSC,LBNL.
 *----------------------------------------------------------------*/

#include <iostream>
using namespace std;

template<typename T> class Array2D
{
   public:
    unsigned n1, n2;
    unsigned size;
    T * dptr;

    inline T& operator() (unsigned i1, unsigned i2)
    {
      return dptr[i2+(n2*i1)];
    }

    Array2D() { n1=n2= 0; size=0; dptr=NULL; }

    Array2D(const Array2D &p)
    {
      cout << "Array2D copy constructor is called" << endl;
      n1=p.n1; n2=p.n2; size=0;
      dptr=p.dptr;
    }

    Array2D(int in1, int in2)
    {
      n1=in1; n2=in2; size=n1*n2;
      dptr = new T[size];
    }

    ~Array2D() { if (size && dptr) delete(dptr); }

    void resize(unsigned in1,unsigned in2)
    {
      if (size && dptr) delete(dptr);
      n1=in1; n2=in2; size=n1*n2;
      dptr = new T[size];
    }
    void setSize(unsigned in1, unsigned in2) { n1 = in1; n2 = in2; size = n1*n2; }  // NB: in bytes
    unsigned getSize() { return size * sizeof(T); }  // NB: in bytes
};


#endif
