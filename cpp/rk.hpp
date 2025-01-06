#ifndef FILE_RK
#define FILE_RK

#include <solve.hpp>
using namespace ngsolve;
#include <python_ngstd.hpp>

#include <iostream>
#include <cmath>
#include <cfloat>


template <class T, class APP>
class ExplicitRK {

  /* This class implements some explicit Runge-Kutta methods for
     solving an ODE system (IVP). These methods have a Butcher tableau
     with C[0] = 0 and lower triangular A: 

            0    |
	   C[1]  | A[1,0]
	   C[2]  | A[2,0]    A[2,1]
	    :    | :
	    :    | :
	  C[s-1] | A[s-1,0]  A[s-1,1]  ...  A[s-1, s-2]
	  -------|--------------------------------------
	         | B[0]      B[1]      ...  B[s-1]

     Reference: https://en.wikipedia.org/wiki/Runge–Kutta_methods.

     To follow this notation, we will set C and B as Vectors,
     while A is set to a Table (with first row length 0).
     An s-stage Runge-Kutta method does the following:

          Y <- Y + h (B₀ K₀ + B₁ K₁ + ... + Bₛ₋₁ Kₛ₋₁),

       where Kᵢ's are vectors defined by

       K₀  = F(t, Y)
       K₁  = F(t + C₁ h, Y + h A₁₀ K₀)
       K₂  = F(t + C₂ h, Y + h (A₂₀ K₀ + A₂₁ K₁))
       :
       :
       :
       i.e., we have

         Kᵢ = F(t + Cᵢ h, Y + h (Aᵢ₀ K₀ + Aᵢ₁ K₁ + ... + Aᵢ,ᵢ₋₁ Kᵢ₋₁)),

       for i = 0, 1, ..., s-1.

  */

protected:

  int _s;                // number of stages
  int _n;                // number of equations in ODE system
  double _h;             // step size

  Vector<> * _B;         // Butcher tableau:     C  |  A
                         //                      ---.-----
  Vector<> * _C;	 //                         |  B^T
  Table<double>  * _A;

  Matrix<T> _K;          // Workspace for intermediate quantities

  
  void inline
  Step(double t, FlatVector<T> Y)  {

    /* Do one RK step with private step size _h (not given as input; see below
       for public member functions that take step size as input).

       Specifically, we implement the formula

       Y <- Y + h (B₀ K₀ + B₁ K₁ + ... + Bₛ₋₁ Kₛ₋₁),

       where Kᵢ's are vectors defined by

       K₀  = F(t, Y)
       K₁  = F(t + C₁ h, Y + h A₁₀ K₀)
       K₂  = F(t + C₂ h, Y + h (A₂₀ K₀ + A₂₁ K₁))
       :
       :

       i.e., we have

       Kᵢ = F(t + Cᵢ h, Y + h (Aᵢ₀ K₀ + Aᵢ₁ K₁ + ... + Aᵢ,ᵢ₋₁ Kᵢ₋₁)),

       for i = 0, 1, ..., s-1. We implement this using the workspace
       member _K thus avoiding new memory allocation.
    */

    for (int i=0; i<_s; i++) {  // K₀, K₁, ..., Kₛ₋₁ will be made here

      // _Kᵢ <– Y
      _K.Row(i) = Y;

      for (int j=0; j<i; j++)
        // Kᵢ <-  h * (Aᵢ₀ K₀ + Aᵢ₁ K₁ + ... + Aᵢ,ᵢ₋₁ Kᵢ₋₁)
        _K.Row(i) += _h * (*_A)[i][j] * _K.Row(j);

      // Kᵢ <-  F(t + Cᵢ h, Kᵢ)
      F(t + (*_C)[i]*_h, _K.Row(i));
    }

    // Y <-  Y + h ∑ᵢ Bᵢ Kᵢ
    for (int i=0; i<_s; i++)
      Y += _h * (*_B)[i] * _K.Row(i);
  }

public:

  ExplicitRK(int s, int n) : _K(s, n) {

    // Construct a pre-defined RK method with s stages

    _s = s;
    _h = 0.0;
    Array<int> rowsizes(_s);
    for (int i=0; i<_s; i++) rowsizes[i]=i;
    _A = new Table<double>(rowsizes);
    vector<vector<double>> A;

    switch(_s) {

    case 1: { // Explicit Euler

      /*  Butcher Tableau:

    	  0 |
    	  ------
    	    | 1
       */

      _C = new Vector<>({0.0});
      _B = new Vector<>({1.0});

      break;
    }
    case 2: { // Midpoint method

      /* Butcher Tableau:

    	    0 |
    	  1/2 | 1/2
    	  -------------
    	      | 0    1
       */

      _C = new Vector<>({0.0, 0.5});
      _B = new Vector<>({0.0, 1.0});
      A = {{0.5}};
      break;
    }

    case 4: { // RK4 method

      /* Butcher Tableau:

    	     0   |
	     1/2 |	1/2
	     1/2 |	0	1/2
	     1   |	0	0	1
	 ----------------------------------
	         | 1/6	1/3	1/3	1/6

       */

      _C = new Vector<>({0.,    1./2., 1./2., 1.   });
      _B = new Vector<>({1./6., 1./3., 1./3., 1./6.});
      A = {{0.5},
           {0.0,  0.5},
           {0.0,  0.0,   1.0}};
      break;
    }

    case 7: { // Dormand-Prince method

      /* Butcher Tableau:

   0    |
   1/5  | 1/5
   3/10 | 3/40       9/40
   4/5  | 44/45      -56/15      32/9
   8/9  | 19372/6561 -25360/2187 64448/6561 -212/729
   1    | 9017/3168  -355/33     46732/5247 49/176   -5103/18656
   1    | 35/384     0           500/1113   125/192  -2187/6784    11/84
   ----------------------------------------------------------------------------
        | 35/384     0           500/1113   125/192  -2187/6784    11/84    0

       */

      _C = new Vector<>({0.      , 1./5., 3./10.    , 4./5.    , 8./9.       , 1.     , 1. });
      _B = new Vector<>({35./384., 0.   , 500./1113., 125./192., -2187./6784., 11./84., 0. });
      A = {{1./5. },
           {3./40.      , 9./40.       },
           {44./45.     , -56./15.     , 32./9.     },
           {19372./6561., -25360./2187., 64448./6561. , -212./729. },
           {9017./3168. , -355./33.    , 46732./5247. , 49./176.    , -5103./18656. },
           {35./384.    , 0.           , 500./1113.   , 125./192.   , -2187./6784.   , 11./84.}};
      break;
    }

    default:
      cout << "No " << _s << " stage RK case implemented!" << endl;
      break;
    }

    for (int r=1; r <_s; r++)
      for (int c=0; c < rowsizes[r]; c++)
        (*_A)[r][c] = A[r-1][c];
  }

  inline APP &
  FromApplication() { return static_cast<APP &>(*this); }

  inline void
  F(double z, FlatVector<T> Y) { return FromApplication().F(z, Y); }

  inline void
  SolveIVP(Vector<T> & Y, double t, double h, int numsteps)  {

    // With initial value at time t given by input Y = Y(t),
    // solve IVP and output solution at time t + numsteps * h.

    _h = h;
    for (int i=0; i<numsteps; i++)
      Step(t+i*_h, Y);
  }

  inline void
  SolveFlow(Matrix<T> & Y, double t0, double h, int numsteps)  {

    // With initial value at time t0 given by input Y.Row(0),
    // solve IVP and output solution in remaining rows of the
    // matrix Y such that  Y.Row(i) = solution at time t + i * h.

    _h = h;

    for (int i=0; i<numsteps-1; i++) {

      Y.Row(i+1) = Y.Row(i);

      Step(t0+i*h, Y.Row(i+1));
    }
  }

  void Print() const {
    cout << "Explicit RK method of " << _s
         << " stages with the following Butcher tableau:" << endl;
    cout << "A:" << endl << *_A << endl;
    cout << "B:" << endl << *_B << endl;
    cout << "C:" << endl << *_C << endl;
  }

  ~ExplicitRK() {

    delete _A;
    delete _B;
    delete _C;
  }

};


// A simple application class  (used in rk_tests)

template <class T>
class App: public ExplicitRK<T, App<T>> {

  function<void(double, FlatVector<T>)> _f;

public:

  App(function<void(double, FlatVector<T>)> ff, int s, int n)
    :  ExplicitRK<T, App<T>>(s, n) { _f = ff; }
  
  inline void F(double z, FlatVector<T> Y) { return _f(z, Y); }

};


// Explicit instantiation for complex vector ODE systems

template class App<Complex>;
typedef App<Complex> RKC;


// See rk_tests
extern bool RunRKTests();

#endif // FILE_RK
